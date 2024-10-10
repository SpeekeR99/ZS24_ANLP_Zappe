import json
import os
import pandas as pd
import csv

from cv02.consts import EMB_FILE, TRAIN_DATA, TEST_DATA

cwd = os.getcwd()
print("Current working directory: {0}".format(cwd))

import argparse
import datetime

import pickle
import random
import sys
from collections import Counter

import wandb
from torch.optim.lr_scheduler import MultiStepLR

from matplotlib import pyplot as plt

import numpy as np
import torch as torch

with open('wandb_config.json', encoding="utf-8") as f:
    wandb_config = json.load(f)
print(f"loaded wandb_config: {wandb_config}")

WORD2IDX = "word2idx.pckl"
VECS_BUFF = "vecs.pckl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

DETERMINISTIC_SEED = 9
random.seed(DETERMINISTIC_SEED)

UNK = "<UNK>"
PAD = "<PAD>"

MAX_SEQ_LEN = 50

BATCH_SIZE = 1000
MINIBATCH_SIZE = 10

EPOCH = 7

run_id = random.randint(100_000, 999_999)


#  file_path is path to the source file for making statistics
#  top_n integer : how many most frequent words
def dataset_vocab_analysis(texts, top_n=-1):
    counter = Counter()
    # CF#1
    #  Count occurrences of words in the data set, and prepare a list of top_n words, split words using l.split(" ")
    for text in texts:
        words = text.split(" ")
        counter.update(words)

    if top_n > 0:
        return [word for word, _ in counter.most_common(top_n)]

    return list(counter)


#  emb_file : a source file with the word vectors
#  top_n_words : enumeration of top_n_words for filtering the whole word vector file
def load_ebs(emb_file, top_n_words: list, wanted_vocab_size, force_rebuild=False):
    print("prepairing W2V...", end="")
    if os.path.exists(WORD2IDX) and os.path.exists(VECS_BUFF) and not force_rebuild:
        # CF#3
        print("...loading from buffer")
        with open(WORD2IDX, 'rb') as idx_fd, open(VECS_BUFF, 'rb') as vecs_fd:
            word2idx = pickle.load(idx_fd)
            vecs = pickle.load(vecs_fd)
    else:
        print("...creating from scratch")

        if wanted_vocab_size < 0:
            wanted_vocab_size = len(top_n_words) + 2  # + 2 for UNK and PAD

        with open(emb_file, 'r', encoding="utf-8") as emb_fd:
            idx = 0
            word2idx = {}

            # CF#2
            #  create map of  word->id  of top according to the given top_n_words
            #  create a matrix as a np.array : word vectors
            #  vocabulary ids corresponds to vectors in the matrix
            #  Do not forget to add UNK and PAD tokens into the vocabulary.

            for word in top_n_words:
                word2idx[word] = idx

                idx += 1

                if idx == wanted_vocab_size - 1:
                    break

            word2idx[UNK] = idx
            word2idx[PAD] = idx + 1

            vecs = [np.zeros(300) for _ in range(wanted_vocab_size)]

            for i, line in enumerate(emb_fd):
                if i == 0:
                    continue

                parts = line.split(" ")
                word = parts[0]
                vec = np.array([float(x) for x in parts[1:]])

                if word in word2idx:
                    vecs[word2idx[word]] = vec

            # assert len(word2idx) > 6820
            # assert len(vecs) == len(word2idx)
            pickle.dump(word2idx, open(WORD2IDX, 'wb'))
            pickle.dump(vecs, open(VECS_BUFF, 'wb'))

    return word2idx, vecs


# This class is used for transforming text into sequence of ids coresponding to word vectors (using dict word2idx).
# It also counts some usable statistics.
class MySentenceVectorizer():
    def __init__(self, word2idx, max_seq_len):
        self._all_words = 0
        self._out_of_vocab = 0
        self.word2idx = word2idx
        self.max_seq_len = max_seq_len

    def sent2idx(self, sentence):
        idx = []
        # todo CF#4
        #  Transform sentence into sequence of ids using self.word2idx
        #  Keep the counters self._all_words and self._out_of_vocab up to date
        #  for checking coverage -- it is also used for testing.


        return idx

    def out_of_vocab_perc(self):
        return (self._out_of_vocab / self._all_words) * 100

    def reset_counter(self):
        self._out_of_vocab = 0
        self._all_words = 0


# Load and preprocess the data from file.
class DataLoader():
    # vectorizer : MySentenceVectorizer
    def __init__(self, vectorizer, data_file_path, batch_size):
        self._data_folder = data_file_path
        self._batch_size = batch_size
        self.a = []
        self.b = []
        self.sts = []
        self.pointer = 0
        self._vectorizer = vectorizer
        print(f"loading data from {self._data_folder} ...", end="")
        self.__load_from_file(self._data_folder)

        self.out_of_vocab = self._vectorizer.out_of_vocab_perc()
        self._vectorizer.reset_counter()

    def __load_from_file(self, file):
        # todo CF#5
        #  load and preprocess the data set from file into self.a self.b self.sts
        #  use vectorizer to store only ids instead of strings
        with open(file, 'r', encoding="utf-8") as fd:
            for i, l in enumerate(fd):
                pass


                # You can use this snippet for faster debuging
                # if i == 4000:
                #     break


    def __iter__(self):
        # todo CF#7
        #   randomly shuffle data in memory and start from begining

        self.pointer = 0

        return self

    def __next__(self):
        # todo CF#6
        #   Implement yielding a batches from preloaded data: self.a,  self.b, self.sts
        batch = dict()


        return batch


class TwoTowerModel(torch.nn.Module):
    def __init__(self, vecs, final_metric):
        super(TwoTowerModel, self).__init__()
        # todo # CF#10a
        #   Initialize building block for architecture described in the assignment
        self.final_metric = final_metric

        # torch.nn.Embedding
        # torch.nn.Linear


        self.emb_layer = None
        self.emb_proj = None

        print("requires grads? : ", self.emb_layer.weight.requires_grad)
        self.relu = None

        self.final_proj_1 = None
        self.final_proj_2 = None


    def _make_repre(self, idx):
        avg = None
        return avg

    def forward(self, batch):
        repre_a = self._make_repre(batch['a'].to(device))
        repre_b = self._make_repre(batch['b'].to(device))

        # todo CF#10b
        #   Implement forward pass for the model architecture described in the assignment.
        #   Use both described similarity measures.
        if self.final_metric == "neural":
            repre = None
        if self.final_metric == "cos":
            repre = None

        return repre


class DummyModel(torch.nn.Module):  # predat dataset a vracet priod
    def __init__(self, file_path):
        super(DummyModel, self).__init__()
        # todo CF#9
        #   Implement DummyModel as described in the assignment.
        self.mean_on_train = None

    def forward(self, batch):
        return torch.tensor([self.mean_on_train for _ in range(len(batch['a']))]).to(device)



# todo CF#8b
# process whole dataset and return loss
# save loss from each batch and divide it by all on the end.
def test(data_set, net, loss_function):
    running_loss = 0
    all = 0
    for i, td in enumerate(data_set):
        predicted_sts = None

        loss = None
    test_loss = running_loss / all
    return test_loss


def train_model(train_dataset, test_dataset, w2v, loss_function, final_metric):
    # net = CzertModel()
    # net = net.to(device)
    net = TwoTowerModel(w2v, final_metric)
    net = net.to(device)

    optimizer = None
    lr_scheduler = None

    train_loss_arr = []
    test_loss_arr = []

    running_loss = 0.0
    sample = 0
    x_axes = []  # vis

    # todo CF#8a Implement training loop
    for epoch in range(EPOCH):
        for i, td in enumerate(train_dataset):
            batch = td
            real_sts = batch['sts'].to(device)

            predicted_sts = None

            loss = loss_function(real_sts, predicted_sts)

            running_loss += loss.item()
            sample += BATCH_SIZE
            wandb.log({"train_loss": loss, "lr": lr_scheduler.get_last_lr()}, commit=False)

            if i % MINIBATCH_SIZE == MINIBATCH_SIZE - 1:
                train_loss = running_loss / MINIBATCH_SIZE
                running_loss = 0.0

                train_loss_arr.append(train_loss)

                test_loss = test(test_dataset, net, loss_function)
                test_loss_arr.append(test_loss)

                wandb.log({"test_loss": test_loss}, commit=False)

                print(f"e{epoch} b{i}\ttrain_loss:{train_loss}\ttest_loss:{test_loss}\tlr:{lr_scheduler.get_last_lr()}")
            wandb.log({})

        lr_scheduler.step()

    print('Finished Training')
    os.makedirs("log", exist_ok=True)
    timestring = datetime.datetime.now().strftime("%b-%d-%Y--%H-%M-%S")
    plt.savefig(f"log/{run_id}-{timestring}.pdf")

    return test_loss_arr


def main(config=None):
    # wandb.init(project=wandb_config["WANDB_PROJECT"], entity=wandb_config["WANDB_ENTITY"], tags=["cv02"], config=config)

    with open(TRAIN_DATA, 'r', encoding="utf-8") as fd:
        train_data_texts = fd.read().split("\n")

    top_n_words = dataset_vocab_analysis(train_data_texts, -1)

    word2idx, word_vectors = load_ebs(EMB_FILE, top_n_words, config['vocab_size'])

    # vectorizer = MySentenceVectorizer(word2idx, MAX_SEQ_LEN)
    #
    # train_dataset = DataLoader(vectorizer, TRAIN_DATA, BATCH_SIZE)
    # test_dataset = DataLoader(vectorizer, TEST_DATA, BATCH_SIZE)
    #
    # dummy_net = DummyModel(train_dataset)
    # dummy_net = dummy_net.to(device)
    #
    # loss_function = torch.nn.MSELoss()
    #
    # test(test_dataset, dummy_net, loss_function)
    # test(train_dataset, dummy_net, loss_function)
    #
    # train_model(train_dataset, test_dataset, word_vectors, loss_function, config["final_metric"])


if __name__ == '__main__':
    my_config = {
        "vocab_size": 20000,
        "random_emb": True
    }

    print(my_config)
    main(my_config)
