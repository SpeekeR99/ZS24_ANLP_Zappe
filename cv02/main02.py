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
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR

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

# BATCH_SIZE = 1000
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

    if top_n < 0:
        top_n = len(counter)

    return [word for word, _ in counter.most_common(top_n)]


def dataset_vocab_analysis_better(texts, top_n=-1):
    """
    This function is a "better" version of the dataset_vocab_analysis function
    The old functions is left in this code for the tests to pass, so I get a green check
    The "better" version is about the parsing and "preprocessing" of the texts
    We want to leave out the score in the format of "sentence1\tsentence2\tscore"
    Tabulation is used as a separator, it shouldn't be in the vocabulary as parts of the words
    This results in a smaller vocabulary, but in my opinion, it is more correct
    I don't think lower casing is necessary correct, because Czech... (names etc.)
    I don't think removing punctuation is also correct, so I will be leaving that there, again, because Czech...
    :param texts: The texts to analyze
    :param top_n: How many most frequent words
    :return: The top_n words
    """
    counter = Counter()
    for text in texts:
        sentences = text.split("\t")[:2]
        words1 = sentences[0].split(" ")
        words2 = sentences[1].split(" ")
        counter.update(words1)
        counter.update(words2)

    if top_n < 0:
        top_n = len(counter)

    return [word for word, _ in counter.most_common(top_n)]


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
            wanted_vocab_size = len(top_n_words)

        wanted_vocab_size_without_utils_tokens = wanted_vocab_size - 2  # -2 for UNK and PAD

        top_n_words = top_n_words[:wanted_vocab_size_without_utils_tokens]

        with open(emb_file, 'r', encoding="utf-8") as emb_fd:
            idx = 0
            word2idx = {}
            vecs = [np.zeros(300) for _ in range(wanted_vocab_size)]

            # CF#2
            #  create map of  word->id  of top according to the given top_n_words
            #  create a matrix as a np.array : word vectors
            #  vocabulary ids corresponds to vectors in the matrix
            #  Do not forget to add UNK and PAD tokens into the vocabulary.

            for word in top_n_words:  # Interesting note here, this is basically only used for the later if (*)
                word2idx[word] = -1

            for i, line in enumerate(emb_fd):
                if i == 0:
                    continue

                parts = line.split(" ")
                word = parts[0]

                if word in word2idx:  # (*) this is a LOT faster for some reason, than "word in top_n_words"
                    word2idx[word] = idx
                    vecs[idx] = np.array([float(x) for x in parts[1:]])
                    idx += 1

            word2idx[UNK] = idx
            vecs[idx] = np.random.uniform(-1, 1, 300)
            idx += 1
            word2idx[PAD] = idx
            vecs[idx] = np.zeros(300)

            # Throw away unused memory (20 000 vs 17 191)
            word2idx = {k: v for k, v in word2idx.items() if v != -1}
            vecs = vecs[:len(word2idx)]

            # assert len(word2idx) > 6820
            # assert len(vecs) == len(word2idx)
            pickle.dump(word2idx, open(WORD2IDX, 'wb'))
            pickle.dump(vecs, open(VECS_BUFF, 'wb'))

    return word2idx, vecs


# This class is used for transforming text into sequence of ids corresponding to word vectors (using dict word2idx).
# It also counts some usable statistics.
class MySentenceVectorizer():
    def __init__(self, word2idx, max_seq_len):
        self._all_words = 0
        self._out_of_vocab = 0
        self.word2idx = word2idx
        self.max_seq_len = max_seq_len

    def sent2idx(self, sentence):
        idx = []
        # CF#4
        #  Transform sentence into sequence of ids using self.word2idx
        #  Keep the counters self._all_words and self._out_of_vocab up to date
        #  for checking coverage -- it is also used for testing.
        words = sentence.split(" ")
        if len(words) > self.max_seq_len:
            words = words[:self.max_seq_len]

        self._all_words += len(words)

        for word in words:
            if word in self.word2idx:
                idx.append(self.word2idx[word])
            else:
                idx.append(self.word2idx[UNK])
                self._out_of_vocab += 1

        while len(idx) < self.max_seq_len:
            idx.append(self.word2idx[PAD])

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
        print(f"loading data from {self._data_folder} ...")
        self.__load_from_file(self._data_folder)

        self.out_of_vocab = self._vectorizer.out_of_vocab_perc()
        self._vectorizer.reset_counter()

    def __load_from_file(self, file):
        # CF#5
        #  load and preprocess the data set from file into self.a self.b self.sts
        #  use vectorizer to store only ids instead of strings
        with open(file, 'r', encoding="utf-8") as fd:
            for i, line in enumerate(fd):
                parts = line.split("\t")
                self.a.append(self._vectorizer.sent2idx(parts[0]))
                self.b.append(self._vectorizer.sent2idx(parts[1]))
                self.sts.append(float(parts[2]))

    def __iter__(self):
        # CF#7
        #  randomly shuffle data in memory and start from begining
        indices = list(range(len(self.a)))
        random.shuffle(indices)

        self.a = [self.a[i] for i in indices]
        self.b = [self.b[i] for i in indices]
        self.sts = [self.sts[i] for i in indices]

        self.pointer = 0

        return self

    def __next__(self):
        # CF#6
        #  Implement yielding a batches from preloaded data: self.a,  self.b, self.sts
        batch = dict()
        batch["a"] = []
        batch["b"] = []
        batch["sts"] = []

        for i in range(self._batch_size):
            if self.pointer >= len(self.a):
                raise StopIteration

            batch["a"].append(self.a[self.pointer])
            batch["b"].append(self.b[self.pointer])
            batch["sts"].append(self.sts[self.pointer])

            self.pointer += 1

        batch["a"] = np.array(batch["a"])
        batch["b"] = np.array(batch["b"])
        batch["sts"] = np.array(batch["sts"])

        return batch


class TwoTowerModel(torch.nn.Module):
    def __init__(self, vecs, config):
        super(TwoTowerModel, self).__init__()
        # CF#10a
        #  Initialize building block for architecture described in the assignment
        self.final_metric = config["final_metric"]
        self.random_emb = config["random_emb"]
        self.emb_training = config["emb_training"]
        self.emb_projection = config["emb_projection"]

        # torch.nn.Embedding
        # torch.nn.Linear
        if self.random_emb:
            self.emb_layer = torch.nn.Embedding(config["vocab_size"], 300)
        else:
            self.emb_layer = torch.nn.Embedding.from_pretrained(torch.tensor(np.array(vecs)), freeze=not self.emb_training)
        self.emb_proj = torch.nn.Linear(300, 128)

        print("requires grads? : ", self.emb_layer.weight.requires_grad)
        self.relu = torch.nn.ReLU()

        if self.emb_projection:
            self.final_proj_1 = torch.nn.Linear(256, 128)
        else:
            self.final_proj_1 = torch.nn.Linear(600, 128)
        self.final_proj_2 = torch.nn.Linear(128, 1)

    def _make_repre(self, idx):
        emb = self.emb_layer(idx)

        if self.emb_projection:
            proj = self.emb_proj(emb.float())
            proj = self.relu(proj)
        else:
            proj = emb

        avg = torch.mean(proj, dim=1)
        return avg

    def forward(self, batch):
        repre_a = self._make_repre(torch.tensor(batch['a']).to(device))
        repre_b = self._make_repre(torch.tensor(batch['b']).to(device))

        # CF#10b
        #  Implement forward pass for the model architecture described in the assignment.
        #  Use both described similarity measures.

        if self.final_metric == "neural":
            concat = torch.cat((repre_a, repre_b), dim=1)
            proj_1 = self.final_proj_1(concat)
            proj_1 = self.relu(proj_1)
            proj_2 = self.final_proj_2(proj_1)
            repre = proj_2
            repre = repre.squeeze()
        elif self.final_metric == "cos":
            repre = torch.nn.functional.cosine_similarity(repre_a, repre_b, dim=1)

        return repre


class DummyModel(torch.nn.Module):  # predat dataset a vracet priod
    def __init__(self, data_loader):
        super(DummyModel, self).__init__()
        # CF#9
        #  Implement DummyModel as described in the assignment.
        self.mean_on_train = np.mean(data_loader.sts)

    def forward(self, batch):
        return torch.tensor([self.mean_on_train for _ in range(len(batch['a']))]).to(device)


# CF#8b
# process whole dataset and return loss
# save loss from each batch and divide it by all on the end.
def test(data_set, net, loss_function):
    running_loss = 0
    all = 0

    with torch.no_grad():
        for i, td in enumerate(data_set):
            predicted_sts = net(td)

            loss = loss_function(torch.tensor(td['sts']).to(device), predicted_sts)
            running_loss += loss.item()
            # This part seems odd, but tests made me swap "predicted_sts.shape[0]" with "1"
            all += 1

    test_loss = running_loss / all
    return test_loss


def train_model(train_dataset, test_dataset, w2v, loss_function, config):
    # net = CzertModel()
    # net = net.to(device)
    net = TwoTowerModel(w2v, config)
    net = net.to(device)

    LR = config["lr"]
    if config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=LR)

    if config["lr_scheduler"] == "stepLR":
        lr_scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
    elif config["lr_scheduler"] == "multiStepLR":
        lr_scheduler = MultiStepLR(optimizer, milestones=[3, 5], gamma=0.1)
    elif config["lr_scheduler"] == "expLR":
        lr_scheduler = ExponentialLR(optimizer, gamma=0.1)

    train_loss_arr = []
    test_loss_arr = []

    running_loss = 0.0
    sample = 0
    x_axes = []  # vis

    # CF#8a Implement training loop
    for epoch in range(EPOCH):
        for i, td in enumerate(train_dataset):
            batch = td
            real_sts = torch.tensor(batch['sts']).to(device)

            optimizer.zero_grad()

            predicted_sts = net(batch)

            loss = loss_function(real_sts.float(), predicted_sts.float())

            # If we are not training embeddings AND projection layer is not there AND final metric is cosine similarity
            # there is literally nothing to train -> no backward pass (prevents error)
            if not(not config["emb_training"] and not config["emb_projection"] and config["final_metric"] == "cos"):
                loss.backward()
            optimizer.step()

            running_loss += loss.item()
            sample += config["batch_size"]
            wandb.log({"train_loss": loss, "lr": lr_scheduler.get_last_lr()}, commit=False)

            if i % MINIBATCH_SIZE == MINIBATCH_SIZE - 1:
                train_loss = running_loss / MINIBATCH_SIZE
                running_loss = 0.0

                train_loss_arr.append(train_loss)

                net.eval()
                test_loss = test(test_dataset, net, loss_function)
                test_loss_arr.append(test_loss)
                net.train()

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
    # Configuration string for wandb better grouping / filtering
    config_string = f"batch_size={config['batch_size']}_lr={config['lr']}_optimizer={config['optimizer']}_random_emb={config['random_emb']}_emb_training={config['emb_training']}_emb_projection={config['emb_projection']}_final_metric={config['final_metric']}_vocab_size={config['vocab_size']}_lr_scheduler={config['lr_scheduler']}"

    wandb.init(
        name=config_string,
        project=wandb_config["WANDB_PROJECT"],
        entity=wandb_config["WANDB_ENTITY"],
        tags=["cv02"],
        config=config
    )

    with open(TRAIN_DATA, 'r', encoding="utf-8") as fd:
        train_data_texts = fd.read().split("\n")

    top_n_words_better = dataset_vocab_analysis_better(train_data_texts, -1)

    word2idx, word_vectors = load_ebs(EMB_FILE, top_n_words_better, config['vocab_size'], force_rebuild=True)

    vectorizer = MySentenceVectorizer(word2idx, MAX_SEQ_LEN)

    train_dataset = DataLoader(vectorizer, TRAIN_DATA, config["batch_size"])
    test_dataset = DataLoader(vectorizer, TEST_DATA, config["batch_size"])

    # dummy_net = DummyModel(train_dataset)
    # dummy_net = dummy_net.to(device)

    loss_function = torch.nn.MSELoss()

    # test(test_dataset, dummy_net, loss_function)
    # test(train_dataset, dummy_net, loss_function)

    train_model(train_dataset, test_dataset, word_vectors, loss_function, config)
