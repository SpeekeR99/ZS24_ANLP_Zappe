# https://pytorch.org/data/main/tutorial.html
# https://towardsdatascience.com/text-classification-with-cnns-in-pytorch-1113df31e79f
import configparser
import os
import pickle
import sys
from collections import defaultdict

import random
import numpy as np
import torch
# from ignite.contrib.handlers.param_scheduler import create_lr_scheduler_with_warmup
from datasets import load_dataset

import wandb
from torch import nn
from torch.utils.data import DataLoader

import wandb_config
from cv02.consts import EMB_FILE
from cv02.main02 import dataset_vocab_analysis, MySentenceVectorizer, PAD, UNK

NUM_CLS = 3

CNN_MODEL = "cnn"
MEAN_MODEL = "mean"

CSFD_DATASET_TRAIN = "cv03/data/csfd-train.tsv"
CSFD_DATASET_TEST = "cv03/data/csfd-test.tsv"

CLS_NAMES = ["neg", "neu", "pos"]

WORD2IDX = "word2idx.pckl"
VECS_BUFF = "vecs.pckl"

from wandb_config import WANDB_PROJECT, WANDB_ENTITY


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
            vecs = [np.random.uniform(-1, 1, 300) for _ in range(wanted_vocab_size)]

            # CF#2
            #  create map of  word->id  of top according to the given top_n_words
            #  create a matrix as a np.array : word vectors
            #  vocabulary ids corresponds to vectors in the matrix
            #  Do not forget to add UNK and PAD tokens into the vocabulary.

            for word in top_n_words:  # Interesting note here, this is basically only used for the later if (*)
                word2idx[word] = idx
                idx += 1

            for i, line in enumerate(emb_fd):
                if i == 0:
                    continue

                parts = line.split(" ")
                word = parts[0]

                if word in word2idx:  # (*) this is a LOT faster for some reason, than "word in top_n_words"
                    vecs[word2idx[word]] = np.array([float(x) for x in parts[1:]])

            word2idx[UNK] = idx
            word2idx[PAD] = idx + 1
            vecs[idx + 1] = np.zeros(300)

            # assert len(word2idx) > 6820
            # assert len(vecs) == len(word2idx)
            pickle.dump(word2idx, open(WORD2IDX, 'wb'))
            pickle.dump(vecs, open(VECS_BUFF, 'wb'))

    return word2idx, vecs


def count_statistics(dataset, vectorizer):
    # CF#01
    for line in dataset["text"]:
        vectorizer.sent2idx(line)

    coverage = 1 - (vectorizer.out_of_vocab_perc() / 100)

    class_distribution = defaultdict(int)
    labels = np.array(dataset["label"])
    uniq, counts = np.unique(labels, return_counts=True)
    for c in uniq:
        class_distribution[c] = counts[c]
    for c in uniq:
        class_distribution[c] /= len(dataset["label"])

    return coverage, class_distribution


class MyBaseModel(torch.nn.Module):
    def __init__(self, config, w2v=None):
        super(MyBaseModel, self).__init__()
        self.config = config
        self.softmax = nn.Softmax(dim=-1)

        if config["activation"] == "relu":
            self.activation = nn.ReLU()
        elif config["activation"] == "gelu":
            self.activation = nn.GELU()

        self.random_emb = config["random_emb"]
        self.emb_training = config["emb_training"]
        self.emb_projection = config["emb_projection"]

        if w2v:
            if self.random_emb:
                self.emb_layer = torch.nn.Embedding(config["vocab_size"], np.array(w2v).shape[1])
            else:
                self.emb_layer = torch.nn.Embedding.from_pretrained(torch.tensor(np.array(w2v)), freeze=not self.emb_training)

            self.emb_proj = torch.nn.Linear(np.array(w2v).shape[1], config["proj_size"])
        else:
            self.emb_layer = torch.nn.Embedding(config["vocab_size"], config["proj_size"])
            self.emb_proj = torch.nn.Linear(config["proj_size"], config["proj_size"])


class MyModelAveraging(MyBaseModel):
    def __init__(self, config, w2v=None):
        super(MyModelAveraging, self).__init__(config, w2v)

        if self.emb_projection:
            self.head = nn.Linear(config["proj_size"], NUM_CLS)
        else:
            self.head = nn.Linear(np.array(w2v).shape[1], NUM_CLS)

    def forward(self, x):
        x = torch.tensor(x).to(self.config["device"])
        emb = self.emb_layer(x).float()

        if self.emb_projection:
            proj = self.emb_proj(emb)
            proj = self.activation(proj)
        else:
            proj = emb

        avg = torch.mean(proj, dim=1)
        final = self.head(avg)
        return self.softmax(final)


class MyModelConv(MyBaseModel):
    def __init__(self, config, w2v=None):
        super(MyModelConv, self).__init__(config, w2v),

        # CF#CNN_CONF
        self.reduced_emb_size = config["proj_size"] if self.emb_projection else np.array(w2v).shape[1]

        self.cnn_architecture = config["cnn_architecture"]  # for unit tests
        if self.cnn_architecture == "A":
            self.config["hidden_size"] = 505
            self.cnn_config = [(1, config["n_kernel"], (2, 1)),
                               (1, config["n_kernel"], (3, 1)),
                               (1, config["n_kernel"], (4, 1))]
        elif self.cnn_architecture == "B":
            self.config["hidden_size"] = 979
            self.cnn_config = [(1, config["n_kernel"], (2, self.reduced_emb_size // 2)),
                               (1, config["n_kernel"], (3, self.reduced_emb_size // 2)),
                               (1, config["n_kernel"], (4, self.reduced_emb_size // 2))]
        elif self.cnn_architecture == "C":
            self.config["hidden_size"] = 35000
            self.cnn_config = [(1, config["n_kernel"], (2, self.reduced_emb_size)),
                               (1, config["n_kernel"], (3, self.reduced_emb_size)),
                               (1, config["n_kernel"], (4, self.reduced_emb_size))]

        self.conv_layers = []
        for i, (in_channels, out_channels, kernel_size) in enumerate(self.cnn_config):
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size))

        self.max_pools = []
        for i, (in_channels, out_channels, kernel_size) in enumerate(self.cnn_config):
            self.max_pools.append(nn.MaxPool2d((config["seq_len"] - kernel_size[0] + 1, 1)))

        self.dropout = nn.Dropout(.5)
        # Explanation of factor (config["proj_size"] - self.cnn_config[0][2][1] + 1):
        #   With one extreme self.cnn_config[0][2][1] = 1, so the whole multiplication is * config["proj_size"]
        #   The other extreme is self.cnn_config[0][2][1] = config["proj_size"], so the whole multiplication is * 1
        self.final_proj = nn.Linear(len(self.cnn_config) * config["n_kernel"] * (self.reduced_emb_size - self.cnn_config[0][2][1] + 1), config["hidden_size"])
        self.head = nn.Linear(config["hidden_size"], NUM_CLS)

        self.modules = nn.ModuleList(self.conv_layers)

    def forward(self, x):
        x = torch.tensor(x).to(self.config["device"])
        emb = self.emb_layer(x).float()

        if self.emb_projection:
            proj = self.emb_proj(emb)
            proj = self.activation(proj)
        else:
            proj = emb

        proj = proj.unsqueeze(1)

        conv_outs = []
        for i, conv in enumerate(self.conv_layers):
            conv_out = conv(proj)
            conv_out = self.activation(conv_out)
            conv_out = self.max_pools[i](conv_out)
            conv_out = self.dropout(conv_out)
            conv_outs.append(conv_out)

        conv_outs = torch.cat(conv_outs, dim=1)
        conv_outs = conv_outs.view(conv_outs.size(0), -1)

        final_proj = self.final_proj(conv_outs)
        final_proj = self.activation(final_proj)

        final = self.head(final_proj)
        return self.softmax(final)


def test_on_dataset(dataset_iterator, vectorizer, model, loss_metric_func):
    test_loss_list = []
    test_acc_list = []
    test_enum_y = []
    test_enum_pred = []

    with torch.no_grad():
        for b in dataset_iterator:
            texts = b["text"]
            labels = b["label"]

            vectorized = []
            for text in texts:
                vectorized.append(vectorizer.sent2idx(text))

            predicted_labels = model(vectorized)
            test_enum_y.append(labels)
            test_enum_pred.append(predicted_labels.argmax(dim=1))

            loss = loss_metric_func(predicted_labels, labels)
            test_loss_list.append(loss.item())

            correct = (predicted_labels.argmax(dim=1) == labels).float()
            test_acc_list.append(correct.mean().item())

    return {
        "test_acc": sum(test_acc_list) / len(test_acc_list),
        "test_loss": sum(test_loss_list) / len(test_loss_list),
        "test_pred_clss": torch.cat(test_enum_pred),
        "test_enum_gold": torch.cat(test_enum_y),
    }


def train_model(cls_train_iterator, cls_val_iterator, vectorizer, w2v, config):
    if config["model"] == CNN_MODEL:
        model = MyModelConv(config, w2v=w2v)
    elif config["model"] == MEAN_MODEL:
        model = MyModelAveraging(config, w2v=w2v)

    num_of_params = 0
    for x in model.parameters():
        print(x.shape)
        num_of_params += torch.prod(torch.tensor(x.shape), 0)
    config["num_of_params"] = num_of_params
    print("num of params:", num_of_params)

    # Configuration string for wandb better grouping / filtering
    config_string = f"model={config['model']}_" \
                    f"vocab_size={config['vocab_size']}_" \
                    f"seq_len={config['seq_len']}_" \
                    f"batches={config['batches']}_" \
                    f"batch_size={config['batch_size']}_" \
                    f"lr={config['lr']}_" \
                    f"activation={config['activation']}_" \
                    f"random_emb={config['random_emb']}_" \
                    f"emb_training={config['emb_training']}_" \
                    f"emb_projection={config['emb_projection']}_" \
                    f"proj_size={config['proj_size']}_" \
                    f"gradient_clip={config['gradient_clip']}_" \
                    f"n_kernel={config['n_kernel']}_" \
                    f"cnn_architecture={config['cnn_architecture']}"
    wandb.init(name=config_string, project=WANDB_PROJECT, entity=WANDB_ENTITY, tags=["cv03"], config=config)
    # wandb.init(name=config_string, project=WANDB_PROJECT, entity=WANDB_ENTITY, tags=["cv03","best"], config=config)

    model.to(config["device"])
    cross_entropy = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2)

    batch = 0
    while True:
        for b in cls_train_iterator:
            texts = b["text"]
            labels = b["label"]

            vectorized = []
            for text in texts:
                vectorized.append(vectorizer.sent2idx(text))

            optimizer.zero_grad()

            predicted_labels = model(vectorized)

            loss = cross_entropy(predicted_labels, labels)
            loss.backward()

            optimizer.step()

            pred = predicted_labels.argmax(dim=1)
            train_acc = (pred == labels).float().mean().item()
            total_norm = nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])

            if batch % 100 == 0:
                model.eval()

                ret = test_on_dataset(cls_val_iterator, vectorizer, model, cross_entropy)
                wandb.log({"val_acc": ret["test_acc"], "val_loss": ret["test_loss"]}, commit=False)
                conf_matrix = wandb.plot.confusion_matrix(preds=ret["test_pred_clss"].cpu().numpy(), y_true=ret["test_enum_gold"].cpu().numpy(), class_names=CLS_NAMES)
                wandb.log({"conf_mat":conf_matrix})
                print(f"batch: {batch}, val_acc: {ret['test_acc']}, val_loss: {ret['test_loss']}")

                model.train()

            wandb.log({"train_loss": loss, "train_acc": train_acc, "lr": lr_scheduler.get_last_lr()[0], "pred": pred, "norm": total_norm})
            print(f"batch: {batch}, train_acc: {train_acc}, train_loss: {loss}")
            batch += 1

        lr_scheduler.step()

        if batch > config["batches"]:
            break

    return model, cross_entropy


def main(config : dict):
    cls_dataset = load_dataset("csv", delimiter='\t', data_files={"train": [CSFD_DATASET_TRAIN], "test": [CSFD_DATASET_TEST]})

    top_n_words = dataset_vocab_analysis(cls_dataset['train']['text'], top_n=-1)

    word2idx, word_vectors = load_ebs(EMB_FILE, top_n_words, config['vocab_size'], force_rebuild=False)

    vectorizer = MySentenceVectorizer(word2idx, config["seq_len"])

    # TODO: my coverage: 0.7969 vs. test: (0.68, 0.78) ; My coverage is too good?
    coverage, cls_dist = count_statistics(cls_dataset['train'], vectorizer)
    print(f"COVERAGE: {coverage}\ncls_dist:{cls_dist}")

    # Split train into train and validation
    split_dataset = cls_dataset['train'].train_test_split(test_size=0.2)
    cls_train_iterator = DataLoader(split_dataset['train'], batch_size=config['batch_size'])
    cls_val_iterator = DataLoader(split_dataset['test'], batch_size=config['batch_size'])
    cls_test_iterator = DataLoader(cls_dataset['test'], batch_size=config['batch_size'])

    model, used_loss = train_model(cls_train_iterator, cls_val_iterator, vectorizer, word_vectors, config)
    ret_dict = test_on_dataset(cls_test_iterator, vectorizer, model, used_loss)
    wandb.log({"test_acc": ret_dict["test_acc"], "test_loss": ret_dict["test_loss"]})
    print(f"test_acc: {ret_dict['test_acc']}, test_loss: {ret_dict['test_loss']}")
