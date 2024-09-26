import argparse
import os
import sys
from collections import defaultdict

import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, ExponentialLR

from wandb_config import WANDB_PROJECT, WANDB_ENTITY
import matplotlib.pyplot as plt


class DenseNet(nn.Module):
    def __init__(self, dp=.2):
        super(DenseNet, self).__init__()
        # TODO CF#Dense : definujte síť s 3 lineárními vrstvami, správnou velikost první dopočítejte z velikosti vstupu
        # Další velikosti vrstev nastavte na 128x128 nakonec aplikujte softmax
        in_features = 28 * 28
        other_features = 128
        self.linear1 = nn.Linear(in_features, other_features)
        self.linear2 = nn.Linear(other_features, other_features)
        self.linear3 = nn.Linear(other_features, other_features)
        self.dropout1 = nn.Dropout(dp)
        self.dropout2 = nn.Dropout(dp)

    def forward(self, x):
        # obrázek musíme nejprve "rozbalit" do 1D vektoru uděláme to ale až od první dimenze, protože první dimenze
        # je batch a tu chceme zachovat
        x = x.flatten(1)

        # poté postupně aplikujeme lineární vrstvy a aktivační funkce
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.linear3(x)

        output = F.log_softmax(x, dim=1)

        return output


class Net(nn.Module):
    def __init__(self, dp=.2):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (3, 3), (1, 1))
        self.conv2 = nn.Conv2d(32, 64, (3, 3), (1, 1))
        self.dropout1 = nn.Dropout(dp)
        self.dropout2 = nn.Dropout(dp)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def test(model, device, test_loader, config):
    SAMPLES = 200
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, d in enumerate(test_loader):

            data, target = d
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += output.size()[0]
            if i * config["batch_size"] == SAMPLES:
                break
    test_loss /= total  # delit totalem
    acc = 100. * correct / total
    print(f"test: avg-loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({acc:.0f}%)\n")
    return test_loss, acc


def count_norm(model_params):
    total_norm = 0
    for p in model_params:
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        else:
            print("NO GRAD")
            pass
    total_norm = total_norm ** 0.5
    return total_norm


def main(config: dict):
    # ukažme rozdíl cpu a gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, tags=["cv01"], config=config)

    EPOCHS = 2
    BATCH_SIZE = config["batch_size"]
    LOG_INTERVAL = 1

    LR = config["lr"]
    DP = config["dp"]

    config["use_normalization"] = False

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset1 = datasets.MNIST('data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=BATCH_SIZE, shuffle=True)

    if config["model"] == "dense":
        model = DenseNet(dp=DP).to(device)
    elif config["model"] == "cnn":
        model = Net(dp=DP).to(device)

    wandb.watch(model, log="all")

    if config["optimizer"] == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=LR)
    elif config["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=LR)

    # training loop
    for epoch in range(EPOCHS):
        for batch_idx, (data, target) in enumerate(train_loader):
            pass
            # plt.imshow(data[0].permute(1, 2, 0))
            # plt.imshow(data[0].permute(2, 1, 0))
            # print(data)
            # plt.show()
            # print(target)

            # print(data[0])
            # print(data[0].shape)

            optimizer.zero_grad()
            output = model(data)  # forward

            norm = count_norm(model.parameters())

            loss = F.nll_loss(output, target)

            loss.backward()
            optimizer.step()

            print(f"e{epoch} b{batch_idx} s{batch_idx * BATCH_SIZE}]\t"  
                  f"Loss: {loss.item():.6f}")
            wandb.log({"train_loss": loss.item()})

            # Evaluation
            if batch_idx % LOG_INTERVAL == LOG_INTERVAL - 1:
                model.eval()
                test_loss, test_acc = test(model, device, test_loader, config)
                wandb.log({"test_loss": test_loss, "test_acc": test_acc})
                model.train()


if __name__ == '__main__':
    config = defaultdict(lambda: False)

    print(config)
    # add parameters lr,optimizer,dp
    main(config)
