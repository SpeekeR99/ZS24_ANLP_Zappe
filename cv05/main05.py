import argparse
import wandb
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from torch.utils.data import DataLoader
from datasets import load_dataset
import pandas as pd
import csv
import torch

WANDB_PROJECT = "anlp-2024_zappe_dominik"
WANDB_ENTITY = "anlp2024"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyRegressor(torch.nn.Module):
    def __init__(self, model):
        super(MyRegressor, self).__init__()
        self.model = model
        self.proj = torch.nn.Linear(model.config.hidden_size, 1)

    def forward(self, input_ids):
        outputs = self.model(input_ids)
        last_hidden_state = outputs.last_hidden_state
        return self.proj(last_hidden_state[:, 0, :]).squeeze(-1)


def main(config):
    # wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, tags=["cv05"], config=config)
    # wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, tags=["cv05", "best"], config=config)

    tokenizer = AutoTokenizer.from_pretrained(config["model_type"])

    if config["task"] == "sts":
        # wandb.log({"test_loss": None, "train_loss": None})

        model = AutoModel.from_pretrained(config["model_type"])
        model = MyRegressor(model)
        loss_func = torch.nn.MSELoss()

        data_dir = "data-sts"
        train_data_fp = f"{data_dir}/anlp01-sts-free-train.tsv"
        test_data_fp = f"{data_dir}/anlp01-sts-free-test.tsv"

        train_data = pd.read_csv(train_data_fp, sep='\t', header=None, encoding="utf-8", quoting=csv.QUOTE_NONE, quotechar='"')
        test_data = pd.read_csv(test_data_fp, sep='\t', header=None, encoding="utf-8", quoting=csv.QUOTE_NONE, quotechar='"')

        train_loader = DataLoader(train_data.values.tolist(), batch_size=config["batch_size"], shuffle=True)
        test_loader = DataLoader(test_data.values.tolist(), batch_size=config["batch_size"], shuffle=False)

    else:  # config["task"] == "sentiment":
        # wandb.log({"test_loss": None, "test_acc": None, "train_loss": None})

        model = AutoModelForSequenceClassification.from_pretrained(config["model_type"], num_labels=3)
        loss_func = torch.nn.CrossEntropyLoss()

        data_dir = "data-sentiment"
        train_data_fp = f"{data_dir}/csfd-train.tsv"
        test_data_fp = f"{data_dir}/csfd-test.tsv"

        cls_dataset = load_dataset("csv", delimiter='\t', data_files={"train": [train_data_fp], "test": [test_data_fp]})

        train_loader = DataLoader(cls_dataset["train"], batch_size=config["batch_size"], shuffle=True)
        test_loader = DataLoader(cls_dataset["test"], batch_size=config["batch_size"], shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    seq_len = config["seq_len"]

    if config["task"] == "sts":
        for epoch in range(config["epochs"]):
            model.train()
            running_loss = 0
            for i, batch in enumerate(train_loader):
                optimizer.zero_grad()

                input_ids = tokenizer(list(batch[0]), padding=True, truncation=True, return_tensors="pt", max_length=seq_len)["input_ids"]
                input_ids_2 = tokenizer(list(batch[1]), padding=True, truncation=True, return_tensors="pt", max_length=seq_len)["input_ids"]
                input = torch.cat([input_ids, input_ids_2], dim=1)
                outputs = model(input)

                loss = loss_func(outputs, batch[2].float())
                loss.backward()

                optimizer.step()

                running_loss += loss.item()

            running_loss /= len(train_loader)
            print(f"Epoch {epoch}, loss: {running_loss}")
            # wandb.log({"train_loss": running_loss})

            running_loss = 0
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(test_loader):
                    input_ids = tokenizer(list(batch[0]), padding=True, truncation=True, return_tensors="pt", max_length=seq_len)["input_ids"]
                    input_ids_2 = tokenizer(list(batch[1]), padding=True, truncation=True, return_tensors="pt", max_length=seq_len)["input_ids"]
                    input = torch.cat([input_ids, input_ids_2], dim=1)
                    outputs = model(input)

                    loss = loss_func(outputs, batch[2].float())

                    running_loss += loss.item()

            running_loss /= len(test_loader)
            print(f"Epoch {epoch}, test loss: {running_loss}")
            # wandb.log({"test_loss": running_loss})

            lr_scheduler.step()
    else:
        for epoch in range(config["epochs"]):
            model.train()
            running_loss = 0
            for i, batch in enumerate(train_loader):
                optimizer.zero_grad()

                input_ids = tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt", max_length=seq_len)["input_ids"]
                outputs = model(input_ids)

                loss = loss_func(outputs.logits, batch["label"])
                loss.backward()

                optimizer.step()

                running_loss += loss.item()

            running_loss /= len(train_loader)
            print(f"Epoch {epoch}, loss: {running_loss}")
            # wandb.log({"train_loss": running_loss})

            running_loss = 0
            running_acc = 0
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(test_loader):
                    input_ids = tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt", max_length=seq_len)["input_ids"]
                    outputs = model(input_ids)

                    loss = loss_func(outputs.logits, batch["label"])

                    running_loss += loss.item()
                    acc = (outputs.logits.argmax(dim=1) == batch["label"]).float().mean()
                    running_acc += acc

            running_loss /= len(test_loader)
            print(f"Epoch {epoch}, test loss: {running_loss}")
            # wandb.log({"test_loss": running_loss})
            running_acc /= len(test_loader)
            print(f"Epoch {epoch}, test acc: {running_acc}")
            # wandb.log({"test_acc": running_acc})

            lr_scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-task", type=str, default="sentiment")  # < 0.65 for sts; > 0.75 for sentiment
    parser.add_argument("-model_type", type=str, default="UWB-AIR/Czert-B-base-cased")
    parser.add_argument("-epochs", type=int, default=10)
    parser.add_argument("-batch_size", type=int, default=32)
    parser.add_argument("-lr", type=float, default=1e-4)
    parser.add_argument("-seq_len", type=int, default=64)
    args = parser.parse_args()

    config = {
        "task": args.task,
        "model_type": args.model_type,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "seq_len": args.seq_len
    }

    main(config)
