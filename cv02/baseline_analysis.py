import csv
import numpy as np
import pandas as pd
import torch


train_data_path = "data/anlp01-sts-free-train.tsv"
test_data_path = "data/anlp01-sts-free-test.tsv"


class RandomModel:
    def __init__(self, number_of_classes):
        self.number_of_classes = number_of_classes

    def forward(self, x):
        return np.random.randint(0, self.number_of_classes)


class MajorityClassModel:
    def __init__(self, majority_class):
        self.majority_class = majority_class

    def forward(self, x):
        return self.majority_class


def main():
    train_data = pd.read_csv(train_data_path, sep='\t', header=None, encoding="utf-8", quoting=csv.QUOTE_NONE, quotechar='"')
    test_data = pd.read_csv(test_data_path, sep='\t', header=None, encoding="utf-8", quoting=csv.QUOTE_NONE, quotechar='"')

    train_classes = train_data[2].to_numpy()
    test_classes = test_data[2].to_numpy()
    all_data = np.concatenate((train_classes, test_classes), axis=0)
    number_of_classes = len(np.unique(all_data))
    print(f"Number of classes: {number_of_classes}")

    test_classes = np.round(test_classes).astype(int)
    all_data = np.concatenate((train_classes, test_classes), axis=0)
    number_of_classes = len(np.unique(all_data))
    print(f"Number of classes (post round): {number_of_classes}")

    majority_class = np.argmax(np.bincount(all_data))
    print(f"Majority class (post round): {majority_class}")

    random_model = RandomModel(number_of_classes)
    majority_class_model = MajorityClassModel(majority_class)

    mse_loss_fn = torch.nn.MSELoss()

    random_train_outputs = []
    majority_class_train_outputs = []
    for target in train_classes:
        random_train_outputs.append(random_model.forward(target))
        majority_class_train_outputs.append(majority_class_model.forward(target))

    random_train_correct_hits = np.sum(np.array(random_train_outputs) == np.array(train_classes))
    random_train_acc = random_train_correct_hits / len(train_classes)
    majority_class_train_correct_hits = np.sum(np.array(majority_class_train_outputs) == np.array(train_classes))
    majority_class_train_acc = majority_class_train_correct_hits / len(train_classes)

    mse_random_train_loss = mse_loss_fn(torch.tensor(random_train_outputs).float(), torch.tensor(train_classes).float())
    mse_majority_class_train_loss = mse_loss_fn(torch.tensor(majority_class_train_outputs).float(), torch.tensor(train_classes).float())

    print(f"\nRandom model train accuracy: {random_train_acc}")
    print(f"Majority class model train accuracy: {majority_class_train_acc}")

    print(f"MSE train loss for random model: {mse_random_train_loss}")
    print(f"MSE train loss for majority class model: {mse_majority_class_train_loss}")

    random_test_outputs = []
    majority_class_test_outputs = []
    for target in test_classes:
        random_test_outputs.append(random_model.forward(target))
        majority_class_test_outputs.append(majority_class_model.forward(target))

    random_test_correct_hits = np.sum(np.array(random_test_outputs) == np.array(test_classes))
    random_test_acc = random_test_correct_hits / len(test_classes)
    majority_class_test_correct_hits = np.sum(np.array(majority_class_test_outputs) == np.array(test_classes))
    majority_class_test_acc = majority_class_test_correct_hits / len(test_classes)

    mse_random_test_loss = mse_loss_fn(torch.tensor(random_test_outputs).float(), torch.tensor(test_classes).float())
    mse_majority_class_test_loss = mse_loss_fn(torch.tensor(majority_class_test_outputs).float(), torch.tensor(test_classes).float())

    print(f"\nRandom model test accuracy: {random_test_acc}")
    print(f"Majority class model test accuracy: {majority_class_test_acc}")

    print(f"MSE test loss for random model: {mse_random_test_loss}")
    print(f"MSE test loss for majority class model: {mse_majority_class_test_loss}")


if __name__ == '__main__':
    main()
