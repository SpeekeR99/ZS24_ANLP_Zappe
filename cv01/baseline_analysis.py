import numpy as np
import torch
from torchvision import datasets, transforms


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
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset1 = datasets.MNIST('data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('data', train=False, transform=transform)

    all_data = torch.utils.data.ConcatDataset([dataset1, dataset2])
    number_of_classes = len(np.unique([y for x, y in all_data]))
    print(f"Number of classes: {number_of_classes}")
    majority_class = np.argmax(np.bincount([y for x, y in all_data]))
    print(f"Majority class: {majority_class}")

    BATCH_SIZE = 1
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=BATCH_SIZE, shuffle=True)

    random_model = RandomModel(number_of_classes)
    majority_class_model = MajorityClassModel(majority_class)

    random_outputs = []
    majority_class_outputs = []
    expected_outputs = []
    # for batch_idx, (data, target) in enumerate(train_loader):
    #     random_outputs.append(random_model.forward(data))
    #     majority_class_outputs.append(majority_class_model.forward(data))
    #     expected_outputs.append(target)
    for batch_idx, (data, target) in enumerate(test_loader):
        random_outputs.append(random_model.forward(data))
        majority_class_outputs.append(majority_class_model.forward(data))
        expected_outputs.append(target)

    random_correct_hits = np.sum(np.array(random_outputs) == np.array(expected_outputs))
    random_acc = random_correct_hits / len(expected_outputs)
    majority_class_correct_hits = np.sum(np.array(majority_class_outputs) == np.array(expected_outputs))
    majority_class_acc = majority_class_correct_hits / len(expected_outputs)

    print(f"Random model accuracy: {random_acc}")
    print(f"Majority class model accuracy: {majority_class_acc}")


if __name__ == '__main__':
    main()
