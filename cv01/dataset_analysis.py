import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets


def main():
    dataset1 = datasets.MNIST('data', train=True, download=True)
    dataset2 = datasets.MNIST('data', train=False)

    print(dataset1.targets.shape)
    print(dataset2.targets.shape)

    training_targets = dataset1.targets.numpy()
    testing_targets = dataset2.targets.numpy()

    # Histogram of training targets (classes)
    plt.hist(training_targets, bins=np.arange(11) - 0.5, rwidth=0.75)
    plt.title("Training targets histogram")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.grid()
    plt.savefig("training_targets.png")
    plt.show()

    # Histogram of testing targets (classes)
    plt.hist(testing_targets, bins=np.arange(11) - 0.5, rwidth=0.75)
    plt.title("Testing targets histogram")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.grid()
    plt.savefig("testing_targets.png")
    plt.show()


if __name__ == '__main__':
    main()
