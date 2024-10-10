import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


train_data_path = "data/anlp01-sts-free-train.tsv"
test_data_path = "data/anlp01-sts-free-test.tsv"


def main():
    train_data = pd.read_csv(train_data_path, sep='\t', header=None, encoding="utf-8", quoting=csv.QUOTE_NONE, quotechar='"')
    test_data = pd.read_csv(test_data_path, sep='\t', header=None, encoding="utf-8", quoting=csv.QUOTE_NONE, quotechar='"')

    print(len(train_data))
    print(len(test_data))

    train_scores = train_data[2].to_numpy()
    test_scores = test_data[2].to_numpy()
    # Round all the test_scores
    test_scores = np.round(test_scores)

    # Histogram of training pair similarities
    plt.hist(train_scores, bins=np.arange(8) - 0.5, rwidth=0.75)
    plt.title("Training pair similarities histogram")
    plt.xlabel("Similarity")
    plt.ylabel("Count")
    plt.grid()
    plt.savefig("img/training_similarity.svg")
    plt.show()

    # Histogram of testing pair similarities
    plt.hist(test_scores, bins=np.arange(8) - 0.5, rwidth=0.75)
    plt.title("Testing pair similarities histogram")
    plt.xlabel("Similarity")
    plt.ylabel("Count")
    plt.grid()
    plt.savefig("img/testing_similarity.svg")
    plt.show()

    # Mean and std dev
    print(f"Train mean: {np.mean(train_scores)}")
    print(f"Train std dev: {np.std(train_scores)}")
    print(f"Test mean: {np.mean(test_scores)}")
    print(f"Test std dev: {np.std(test_scores)}")


if __name__ == "__main__":
    main()
