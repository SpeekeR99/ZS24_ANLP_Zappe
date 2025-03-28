import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset

train_data_path = "data/csfd-train.tsv"
test_data_path = "data/csfd-test.tsv"


def main():
    train_data = pd.read_csv(train_data_path, sep='\t', header=None, encoding="utf-8", quoting=csv.QUOTE_NONE, quotechar='"')
    test_data = pd.read_csv(test_data_path, sep='\t', header=None, encoding="utf-8", quoting=csv.QUOTE_NONE, quotechar='"')
    train_data = train_data[1:]
    test_data = test_data[1:]

    print(len(train_data))
    print(len(test_data))

    cls_dataset = load_dataset("csv", delimiter='\t', data_files={"train": [train_data_path],
                                                                  "test": [test_data_path]})
    print(cls_dataset)

    train_scores = train_data[1].to_numpy()
    test_scores = test_data[1].to_numpy()

    # Histogram of training labels
    plt.hist(train_scores, bins=np.arange(4) - 0.5, rwidth=0.75)
    plt.title("Training labels histogram")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.grid()
    plt.savefig("img/training_labels_hist.svg")
    plt.show()

    # Histogram of testing labels
    plt.hist(test_scores, bins=np.arange(4) - 0.5, rwidth=0.75)
    plt.title("Testing labels histogram")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.grid()
    plt.savefig("img/testing_labels_hist.svg")
    plt.show()

    # Sequence length histogram
    train_seq_len = train_data[0].apply(lambda x: len(x.split())).to_numpy()
    test_seq_len = test_data[0].apply(lambda x: len(x.split())).to_numpy()

    plt.hist(train_seq_len, bins=50, rwidth=0.75)
    plt.title("Training sequence length histogram")
    plt.xlabel("Sequence length")
    plt.ylabel("Count")
    plt.grid()
    plt.savefig("img/training_seq_len_hist.svg")
    plt.show()

    plt.hist(test_seq_len, bins=50, rwidth=0.75)
    plt.title("Testing sequence length histogram")
    plt.xlabel("Sequence length")
    plt.ylabel("Count")
    plt.grid()
    plt.savefig("img/testing_seq_len_hist.svg")
    plt.show()

    # Mean + std dev of sequence length
    print(f"Mean training sequence length: {train_seq_len.mean()}")
    print(f"Std dev training sequence length: {train_seq_len.std()}")
    print(f"Mean testing sequence length: {test_seq_len.mean()}")
    print(f"Std dev testing sequence length: {test_seq_len.std()}")

    # Q3
    print(f"Q3 training sequence length: {np.percentile(train_seq_len, 75)}")
    print(f"Q3 testing sequence length: {np.percentile(test_seq_len, 75)}")
    # 90-percentile
    print(f"90-percentile training sequence length: {np.percentile(train_seq_len, 90)}")
    print(f"90-percentile testing sequence length: {np.percentile(test_seq_len, 90)}")


if __name__ == "__main__":
    main()
