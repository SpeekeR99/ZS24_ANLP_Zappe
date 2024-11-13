import numpy as np
import matplotlib.pyplot as plt
import transformers

train_path_CNEC = "data/train.txt"
dev_path_CNEC = "data/dev.txt"
test_path_CNEC = "data/test.txt"
data_path_CNEC = [train_path_CNEC, dev_path_CNEC, test_path_CNEC]
train_path_UD = "data-mt/train.txt"
dev_path_UD = "data-mt/dev.txt"
test_path_UD = "data-mt/test.txt"
data_path_UD = [train_path_UD, dev_path_UD, test_path_UD]


def load_file(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as fp:
        seq = {"word": [], "label": []}
        for line in fp:
            if line == "\n":
                if seq["word"]:
                    data.append(seq)
                    seq = {"word": [], "label": []}
            else:
                split = line.strip().split(" ")
                if len(split) == 1:
                    split = line.strip().split("\t")
                word = " ".join(split[:-1])
                seq["word"].append(word)
                label = split[-1]
                seq["label"].append(label)
        if seq["word"]:
            data.append(seq)
    return data


def avg_len(data):
    return np.mean([len(seq["word"]) for seq in data])


def main():
    CNEC_data = [load_file(file_path) for file_path in data_path_CNEC]
    UD_data = [load_file(file_path) for file_path in data_path_UD]

    # How large are the two datasets (train, eval, test, overall)
    print("CNEC dataset:")
    for i, data in enumerate(CNEC_data):
        print(f"\t{['train', 'dev', 'test'][i]}: {len(data)} sentences")
    print(f"\tOverall: {sum(len(data) for data in CNEC_data)} sentences")

    print("UD dataset:")
    for i, data in enumerate(UD_data):
        print(f"\t{['train', 'dev', 'test'][i]}: {len(data)} sentences")
    print(f"\tOverall: {sum(len(data) for data in UD_data)} sentences")

    # What is the average length of a training example for the individual datasets
    # in number of whole words tokens as pre-tokenized in the dataset files

    print("CNEC dataset:")
    for i, data in enumerate(CNEC_data):
        print(f"\t{['train', 'dev', 'test'][i]}: {avg_len(data)} words")
    print(f"\tOverall: {avg_len([seq for data in CNEC_data for seq in data])} words")

    print("UD dataset:")
    for i, data in enumerate(UD_data):
        print(f"\t{['train', 'dev', 'test'][i]}: {avg_len(data)} words")
    print(f"\tOverall: {avg_len([seq for data in UD_data for seq in data])} words")

    # What is the average length of a token for the individual datasets
    # in number of subword tokens when using tokenizer

    tokenizer = transformers.BertTokenizerFast.from_pretrained("UWB-AIR/Czert-B-base-cased")
    CNEC_tokenized = []
    for data in CNEC_data:
        tokenized = []
        for seq in data:
            sentence = " ".join(seq["word"])
            tokenized.append(tokenizer.tokenize(sentence))
        CNEC_tokenized.append(tokenized)

    UD_tokenized = []
    for data in UD_data:
        tokenized = []
        for seq in data:
            sentence = " ".join(seq["word"])
            tokenized.append(tokenizer.tokenize(sentence))
        UD_tokenized.append(tokenized)

    print("CNEC dataset:")
    for i, data in enumerate(CNEC_tokenized):
        print(f"\t{['train', 'dev', 'test'][i]}: {np.mean([len(seq) for seq in data])} tokens")
    print(f"\tOverall: {np.mean([len(seq) for data in CNEC_tokenized for seq in data])} tokens")

    print("UD dataset:")
    for i, data in enumerate(UD_tokenized):
        print(f"\t{['train', 'dev', 'test'][i]}: {np.mean([len(seq) for seq in data])} tokens")
    print(f"\tOverall: {np.mean([len(seq) for data in UD_tokenized for seq in data])} tokens")

    # Count statistics about class distribution in dataset (train/dev/test) for the individual datasets

    for i, data in enumerate(CNEC_data):
        labels = [label for seq in data for label in seq["label"]]
        unique, counts = np.unique(labels, return_counts=True)
        # counts = counts / np.sum(counts)
        print(f"CNEC {['train', 'dev', 'test'][i]}:")
        for u, c in zip(unique, counts):
            print(f"\t{u}: {c:.4f}")

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.bar(unique, counts)
        ax.set_title(f"CNEC {['train', 'dev', 'test'][i]} label distribution")
        ax.set_xlabel("Label")
        plt.xticks(rotation=45)
        ax.set_ylabel("Count")
        plt.grid()
        plt.savefig(f"img/CNEC_{['train', 'dev', 'test'][i]}_label_dist.svg")
        plt.show()

    for i, data in enumerate(UD_data):
        labels = [label for seq in data for label in seq["label"]]
        unique, counts = np.unique(labels, return_counts=True)
        # counts = counts / np.sum(counts)
        print(f"UD {['train', 'dev', 'test'][i]}:")
        for u, c in zip(unique, counts):
            print(f"\t{u}: {c:.4f}")

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.bar(unique, counts)
        ax.set_title(f"UD {['train', 'dev', 'test'][i]} label distribution")
        ax.set_xlabel("Label")
        plt.xticks(rotation=45)
        ax.set_ylabel("Count")
        plt.grid()
        plt.savefig(f"img/UD_{['train', 'dev', 'test'][i]}_label_dist.svg")
        plt.show()


if __name__ == "__main__":
    main()
