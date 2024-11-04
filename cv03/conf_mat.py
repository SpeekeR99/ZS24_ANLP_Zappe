import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_matrix(matrix, name):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Draw matrix
    mat = ax.matshow(matrix, cmap="Greens")
    plt.colorbar(mat)

    # Draw labels
    labels = ["Negative", "Neutral", "Positive"]
    for (i, j), z in np.ndenumerate(matrix):
        ax.text(j, i, "{:0.1f}".format(z), ha="center", va="center", fontsize=10)

    # Set ticks
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set_xlabel("True label")
    ax.set_ylabel("Predicted label")
    ax.set_title(f"Confusion matrix -- {name}")

    # Adjust subplot parameters to shift the figure to the right
    # plt.subplots_adjust(left=0.15)

    plt.savefig(f"img/conf_mat_{name}.svg")
    plt.show()


base_fp = "data/conf_mat_table_"
possibilities = ["mean", "cnn_a", "cnn_b", "cnn_c"]
extension = ".csv"

for p in possibilities:
    df = pd.read_csv(base_fp + p + extension, index_col=0)
    df = df.groupby(["Actual", "Predicted"]).mean()
    df = df.round(0)

    mat = df.to_numpy().reshape(3, 3)
    plot_matrix(mat, p)
