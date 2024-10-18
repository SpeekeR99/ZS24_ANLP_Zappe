import argparse
from cv02.main02 import main

"""
From test.py

"lr": [0.01, 0.001, 0.0001, 0.00001],
"optimizer": ["sgd", "adam"],
"random_emb": [True, False],
"emb_training": [True, False],
"emb_projection": [True, False],
"final_metric": ["cos", "neural"],
"vocab_size": [20_000]
"""

"""
My grid search

"batch_size": [500, 1000, 2000],
"lr": [0.01, 0.001, 0.0001, 0.00001],
"optimizer": ["sgd", "adam"],
"lr_scheduler": ["stepLR", "multiStepLR", "expLR"],
"random_emb": [True, False],
"emb_training": [True, False],
"emb_projection": [True, False],
"final_metric": ["cos", "neural"],
"vocab_size": [10000, 20000, 50000]
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', type=int, default=1000)
    parser.add_argument('-lr', type=float, default=0.01)
    parser.add_argument('-optimizer', type=str, default="sgd")
    parser.add_argument('-random_emb', type=bool, default=True)
    parser.add_argument('-emb_training', type=bool, default=True)
    parser.add_argument('-emb_projection', type=bool, default=True)
    parser.add_argument('-final_metric', type=str, default="cos")
    parser.add_argument('-vocab_size', type=int, default=20_000)
    parser.add_argument('-lr_scheduler', type=str, default="stepLR")

    args = parser.parse_args()

    config = {
        "batch_size": args.batch_size,
        "lr": args.lr,
        "optimizer": args.optimizer,
        "random_emb": args.random_emb,
        "emb_training": args.emb_training,
        "emb_projection": args.emb_projection,
        "final_metric": args.final_metric,
        "vocab_size": args.vocab_size
    }

    main(config)
