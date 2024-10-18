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

"batch_size": [1000],
"lr": [0.01, 0.001, 0.0001, 0.00001],
"optimizer": ["sgd", "adam"],
"lr_scheduler": ["multiStepLR", "expLR"],
"random_emb": [True, False],
"emb_training": [True, False],
"emb_projection": [True, False],
"final_metric": ["cos", "neural"],
"vocab_size": [20000, 50000]
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', type=int, default=1000)
    parser.add_argument('-lr', type=float, default=0.01)
    parser.add_argument('-optimizer', type=str, default="sgd")
    parser.add_argument('-random_emb', type=str, default="false")
    parser.add_argument('-emb_training', type=str, default="false")
    parser.add_argument('-emb_projection', type=str, default="true")
    parser.add_argument('-final_metric', type=str, default="neural")
    parser.add_argument('-vocab_size', type=int, default=20_000)
    parser.add_argument('-lr_scheduler', type=str, default="multiStepLR")

    args = parser.parse_args()

    config = {
        "batch_size": args.batch_size,
        "lr": args.lr,
        "optimizer": args.optimizer,
        "random_emb": args.random_emb == "true",
        "emb_training": args.emb_training == "true",
        "emb_projection": args.emb_projection == "true",
        "final_metric": args.final_metric,
        "vocab_size": args.vocab_size,
        "lr_scheduler": args.lr_scheduler
    }

    main(config)
