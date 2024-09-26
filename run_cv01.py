import argparse
from cv01.main01 import main

"""
`model: ["dense", "cnn"]`
`lr: [0.1, 0.01, 0.001, 0.0001, 0.00001]`
`optimizer: ["sgd","adam"]`
`dp: [0, 0.1, 0.3, 0.5]`
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, default="dense")
    parser.add_argument('-lr', type=float, default=0.01)
    parser.add_argument('-optimizer', type=str, default="sgd")
    parser.add_argument('-dp', type=float, default=0.0)

    args = parser.parse_args()

    config = {
        "use_normalization": False,
        "batch_size": 10,
        "scheduler": "exponential",
        "model": args.model,
        "lr": args.lr,
        "optimizer": args.optimizer,
        "dp": args.dp
    }

    main(config)
