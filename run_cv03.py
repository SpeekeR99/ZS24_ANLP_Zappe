import argparse
import torch


from cv03.main03 import main, CNN_MODEL, MEAN_MODEL

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--learning_rate',
                        help='',
                        type=float,
                        default=1e-6)

    parser.add_argument('--batch_size',
                        help='',
                        type=int,
                        default=200)

    args = vars(parser.parse_args())

    config = {
        "model": CNN_MODEL,
        "batch_size": 33,
        "lr": 0.0001,

        "emb_size": 100,
        "lstm_hidden": 1024,
        "lstm_stack": 4,
        "gradient_clip": 1000000,
        "batches": 500000,

        "seq_len": 100,
        "vocab_size": 20000,
        "emb_training": False,
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        "n_kernel": 64,
        "proj_size": 100,
        "activation": "relu",
        "random_emb": False,
        "emb_projection": True,
        "cnn_architecture": "B",
    }

    config.update(args)

    main(config)
