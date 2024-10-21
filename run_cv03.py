import argparse


from cv03.main03 import main, CNN_MODEL

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
        "model" : CNN_MODEL,
        "batch_size": None,
        "learning_rate": None,

        "emb_size": 100,
        "lstm_hidden": 1024,
        "lstm_stack": 4,
        "gradient_clip": 1000000,
        "batches": 500000
    }

    config.update(args)

    main(config)
