from cv01.main01 import main

if __name__ == '__main__':
    config = {
        "lr": 0.01,
        "use_normalization":False,
        "optimizer": "sgd", # ADAM,
        "batch_size":10,
        "dp":0,
        "scheduler":"exponential"

    }

    main(config)
    # main()
