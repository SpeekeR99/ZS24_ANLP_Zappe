from cv02.main02 import main

if __name__ == '__main__':
    my_config = {
        "vocab_size": 20000,
        "random_emb": True
    }
    print(my_config)
    main(my_config)
