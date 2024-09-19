UNK = "<UNK>"
PAD = "<PAD>"

class MySentenceVectorizer():
    def __init__(self, word2idx, max_seq_len, pad, unk):
        self.pad = pad
        self.unk = unk
        self.max_seq_len = max_seq_len
        self._all_words = 0
        self._out_of_vocab = 0
        self.word2idx = word2idx

    def sent2idx(self, sentence):
        idx = []
        # todo CF#4
        #  Transform sentence into sequence of ids using self.word2idx
        #  Keep the counters self._all_words and self._out_of_vocab up to date
        #  for checking coverage -- it is also used for testing.


        return idx

