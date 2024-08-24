import os
from io import open
import torch


class Dictionary(object):
    def __init__(self):
        char2idx = {}
        for i in range(32, 127):
            char2idx[chr(i)] = i - 32

        char2idx['\n'] = 95
        char2idx['\t'] = 96

        self.char2idx = char2idx

        self.idx2char = {v: k for k, v in char2idx.items()}

    def __len__(self):
        return len(self.char2idx)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """
        Tokenizes a text file.
        :param path: file path
        :return: tensor of shape [N, ] where N is the number of characters in the file, including the eos token
        """
        assert os.path.exists(path)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                chars = [c for c in list(line) if c in self.dictionary.char2idx]

                ids = []
                for char in chars:
                    ids.append(self.dictionary.char2idx[char])

                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids
