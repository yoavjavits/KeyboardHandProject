import os
from io import open
import torch


class Dictionary(object):
    def __init__(self):
        self.char2idx = {'eos': 0, 'unk': 1, 'pad': 2}
        for i in range(0, 255):
            self.char2idx[chr(i)] = i + 3

        self.idx2char = {v: k for k, v in self.char2idx.items()}

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
                # split line to individual characters, remove newline character, and add eos token
                chars = list(line)[:-1]
                chars.append('eos')

                ids = []
                for char in chars:
                    if char not in self.dictionary.char2idx:
                        continue

                    ids.append(self.dictionary.char2idx[char])

                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids
