import os
from io import open
import torch
import multiprocessing
from itertools import chain


class Dictionary(object):
    def __init__(self):
        self.char2idx = {'eos': 0,
                         'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10, 'k': 11,
                         'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21,
                         'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26,
                         ' ': 27}

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
                chars = list(line)

                # Define number of workers for multiprocessing
                num_workers = multiprocessing.cpu_count()

                print("Number of workers: ", num_workers)

                # Determine chunk size for each worker
                chunk_size = len(chars) // num_workers

                # Split the content into chunks for each worker
                chunks = [chars[i:i + chunk_size] for i in range(0, len(chars), chunk_size)]

                # Use multiprocessing to tokenize chunks in parallel
                with multiprocessing.Pool(processes=num_workers) as pool:
                    results = pool.map(self.process_chunk, chunks)

                # Flatten the list of results and convert to tensor
                ids = list(chain.from_iterable(results))

            idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids

    def process_chunk(self, chunk):
        """
        Process a chunk of text to convert characters to their corresponding indices.
        :param chunk: A string chunk of text.
        :return: A list of character indices.
        """

        return [self.dictionary.char2idx[char] for char in chunk if char in self.dictionary.char2idx]
