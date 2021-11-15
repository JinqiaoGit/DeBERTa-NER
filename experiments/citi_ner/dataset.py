from typing import List, Dict

import torch
from torch.utils.data import Dataset
import numpy as np

from constant import PADDING_LABEL, LABEL2IX
from data_utils import get_word2ix


class DocDataset(Dataset):
    def __init__(
            self, sentences: List[List[str]],
            labels: List[List[str]],
            word2ix: Dict[str, int] = None,
            padding_size: int = None,
            label2ix: Dict[str, int] = LABEL2IX):
        # set word to idx dictionary
        if not word2ix and isinstance(sentences, list) and isinstance(labels, list):
            self.word2ix = get_word2ix([(sent, label) for sent, label in zip(sentences, labels)])
        else:
            self.word2ix = word2ix
        if not sentences or not labels:
            raise ValueError("Empty input and output of dataset, please check your data")
        self.sentences = sentences
        self.labels = labels
        self.padding_size = padding_size
        self.label2ix = label2ix
        self.train_data, self.train_label = None, None
        # padding dataset
        self._dataset_padding()
        # convert array to torch Tensor
        self.train_data, self.train_label = torch.LongTensor(self.train_data), torch.LongTensor(self.train_label)

    def __len__(self):
        """
        return the number of sentence
        """
        return len(self.sentences)

    def __getitem__(self, idx):
        """
        return the batch item based on the idx
        """
        return self.train_data[idx], self.train_label[idx]

    def _dataset_padding(self):
        """
        padding all sentences and labels based on the max length of sentence.
        notice that each sentence and label has the same length,
        the padding size is human defined, which should be in the list [64, 128, 256, 512, ...]
        """
        if not isinstance(self.padding_size, int):
            max_sentence_len = max(len(sent) for sent in self.sentences)
            # find the minimum padding size
            for interval in [64, 128, 256, 512]:
                if max(interval, max_sentence_len) == interval:
                    self.padding_size = interval
                    break
                if interval == 512:
                    self.padding_size = interval
                    break

        # padding train data with padding label, which is index 0 in my settings
        self.train_data = self.word2ix[PADDING_LABEL] * np.ones((len(self.sentences), self.padding_size))

        # padding label data with -1
        self.train_label = -1 * np.ones((len(self.sentences), self.padding_size))

        # copy the data to numpy array
        for sent_idx, sent in enumerate(self.sentences):
            sent_length = len(sent)
            self.train_data[sent_idx][:sent_length] = [self.word2ix[token] for token in sent]
            self.train_label[sent_idx][:sent_length] = [self.label2ix[label] for label in self.labels[sent_idx]]
