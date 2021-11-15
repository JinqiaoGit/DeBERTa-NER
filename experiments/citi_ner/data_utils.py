from typing import Dict, List, Tuple

import torch


def get_word2ix(trainset: List[Tuple[List[str], List[str]]]) -> Dict[str, int]:
    """
    generate one-hot code of tokens
    :param trainset: a list of tuple contains tokens and labels
    :return: a dict contains the map between token and index
    """
    # set <PAD> label as idx 0
    word_to_ix: Dict[str, int] = {"<PAD>": 0}
    for sentence, _ in trainset:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    return word_to_ix


def prepare_sequence(seq: List[str], to_ix: Dict[str, int], device='cpu') -> torch.Tensor:
    """
    convert sequential word to the index in one-hot dictionary.
    """
    idxs = [[to_ix[w] for w in seq]]
    return torch.tensor(idxs, dtype=torch.long, device=device)


def data_refactor():
    pass
