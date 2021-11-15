# _*_ coding: utf-8 _*_
import torch


def cross_entropy_loss(outputs: torch.LongTensor, labels: torch.LongTensor):
    """
    This is the cross entropy loss function

    :param outputs:
        this is the output of model
    :param labels:
        this is the ground truth of the token's label
    :return:
        the loss array based on cross-entropy
    """
    # reshape labels to give a flat vector with length batch_size*seq_len
    labels = labels.reshape(-1)

    # mask out '<PAD>' tokens
    mask = (labels >= 0).float()

    # the number of tokens is the sum of elements in mask
    num_tokens = int(torch.sum(mask).data)

    # pick the values corresponding to labels and multiply by mask
    outputs = outputs[range(outputs.shape[0]), labels] * mask

    # cross entropy loss for all non <PAD> tokens
    return -torch.sum(outputs) / num_tokens
