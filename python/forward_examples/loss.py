# -*- coding: utf8 -*-
#
import torch

from python.forward_examples.activation import sigmoid, softmax
from torch.nn import CrossEntropyLoss


# bce_loss = BCEWithLogitsLoss()

def bce_loss_with_logit(y_pred, y_true, reduction='mean'):
    y_pred = sigmoid(y_pred)

    loss = -y_true * torch.log(y_pred) - (1 - y_true) * torch.log(1 - y_pred)
    if reduction == 'mean':
        return torch.mean(loss)
    raise NotImplementedError


def cross_entropy_loss(y_pred, y_true):
    y_pred = softmax(y_pred)
    loss = torch.sum(- y_true * torch.log(y_pred))

    return torch.mean(loss)
