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


def cross_entropy_loss(y_pred, y_true, reduction="mean"):
    # input.shape: torch.size([-1, class])
    # target.shape: torch.size([-1])
    # reduction = "mean" or "sum"
    # input是模型输出的结果，与target求loss
    # target的长度和input第一维的长度一致
    # target的元素值为目标class
    # reduction默认为mean，即对loss求均值
    # 还有另一种为sum，对loss求和

    # 这里对input所有元素求exp
    exp = torch.exp(y_pred)
    # 根据target的索引，在exp第一维取出元素值，这是softmax的分子
    tmp1 = exp.gather(1, y_true.unsqueeze(-1)).squeeze()
    # 在exp第一维求和，这是softmax的分母
    tmp2 = exp.sum(1)
    # softmax公式：ei / sum(ej)
    softmax = tmp1 / tmp2
    # cross-entropy公式： -yi * log(pi)
    # 因为target的yi为1，其余为0，所以在tmp1直接把目标拿出来，
    # 公式中的pi就是softmax的结果
    log = -torch.log(softmax)
    # 官方实现中，reduction有mean/sum及none
    # 只是对交叉熵后处理的差别
    if reduction == "mean":
        return log.mean()
    else:
        return log.sum()
