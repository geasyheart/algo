# -*- coding: utf8 -*-
#
import torch


def relu(x):
    x[x < 0] = 0
    return x


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def softmax(x):
    return torch.exp(x) / torch.sum(torch.exp(x))


def tanh(x):
    return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))


if __name__ == '__main__':
    print([sigmoid(torch.tensor(x, dtype=torch.float)) for x in range(100)])
    print(sigmoid(torch.tensor(list(range(100)), dtype=torch.float), ))
    print('-'.center(60, '-'))
    print(softmax(torch.tensor([2, 3, 5], dtype=torch.float)))
