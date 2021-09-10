# -*- coding: utf8 -*-
#
import torch

from python.forward_examples.activation import relu


class Linear(torch.nn.Module):
    def __init__(self, in_feature, out_feature):
        super(Linear, self).__init__()
        self.weights = torch.nn.Parameter(torch.randn(in_feature, out_feature))
        self.bias = torch.nn.Parameter(torch.zeros(out_feature))

    def forward(self, x):
        y = torch.matmul(x, self.weights) + self.bias
        return y


class MLP(torch.nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.linear1 = Linear(in_feature, in_feature + out_feature)

        self.linear2 = Linear(in_feature + out_feature, out_feature=out_feature)

    def forward(self, x):
        y = self.linear1(x)
        y = relu(y)
        y = self.linear2(y)
        return y
