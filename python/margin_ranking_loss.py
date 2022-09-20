# -*- coding: utf8 -*-
#


import torch
import torch.nn as nn


def margin_ranking_loss(input1, input2, target, margin):
    """
    如果target为1,则意味着input1要在input2的前面
    比如y == 1：
        x1 == 0.1, x2 == 0.2, 那么 x2 - x1 == 0.1，即为loss
        x1 == 0.2, x1 == 0.1, 那么 x2 - x1 == -0.1，表明x2就在x1的后面，就不需要计算loss了，所以loss_val == 0
    所以计算的是x1和x2的margin

    实例可参考pairwise
    """
    val = 0
    for x1, x2, y in zip(input1, input2, target):
        loss_val = max(0, -y * (x1 - x2) + margin)
        val += loss_val
    return val / input1.nelement()


torch.manual_seed(10)
margin = 0
loss = nn.MarginRankingLoss()
input1 = torch.randn([3], requires_grad=True)
input2 = torch.randn([3], requires_grad=True)
target = torch.tensor([1, 1, 1])
print(target)
output = loss(input1, input2, target)
print(output.item())

output = margin_ranking_loss(input1, input2, target, margin)
print(output.item())

loss = nn.MarginRankingLoss(reduction="none")
output = loss(input1, input2, target)
print(output)
