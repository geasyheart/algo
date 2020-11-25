# -*- coding: utf8 -*-
import random

import torch
import numpy as np
from torch import nn, optim

from time import time

from torch.nn import init
from torch.utils import data

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

# 读取数据
batch_size = 10
# 将训练数据的特征和标签组合
dataset = data.TensorDataset(features, labels)
# 随机读取小批量
data_iter = data.DataLoader(dataset, batch_size, shuffle=True)


#
# for x, y in data_iter:
#     print(x, y)

# 定义模型

class LinerNet(nn.Module):
    def __init__(self, n_feature):
        super(LinerNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    def forward(self, x):
        y = self.linear(x)
        return y


net = LinerNet(num_inputs)

init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)  # 也可以直接修改bias的data: net[0].bias.data.fill_(0)

loss = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.03)

num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad()  # 梯度清零，等价于net.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))
