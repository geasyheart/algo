# -*- coding: utf8 -*-
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.nn import init
from torch.utils import data

# from torch_learn import sgd, evaluate_accuracy

mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True,
                                                transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True,
                                               transform=transforms.ToTensor())

# print(mnist_test[0][0].shape)
# 读取数据
train_iter = data.DataLoader(mnist_train, batch_size=256, shuffle=True, num_workers=4)
test_iter = data.DataLoader(mnist_test, batch_size=256, shuffle=False, num_workers=4)

# for X, y in train_iter:
#     print(X.shape, y)

# ##### model

num_inputs = 784
num_outputs = 10


class LinerNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinerNet, self).__init__()
        self.liner = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        # origin = torch.ones(256, 1, 28, 28)       shape: torch.Size([256, 1, 28, 28])
        # to = origin.view(origin.shape[0], -1)     shape: torch.Size([256, 784])

        y = self.liner(x.view(x.shape[0], -1))
        return y


net = LinerNet(num_inputs, num_outputs)

init.normal_(net.liner.weight, mean=0, std=0.01)
init.constant_(net.liner.bias, val=0)

loss = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 5

for epoch in range(num_epochs):
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y).sum()

    #     # 梯度清零
    #     if optimizer is not None:
    #         optimizer.zero_grad()
    #
    #     l.backward()
    #     if optimizer is None:
    #         sgd(params, lr, batch_size)
    #     else:
    #         optimizer.step()  # “softmax回归的简洁实现”一节将用到
    #
    #     train_l_sum += l.item()
    #     train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
    #     n += y.shape[0]
    # test_acc = evaluate_accuracy(test_iter, net)
    # print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
    #       % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
    #
    #
