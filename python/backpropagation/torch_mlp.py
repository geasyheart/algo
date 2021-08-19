# -*- coding: utf8 -*-
#
from random import random

import torch
from torch import nn
from torch.optim import SGD



class MLPModel(nn.Module):
    def __init__(self, n_in=2, n_out=1):
        super(MLPModel, self).__init__()

        self.s = nn.Linear(n_in, 5)
        self.o = nn.Linear(5, n_out)

    def forward(self, x):
        return self.o(self.s(x))


class MLP(object):
    def __init__(self):
        self.model = MLPModel()

    def loss(self, y_pred, y_true, loss_func):
        return loss_func(y_pred, y_true)

    def train(self, x, y, epochs, lr):
        self.model.train()
        loss_func = nn.MSELoss()
        optimizer = SGD(params=self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            total_loss = 0
            for input, target in zip(x, y):
                # input = input.reshape(1, input.shape[0])
                # target = target.reshape(1, target.shape[0])
                pred = self.model(input)
                loss = self.loss(pred, target, loss_func)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            print(f'Epoch {epoch} Loss {total_loss}')

    @torch.no_grad()
    def predict(self, x):
        self.model.eval()
        return self.model(x)


inputs = torch.tensor([[random() / 2 for _ in range(2)] for _ in range(1000)])
targets = torch.tensor([[i[0] + i[1]] for i in inputs])

mlp = MLP()
mlp.train(inputs, targets, epochs=10, lr=0.1)

input = torch.tensor([0.3, 0.1])
pred = mlp.predict(input)
print('Our network believes that {} + {} equal to {}'.format(input[0], input[1], 0.4))
