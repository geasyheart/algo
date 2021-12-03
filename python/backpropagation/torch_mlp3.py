# -*- coding: utf8 -*-
#
from random import random

import torch
from torch import nn
from torch.optim import SGD


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.f1 = nn.Linear(1, 10)
        self.f2 = nn.Linear(1, 10)
        self.out = nn.Linear(20, 1)

    def forward(self, input1, input2):
        out1 = self.f1(input1)
        out2 = self.f2(input2)
        out = torch.cat([out1, out2], dim=-1)
        return self.out(out)


if __name__ == '__main__':

    inputs = [[random() / 2 for _ in range(2)] for _ in range(1000)]
    targets = [[i[0] + i[1]] for i in inputs]

    model = Model()
    loss_func = nn.MSELoss()
    optimizer = SGD(params=model.parameters(), lr=1e-3)

    for epoch in range(1, 101):
        total_loss = 0
        for input, target in zip(inputs, targets):
            out = model(
                torch.tensor(input[0]).reshape(1, -1),
                torch.tensor(input[1]).reshape(1, -1),
            )
            loss = loss_func(out, torch.tensor(target).float().reshape(1, -1))
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f'Epoch {epoch} Loss {total_loss}')

    with torch.no_grad():
        model.eval()
        out = model(
            torch.tensor(0.33).reshape(1,-1),
            torch.tensor(0.47).reshape(1,-1),

        )
        print(f'predict result: {out}')
