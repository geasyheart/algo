# -*- coding: utf8 -*-
#
import torch
from random import random
from torch import nn
from torch.optim import SGD


class Model(torch.nn.Module):
    def __init__(self, n_in, n_out):
        super(Model, self).__init__()
        self.embed = nn.Embedding(n_in+1, embedding_dim=10)
        self.out = nn.Linear(20, n_out)

    def forward(self, input1, input2):
        out1 = self.embed(input1)
        out2 = self.embed(input2)
        out = torch.cat([out1, out2], dim=-1)
        return self.out(out)


if __name__ == '__main__':

    inputs = [[random() / 2 for _ in range(2)] for _ in range(1000)]
    targets = [[i[0] + i[1]] for i in inputs]

    input_map = {}
    for lst in inputs:
        for ele in lst:
            input_map[ele] = len(input_map) + 1

    target_map = {}
    for lst in targets:
        for ele in lst:
            target_map[ele] = len(target_map) + 1

    model = Model(len(input_map), n_out=1)
    loss_func = nn.MSELoss()
    optimizer = SGD(params=model.parameters(), lr=1e-5)

    for epoch in range(1, 101):
        total_loss = 0
        for input, target in zip(inputs, targets):
            input_id = [input_map[_] for _ in input]
            target_id = [target_map[_] for _ in target]

            out = model(
                torch.tensor(input_id[0]),
                torch.tensor(input_id[1]),
            )
            loss = loss_func(out, torch.tensor(target_id).float())
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f'Epoch {epoch} Loss {total_loss}')

    with torch.no_grad():
        model.eval()
        # FIXME: 你觉得这个地方应该怎么写，全都固定了，都在俩 *_map 里面了。。。
