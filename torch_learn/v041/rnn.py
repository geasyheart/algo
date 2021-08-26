# -*- coding: utf8 -*-
#
from torch.nn import Embedding, Sigmoid, Linear
from torch.nn.modules.rnn import RNN

import torch
from torch.nn import BCELoss
from torch.optim import RMSprop


class RNNModel(torch.nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.embed = Embedding(6, 30)
        self.rnn = RNN(input_size=30, hidden_size=4, batch_first=True, num_layers=10)
        self.linear = Linear(4, 1)

    def forward(self, input):
        input = self.embed(input)
        # rnn_out: (32, 3, 4) 返回batch_size下每个时刻(sequence_length)的最终输出
        # _ : (10, 32, 4) 表示batch_size下每个rnn layer最后一个时刻的输出
        rnn_out, _ = self.rnn(input)
        y_pred = self.linear(_.sum(0))
        return y_pred


class RNNDemo(object):
    def __init__(self):
        self.model = RNNModel()

        self.optim = RMSprop(self.model.parameters())

    def loss(self, y_pred, y_true):
        y_pred = Sigmoid()(y_pred)
        return BCELoss()(y_pred, y_true.float())

    def train(self, x, y):
        for i in range(100):
            for _x, _y in zip(x, y):
                _x = _x.repeat(32, 1)
                _y = _y.repeat(32, 1)
                y_pred = self.model(_x)
                loss = self.loss(y_pred, _y)
                loss.backward()
                print(loss.item())
                self.optim.step()
                self.optim.zero_grad()


if __name__ == '__main__':
    train_data = [
        (['A', 'B', 'C'], 'right'),
        (['C', 'D', 'E'], 'false')
    ]
    id_map = {
        'A': 1,
        'B': 2,
        'C': 3,
        'D': 4,
        'E': 5,
        'right': 6,
        'false': 7

    }
    rnn = RNNDemo()
    rnn.train(
        [torch.tensor([id_map[i] for i in sample[0]]).reshape(1, 3) for sample in train_data],
        [torch.tensor(id_map[sample[1]]).reshape(1, 1) for sample in train_data]
    )

