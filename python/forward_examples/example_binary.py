# -*- coding: utf8 -*-
#
import torch
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader

from torch.utils.data.dataset import T_co

from python.forward_examples.linear_layer import MLP
from python.forward_examples.loss import bce_loss_with_logit


# 任务目标，二分类
# 如果为单数，则为负样本，否则正样本

class MyDataSet(Dataset):
    def __init__(self):
        self.datas = [i for i in range(100000)]

    def __getitem__(self, index) -> T_co:
        sample = self.datas[index]
        if sample % 2 == 0:
            return torch.tensor([sample], dtype=torch.float), torch.tensor([1], dtype=torch.float)
        return torch.tensor([sample], dtype=torch.float), torch.tensor([0], dtype=torch.float)

    def __len__(self):
        return len(self.datas)

    def to_dataloader(self):
        return DataLoader(self, batch_size=32)


def train():
    d = MyDataSet().to_dataloader()

    mlp = MLP(in_feature=1, out_feature=1)
    mlp.train()

    optim = SGD(mlp.parameters(), lr=1.)

    for i in range(10):
        total_loss = 0
        for x, y in d:
            y_pred = mlp(x)

            loss = bce_loss_with_logit(y_pred, y)
            total_loss += loss.item()
            loss.backward()

            optim.step()
            optim.zero_grad()
        print(f'Epoch {i} , loss {total_loss}')


if __name__ == '__main__':
    train()
