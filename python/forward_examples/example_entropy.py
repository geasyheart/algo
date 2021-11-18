# -*- coding: utf8 -*-
#
import torch
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co

from python.forward_examples.linear_layer import MLP
from python.forward_examples.loss import cross_entropy_loss


class MyDataSet(Dataset):
    def __init__(self):
        self.datas = [i for i in range(100000)]

    def __getitem__(self, index) -> T_co:
        sample = self.datas[index]
        if sample % 2 == 0:
            return torch.tensor([sample], dtype=torch.float), torch.tensor(2, dtype=torch.long)
        elif sample % 3 == 0:
            return torch.tensor([sample], dtype=torch.float), torch.tensor(3, dtype=torch.long)
        else:
            return torch.tensor([sample], dtype=torch.float), torch.tensor(1, dtype=torch.long)

    def __len__(self):
        return len(self.datas)

    def to_dataloader(self):
        return DataLoader(self, batch_size=32)


def torch_imp_cross_entropy_loss(y_pred, y_true):
    loss_func = torch.nn.CrossEntropyLoss()
    y_pred = torch.log_softmax(y_pred, dim=-1)
    return loss_func(y_pred, y_true)


def train():
    d = MyDataSet().to_dataloader()

    mlp = MLP(in_feature=1, out_feature=3)
    mlp.train()

    optim = SGD(mlp.parameters(), lr=1.)

    for i in range(10):
        total_loss = 0
        for x, y in d:
            y_pred = mlp(x)
            y -= 1  # 为什么要减1,因为torch是从0开始算起，所以label[1,2,3]应该为[0,1,2]
            loss = cross_entropy_loss(y_pred, y)
            torch_loss = torch_imp_cross_entropy_loss(y_pred, y)
            if round(loss.item(), 4) != round(torch_loss.item(), 4):
                raise ValueError(f'loss not equal torch cross loss, {loss.item()}, {torch_loss.item()}')
            total_loss += loss.item()
            loss.backward()

            optim.step()
            optim.zero_grad()
        print(f'Epoch {i} , loss {total_loss}')


if __name__ == '__main__':
    train()
