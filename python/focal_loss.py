# -*- coding: utf8 -*-
#
from torch import nn
import torch
from torch.functional import F


class FocalLoss(nn.Module):
    """
    Softmax and sigmoid focal loss.
    copy from https://github.com/lonePatient/TorchBlocks
    """

    def __init__(self, num_labels, activation_type='softmax', gamma=2.0, alpha=0.25, epsilon=1.e-9):

        super(FocalLoss, self).__init__()
        self.num_labels = num_labels
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.activation_type = activation_type

    def forward(self, input, target):
        """
        Args:
            logits: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
        Returns:
            shape of [batch_size]
        """
        if self.activation_type == 'softmax':
            idx = target.view(-1, 1).long()
            one_hot_key = torch.zeros(idx.size(0), self.num_labels, dtype=torch.float32, device=idx.device)
            one_hot_key = one_hot_key.scatter_(1, idx, 1)
            logits = torch.softmax(input, dim=-1)
            loss = -self.alpha * one_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
            loss = loss.sum(1)
        elif self.activation_type == 'sigmoid':
            multi_hot_key = target
            logits = torch.sigmoid(input)
            zero_hot_key = 1 - multi_hot_key
            loss = -self.alpha * multi_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
            loss += -(1 - self.alpha) * zero_hot_key * torch.pow(logits, self.gamma) * (1 - logits + self.epsilon).log()
        else:
            raise NotImplementedError
        return loss.mean()


def my_cross_entropy(input, target, reduction="mean"):
    # input.shape: torch.size([-1, class])
    # target.shape: torch.size([-1])
    # reduction = "mean" or "sum"
    # input是模型输出的结果，与target求loss
    # target的长度和input第一维的长度一致
    # target的元素值为目标class
    # reduction默认为mean，即对loss求均值
    # 还有另一种为sum，对loss求和

    # 这里对input所有元素求exp
    exp = torch.exp(input)
    # 根据target的索引，在exp第一维取出元素值，这是softmax的分子
    tmp1 = exp.gather(1, target.unsqueeze(-1)).squeeze()
    # 在exp第一维求和，这是softmax的分母
    tmp2 = exp.sum(1)
    # softmax公式：ei / sum(ej)
    softmax = tmp1 / tmp2
    # cross-entropy公式： -yi * log(pi)
    # 因为target的yi为1，其余为0，所以在tmp1直接把目标拿出来，
    # 公式中的pi就是softmax的结果
    log = -torch.log(softmax)
    # 官方实现中，reduction有mean/sum及none
    # 只是对交叉熵后处理的差别
    if reduction == "mean":
        return log.mean()
    else:
        return log.sum()

if __name__ == '__main__':
    # 比如预测的很准的情况下
    input = torch.tensor([[0.1, 0.9], [0.9, 0.1]])
    target = torch.tensor([1, 0])
    print(FocalLoss(2)(input, target))
    # 预测不准的情况下
    input = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    target = torch.tensor([1, 0])
    print(FocalLoss(2)(input, target))

    # 总结，看torch.pow((1 - logits), self.gamma)这里
    # 可以看到，当预测准的情况下，1-logits就会变得很小，那么最终loss就会很小（因为有one_hot_key，所以只会关注正确的标签的logits）
    # 预测不准的情况下,1-logits就会变得很大，torch.pow本质来讲就是选择一个gamma来做平滑，其他项暂跳过...