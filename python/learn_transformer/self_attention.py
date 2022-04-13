# -*- coding: utf8 -*-
#
import torch
from torch import nn
from transformers.models.bert.modeling_bert import BertSelfAttention


class Config(object):
    hidden_size = 4
    num_attention_heads = 1
    attention_probs_dropout_prob = 0.1


attn = BertSelfAttention(Config())
attn.query = nn.Identity()
attn.key = nn.Identity()
attn.value = nn.Identity()

if __name__ == '__main__':
    # 比如
    # 我： [1,2,3,4]
    # 你: [5,6,7,8]
    # 这里表示一个batch
    # 可以看到attention的计算
    input = torch.tensor([[1., 2., 3., 4.], [5., 6., 7., 8.]]).reshape(1, 2, 4)
    attn(input)

