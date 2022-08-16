# -*- coding: utf8 -*-
# https://www.jb51.net/article/213383.htm
import numpy as np
import torch
from torch.functional import F

a = np.array([[1, 2, 4, 1],
              [6, 3, 2, 4],
              [2, 4, 6, 1]])

v = a.var(axis=1)  # 方差
# print(v, a.std(axis=1))
m = a.mean(axis=1)  # 平均值

v = v.reshape(3, 1)
m = m.reshape(3, 1)

# print(e, m)
# layer norm
result = (a - m) / np.sqrt(v + 1e-5)
print(result)
print(torch.nn.LayerNorm(4)(torch.from_numpy(a).float()))

print(F.layer_norm(torch.from_numpy(a).float(), [4]))

# 另外进行引伸，为什么BN无法在RNN上应用??
# 1. 方向不同，BN是在batch_size那一维做，另外不能代表整体的normalization
# 2. 每个batch的长度不一样，引入mask后有影响
