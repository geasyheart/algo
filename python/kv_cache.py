import torch
import torch.nn as nn


class SimpleAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, past_k=None, past_v=None):
        q = self.q_proj(x)  # (B, 1, D)
        k = self.k_proj(x)  # 当前步的 key
        v = self.v_proj(x)  # 当前步的 value

        # 如果之前有缓存
        if past_k is not None and past_v is not None:
            k = torch.cat([past_k, k], dim=1)  # 连接历史的 K
            v = torch.cat([past_v, v], dim=1)  # 连接历史的 V

        attn_weights = self.softmax(q @ k.transpose(-2, -1) / (k.size(-1) ** 0.5))  # QK^T
        output = attn_weights @ v  # attention 输出
        return output, k, v  # 返回新的 K/V 供缓存


# 模拟生成过程
attn = SimpleAttention(dim=4)

# 第一步输入
x1 = torch.rand(1, 1, 4)  # Batch=1, Seq=1, Dim=4
out1, k1, v1 = attn(x1)  # 初次，没有cache

# 第二步输入
x2 = torch.rand(1, 1, 4)
out2, k2, v2 = attn(x2, past_k=k1, past_v=v1)  # 用上一步缓存的k1, v1

print("输出2:", out2)
