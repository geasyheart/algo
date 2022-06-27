# -*- coding: utf8 -*-
#
from typing import Optional, Tuple

import torch
from torch import nn


class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, layer_norm: bool = False):
        super(LSTMCell, self).__init__()

        self.hidden_lin = nn.Linear(hidden_size, 4 * hidden_size)

        self.input_lin = nn.Linear(input_size, 4 * hidden_size, bias=False)

        if layer_norm:
            self.layer_norm = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(4)])
            self.layer_norm_c = nn.LayerNorm(hidden_size)
        else:
            self.layer_norm = nn.ModuleList([nn.Identity() for _ in range(4)])
            self.layer_norm_c = nn.Identity()

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        ifgo = self.hidden_lin(h) + self.input_lin(x)
        ifgo = ifgo.chunk(4, dim=-1)

        ifgo = [self.layer_norm[i](ifgo[i]) for i in range(4)]
        i, f, g, o = ifgo

        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)

        h_next = torch.sigmoid(o) * torch.tanh(self.layer_norm_c(c_next))
        return h_next, c_next


class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, n_layers: int):
        super(LSTM, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.cells = nn.ModuleList(
            [LSTMCell(input_size, hidden_size)] +
            [LSTMCell(input_size, hidden_size) for _ in range(n_layers - 1)]
        )

    def forward(self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        State
        """

        n_steps, batch_size = x.shape[:2]
        if state is None:
            h = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.n_layers)]
            c = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.n_layers)]
        else:
            (h, c) = state
            h, c = list(torch.unbind(h)), list(torch.unbind(c))
        out = []
        for t in range(n_steps):
            inp = x[t]
            # 下一层的输入来自上层的输出
            for layer in range(self.n_layers):
                h[layer], c[layer] = self.cells[layer](inp, h[layer], c[layer])
                inp = h[layer]
            out.append(h[-1])

        out = torch.stack(out)
        h = torch.stack(h)
        c = torch.stack(c)
        return out, (h, c)


if __name__ == '__main__':
    i = torch.randn(128, 32, 111)
    lstm = LSTM(111, 222, 1)
    lstm(i)
