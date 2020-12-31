# -*- coding: utf8 -*-

# 简单 RNN 的 Numpy 实现

import numpy as np

timesteps = 100  # 输入序列的时间步数
input_features = 32  # 输入特征空间的维度
output_features = 64  # 输出特征空间的维度
inputs = np.random.random((timesteps, input_features))
state_t = np.zeros((output_features,))  # 初始状态
W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features,))
successive_outputs = []
for input_t in inputs:
    # output_t = activation(dot(W, input_t) + dot(U, state_t) + b)
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
    successive_outputs.append(output_t)
    state_t = output_t
final_output_sequence = np.stack(successive_outputs, axis=0)  # shape: 100 × 64

# return_sequence表示返回final_output_sequence最后一层的信息，即最后一层（最后一个的timesteps）就包含了整个timesteps的所有信息。
