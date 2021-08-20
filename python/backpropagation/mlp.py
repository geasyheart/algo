# -*- coding: utf8 -*-
#
from random import random

import numpy as np


class MLP(object):
    def __init__(self, num_inputs: int = 3, hidden_layers=[3, 3], num_outputs: int = 2):
        self.num_inputs = num_inputs
        self.hidden_layer = hidden_layers
        self.num_outputs = num_outputs

        layers = [num_inputs] + hidden_layers + [num_outputs]

        # 创建随机权重，表示多个linear的权重汇总
        weights = []

        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            weights.append(w)
        self.weights = weights

        #
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

        #
        derivatives = []

        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(d)

        self.derivatives = derivatives

    def forward_propagate(self, inputs):
        """
        前向过程，activations保存每一层linear的结果,公式y = sigmoid(wx)
        """
        activations = inputs

        self.activations[0] = inputs

        for i, w in enumerate(self.weights):
            net_inputs = np.dot(activations, w)
            activations = self._sigmoid(net_inputs)
            self.activations[i + 1] = activations
        return activations

    def back_propagate(self, error, verbose=False):
        """
        反向过程，

        error = error * sigmoid导数 * 上一层的结果
        derivative = W(i) * sigmoid导数 * 上一层的结果

        """

        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i + 1]
            delta = error * self._sigmoid_derivative(activations)
            delta_reshaped = delta.reshape(delta.shape[0], -1).T
            current_activations = self.activations[i]
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)
            self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)
            # 权重过大或过小会导致梯度消失或者爆炸
            error = np.dot(delta, self.weights[i].T)

            if verbose:
                print("Derivatives for W{}: {}".format(i, self.derivatives[i]))
        return error

    def gradient_descent(self, learning_rate):
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * learning_rate

    def train(self, inputs, targets, epochs, learning_rate):
        for i in range(epochs):
            sum_error = 0
            for input, target in zip(inputs, targets):
                output = self.forward_propagate(input)
                sum_error += self._mse(target, output)

                error = target - output
                self.back_propagate(error)
                self.gradient_descent(learning_rate)

            print('Error: {} at Epoch {}'.format(sum_error / len(inputs), i))

    def _mse(self, target, output):
        return np.average((target - output) ** 2)

    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)

    def _sigmoid(self, x):
        y = 1.0 / (1 + np.exp(-x))
        return y


if __name__ == '__main__':
    """
    w: [2 * 5, 5 * 1] rand
    a: [2, 5, 1]      zero
    d: [2 * 5, 5 * 1] zero
    
    """
    # 预测两个数相加
    inputs = np.array([[random() / 2 for _ in range(2)] for _ in range(1000)])
    target = np.array([[i[0] + i[1]] for i in inputs])

    mlp = MLP(2, [5], 1)
    mlp.train(inputs, target, 50, 0.1)

    input = np.array([0.3, 0.1])
    target = np.array([0.4])
    output = mlp.forward_propagate(input)
    print('Our network believes that {} + {} equal to {}'.format(input[0], input[1], output[0]))
