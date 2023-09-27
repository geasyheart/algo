# -*- coding: utf8 -*-
#
import numpy as np


def sample(preds, temperature=1.0):
    """
    设置不同的温度进行采样
    """
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds)
    return np.argmax(probas)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference


if __name__ == '__main__':
    a = np.random.rand(1, 10).reshape(-1)
    b = softmax(a)
    c = sample(b)
    print(c)
