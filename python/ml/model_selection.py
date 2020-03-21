# coding=utf-8

from sklearn.datasets import load_iris
import numpy as np


# shuffle_indexes = np.random.permutation(x_data.shape[0])
#
# test_radio = 0.2
# test_size = int(len(x_data) * test_radio)
#
# test_indexes = shuffle_indexes[:test_size]
# train_indexes = shuffle_indexes[test_size:]
#
# x_train = x_data[train_indexes]
# y_train = y_target[train_indexes]
#
# x_test = x_data[test_indexes]
# y_test = y_target[test_indexes]
# print(x_test)


def train_test_split(x, y, test_radio=0.2, seed=None):

    assert x.shape[0] == y.shape[0], "sample size not equal."
    # 随机种子
    if seed:
        np.random.seed(seed)

    shuffle_indexes = np.random.permutation(len(x))
    test_size = int(len(x) * test_radio)
    test_indexes = shuffle_indexes[:test_size]
    train_indexes = shuffle_indexes[test_size:]
    # 人家直接可以取出来。。。
    return x[test_indexes], y[test_indexes], x[train_indexes], y[train_indexes]


#
