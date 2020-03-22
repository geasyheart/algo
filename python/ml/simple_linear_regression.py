# coding=utf-8
import numpy as np


class SimpleLinearRegression(object):
    """简单线性回归"""

    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        assert x_train.shape[0] == y_train.shape[0], "sample data error."

        # num, d = 0, 0
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        # for x_i, y_i in zip(x_train, y_train):
            # num += (x_i - x_mean) * (y_i - y_mean)
            # d += (x_i - x_mean) ** 2
        # 使用点积的方式
        num = (x_train - x_mean).dot(y_train - y_mean)
        d = (x_train - x_mean).dot(x_train - x_mean)
        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def _predict(self, x):
        return self.a_ * x + self.b_

    def predict(self, x):
        assert x.ndim == 1, "error."
        return np.array([self._predict(i) for i in x])

    def __repr__(self):
        return self.__class__.__name__
