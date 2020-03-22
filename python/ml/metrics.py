# coding=utf-8
import numpy as np


def accuracy_core(y_true, y_predict):
    """准确度值"""
    assert y_true.shape[0] == y_predict.shape[0], "sample data error."
    return sum(y_true == y_predict) / len(y_predict)


def mse(y_predict, y_test):
    return np.sum((y_predict - y_test) ** 2) / len(y_test)


def rms(y_predict, y_test):
    mse_test = mse(y_predict=y_predict, y_test=y_test)
    return np.math.sqrt(mse_test)


def mae(y_predict, y_test):
    return np.sum(np.absolute(y_predict - y_test)) / len(y_test)


def r2_square(y_predict, y_test):
    """R Square"""
    return 1 - mse(y_predict, y_test) / np.var(y_test)
