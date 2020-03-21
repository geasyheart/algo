# coding=utf-8


def accuracy_core(y_true, y_predict):
    """准确度值"""
    assert y_true.shape[0] == y_predict.shape[0], "sample data error."
    return sum(y_true == y_predict) / len(y_predict)
