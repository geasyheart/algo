# coding=utf-8

"""
衡量线性回归法的指标 MSE RMS MAE

"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

from metrics import mse, rms, mae
from simple_linear_regression import SimpleLinearRegression

# 此处使用boston中的CM来简单模拟线性回归
boston_data = load_boston()
# print(boston_data.feature_names)
x_train = boston_data.data[:, 5]
y_target = boston_data.target

# plt.scatter(x_train, y_target)
# plt.show()

# 此时绘图会发现有好多50的点，引出此问题的原因可能在于50为整个数据集的上限，那么这里把==50的点去掉

x_train = x_train[y_target < 50]
y_target = y_target[y_target < 50]

x_train, x_test, y_train, y_test = train_test_split(x_train, y_target)
# 再次绘图
plt.scatter(x_train, y_train)
# plt.show()

simple_linear_regression = SimpleLinearRegression()
simple_linear_regression.fit(x_train, y_train)
plt.plot(x_train, simple_linear_regression.a_ * x_train + simple_linear_regression.b_, color='r')

# plt.show()

# 进行预测
y_predict = simple_linear_regression.predict(x_test)

# 衡量线性回归法的指标 MSE RMS MAE

# 1. 使用mse进行衡量
print(mse(y_predict, y_test))

# 2. RMS # 表示在RMS的衡量标准上误差在多少美元
print(rms(y_predict, y_test))

# 3. mae #  表示在MAE的衡量标准上误差在多少美元

print(mae(y_predict, y_test))
