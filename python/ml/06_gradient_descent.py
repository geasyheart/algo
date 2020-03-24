# coding=utf-8

"""
梯度随机下降
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

plot_x = np.linspace(-1, 6, 141).reshape(-1, 1)
plot_y = ((plot_x - 2.5) ** 2 - 1).reshape(-1, 1)
# print(plot_x.shape)

plt.plot(plot_x, plot_y)
# plt.show()

# 预处理
X_train, X_test, y_train, y_test = train_test_split(
    plot_x, plot_y, test_size=0.33, random_state=42)

standard_scaler = StandardScaler()
standard_scaler.fit(X_train, y_train)
X_train = standard_scaler.transform(X_train)

standard_scaler.fit(X_test, y_test)
X_test = standard_scaler.transform(X_test)

sgd_reg = SGDRegressor(n_iter_no_change=1e5, max_iter=1e6, verbose=2)
sgd_reg.fit(X_train, y_train)
# plt.show()
