# coding=utf-8
"""
线性回归
"""

import numpy as np
from matplotlib import pyplot as plt
from simple_linear_regression import SimpleLinearRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
x = np.array([1., 2., 3., 4., 5.0])
y = np.array([1., 3., 2., 4., 5.0])

plt.scatter(x, y)


simple_linear_regression = SimpleLinearRegression()
simple_linear_regression.fit(x, y)
print(simple_linear_regression.predict(np.array([1.1])))
y_predict = simple_linear_regression.a_ * x + simple_linear_regression.b_
plt.plot(x, y_predict, color='r')
# plt.show()


# 使用正规方程预测
boston = load_boston()
X, y = boston.data, boston.target

X = X[y < 50]
y = y[y < 50]

# X_train, X_test, y_train, y_test = train_test_split(
#        x_train, y_train)

linear = LinearRegression()
linear.fit(X, y)
# 正数表示此房价与此feature成正比关系
print(pd.DataFrame({"feature": boston.feature_names, "coef": linear.coef_}))
