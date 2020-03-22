# coding=utf-8
"""
线性回归
"""

import numpy as np
from matplotlib import pyplot as plt
from simple_linear_regression import SimpleLinearRegression
x = np.array([1., 2., 3., 4., 5.0])
y = np.array([1., 3., 2., 4., 5.0])

plt.scatter(x, y)


simple_linear_regression = SimpleLinearRegression()
simple_linear_regression.fit(x, y)
print(simple_linear_regression.predict(np.array([1.1])))
y_predict = simple_linear_regression.a_ * x + simple_linear_regression.b_
plt.plot(x, y_predict, color='r')
plt.show()
