import math
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

import matplotlib.pyplot as plt

raw_data_X = [
    [3.393533211, 2.3312733811, ],
    [3.110073483, 1.7815396381, ],
    [1.343808831, 3.3683609541, ],
    [3.582294042, 4.6791791101, ],
    [2.280362439, 2.8669902631, ],
    [7.423436942, 4.6965228751, ],
    [5.745051997, 3.5339898031, ],
    [9.172168622, 2.5111010451, ],
    [7.792783481, 3.4240889411, ],
    [7.939820817, 0.7916372311, ],
]

raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

X_train = np.array(raw_data_X)
y_train = np.array(raw_data_y)
#
# plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], color='g')
# plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color='r')
# plt.show()

# 新的数据
x = np.array([8.093607318, 3.365731514])


# plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], color='g')
# plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color='r')
# plt.scatter(x[0], x[1], color='b')
# plt.show()


# knn过程
class KNNClassifier(object):
    def __init__(self, k: int):
        self.k = k

    def fit(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        assert self.x.shape[0] == self.y.shape[0], "sample not equal."

    def _predict(self, x):
        #  (欧拉距离)
        distances = []
        for x_train in self.x:
            d = math.sqrt(np.sum(x_train - x) ** 2)
            distances.append(d)
        nearest = np.argsort(distances)
        topk_y = [self.y[i] for i in nearest[:self.k]]
        counter = Counter(topk_y)
        return counter.most_common(1)[0][0] if counter else None

    def predict(self, x):
        return np.array([self._predict(_x) for _x in x])


if __name__ == '__main__':
    knn = KNNClassifier(2)
    knn.fit(X_train, y_train)
    rs = knn.predict(x)
    print(rs)

    knn_neighbors = KNeighborsClassifier(n_neighbors=5)
    knn_neighbors.fit(X_train, y_train)
    print(knn_neighbors.predict(x.reshape(1, -1)))
