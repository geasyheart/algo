# coding=utf-8

"""
数据归一化
"""

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
iris_data = load_iris()

x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.33)
standard_scaler = StandardScaler()

standard_scaler.fit(x_train)
standard_scaler_train = standard_scaler.transform(x_train)


knn = KNeighborsClassifier()
knn.fit(standard_scaler_train, y_train)

standard_scaler.fit(x_test)
x_test_standard = standard_scaler.transform(x_test)
y_predict = knn.predict(x_test_standard)
#
score = accuracy_score(y_test, y_predict)
print(score)
