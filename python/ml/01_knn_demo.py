# coding=utf-8
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

from model_selection import train_test_split
from knn import KNNClassifier
from sklearn.metrics import accuracy_score
from metrics import accuracy_core as accuracy_core_2

iris_data = load_iris()
x_data = iris_data.data
y_target = iris_data.target
x_test, y_test, x_train, y_train = train_test_split(x_data, y_target)
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
x_test_predict = knn.predict(x_test)
print(accuracy_score(y_test, x_test_predict))

knn2 = KNNClassifier(20) # k的值无法自动选取一个合适的,超参数问题
knn2.fit(x_train, y_train)
x_test_predict2 = knn2.predict(x_test)
print(accuracy_core_2(y_test, x_test_predict2))

