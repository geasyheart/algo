# coding=utf-8

"""
通过网格搜索的方式进行获取最优解
"""

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
data = load_iris()
x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

knn = KNeighborsClassifier()

param_grid = [
    {
        "weights":['uniform'],
        "n_neighbors":[i for i in range(1, 11)]
    },
    {
        "weights": ['distance'],
        "n_neighbors":[i for i in range(1, 11)],
        "p": [i for i in range(1, 6)]
    }
]

grid_search = GridSearchCV(knn, param_grid, n_jobs=-1) # -1 表示使用所有的cpu，默认为1
grid_search.fit(x_train, y_train)

print("最佳的分类器是: ", grid_search.best_estimator_, grid_search.best_score_)


x_test_predict = grid_search.best_estimator_.predict(x_test)
score = accuracy_score(y_test, x_test_predict)
print(score)
