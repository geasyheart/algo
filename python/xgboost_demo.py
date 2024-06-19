import dtreeviz
import pandas as pd
import numpy as np
import shap
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt


def reg_demo():
    """
    回归模型
    """
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

    xgb_reg = xgb.XGBRegressor()
    xgb_reg.fit(X_train, y_train)
    y_pred = xgb_reg.predict(X_test)

    # 计算R平方和MSE
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print('R^2: {:.2f}'.format(r2))
    print('MSE: {:.2f}'.format(mse))

    xgb.plot_tree(xgb_reg)
    plt.show()


def cls_demo():
    """
    分类
    """
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
    xgb_cls = xgb.XGBClassifier(eval_metric='mlogloss', n_estimators=10)
    xgb_cls.fit(X_train, y_train)
    y_pred = xgb_cls.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print('Accuracy: {:.2f}'.format(accuracy))

    print(classification_report(y_test, y_pred, ))
    # plt.figure(figsize=(20, 10))
    print(iris.feature_names)
    viz_model = dtreeviz.model(xgb_cls, X_train=X_train, y_train=y_train,
                               tree_index=1,
                               feature_names=iris.feature_names,
                               # feature_names=['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'],
                               target_name='iris',
                               class_names=iris.target_names,
                               )
    v = viz_model.view()
    v.show()


def cls_demo2():
    """分类

    更多细看dtreeviz功能
    """

    # dataset_url = "https://raw.githubusercontent.com/parrt/dtreeviz/master/data/titanic/titanic.csv"
    dataset_url = '/home/yuzhang/windows_share/2024/titanic.csv'
    dataset = pd.read_csv(dataset_url, )
    # Fill missing values for Age
    dataset.fillna({"Age": dataset.Age.mean()}, inplace=True)
    # Encode categorical variables
    dataset["Sex_label"] = dataset.Sex.astype("category").cat.codes
    dataset["Cabin_label"] = dataset.Cabin.astype("category").cat.codes
    dataset["Embarked_label"] = dataset.Embarked.astype("category").cat.codes

    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)


    features = ["Pclass", "Age", "Fare", "Sex_label", "Cabin_label", "Embarked_label"]
    target = "Survived"

    dtrain = xgb.DMatrix(train_dataset[features], train_dataset[target])
    dtest = xgb.DMatrix(test_dataset[features], test_dataset[target])

    params = {"max_depth": 3, "eta": 0.05, "objective": "binary:logistic", 'eval_metric': 'logloss'}
    bst = xgb.train(params=params, dtrain=dtrain, num_boost_round=8)

    # 创建SHAP解释器
    explainer = shap.Explainer(bst)

    # 计算SHAP值
    shap_values = explainer(test_dataset[features])

    # 可视化单个样本的SHAP值
    shap.plots.waterfall(shap_values[0])

    # 可视化所有样本的SHAP值
    shap.summary_plot(shap_values, test_dataset[features])

    # 可视化特征重要性
    shap.summary_plot(shap_values, test_dataset[features], plot_type="bar")



    # viz_model = dtreeviz.model(bst, tree_index=1,
    #                            X_train=dataset[features], y_train=dataset[target],
    #                            feature_names=features,
    #                            target_name=target, class_names=["perish", "survive"])
    #
    # # v = viz_model.view(fancy=False)
    # # v.show()
    # x = dataset[features].iloc[10]
    # v = viz_model.view(x=x)
    # v.show()


if __name__ == '__main__':
    # reg_demo()
    cls_demo2()
