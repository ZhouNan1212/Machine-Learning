# -*- coding: utf-8 -*-
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class LinearRegressionModel(object):
    def __init__(self):
        # read_csv里面的参数是csv在你电脑上的路径，此处csv文件放在notebook运行目录下面的CCPP目录里
        data = pd.read_csv('./CCPP/test.csv')
        # 读取前五行数据，如果是最后五行，用data.tail()
        x, y = data[['AT', 'V', 'AP', 'RH']], data['PE']
        self.x, self.y, self.data = x, y, data
        self.model = LinearRegression()
        """
        print(x.head())  # 输出特征列的前5行
        print(y.head())  # 输出结果的前5行
        """

    def split_dataset(self):
        x, y, data = self.x, self.y, self.data
        """
        test_size: 样本占比，如果是整数的话就是样本的数量；

        random_state: 随机数的种子，不同的种子会造成不同的随机采样结果，相同的种子采样结果相同。

        stratify: 是为了保持split前类的分布。training集和testing集的类的比例是A：B= 4：1，
        等同于split前的比例（80：20）。通常在这种类分布不平衡的情况下会用到stratify。
        这里的输入应该是待划分的结果lable列表

        shuffle: 是否在分割之前对原数据进行洗牌
        """

        data_train, data_test, result_train, result_test = train_test_split(x, y, test_size=0.33, random_state=42)
        return data_train, data_test, result_train, result_test

    def linear_regression_model(self, data_train, result_train, data_test, result_test):
        model = self.model
        model.fit(data_train, result_train)
        print(model.intercept_)
        print(model.coef_)

        result_predict = model.predict(data_test)

        """
        用scikit-learn计算MSE和RMSE
        均方差（Mean Squared Error, MSE）
        均方根差(Root Mean Squared Error, RMSE)
        """
        print("MSE:", metrics.mean_squared_error(result_test, result_predict))
        print("RMSE:", np.sqrt(metrics.mean_squared_error(result_test, result_predict)))

    def cross_validation(self):
        model, x, y = self.model, self.x, self.y
        predicted = cross_val_predict(model, x, y, cv=10)
        print("MSE:", metrics.mean_squared_error(y, predicted))
        print("RMSE:", np.sqrt(metrics.mean_squared_error(y, predicted)))
        return y, predicted

    def plot(self, x, y):
        fig, ax = plt.subplots()
        ax.scatter(x, y)
        ax.plot([x.min(), x.max()], [x.min(), x.max()], 'k--', lw=4)
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')
        plt.show()
        # plt.savefig("test.png")

if __name__ == '__main__':
    s = LinearRegressionModel()
    real_result, predicted_result = s.cross_validation()
    s.plot(real_result, predicted_result)
    # linear_regression_model(x_train, y_train)
    # 模型拟合测试集



