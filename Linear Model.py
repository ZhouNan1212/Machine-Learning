# -*- coding: utf-8 -*-
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='myapp.log',
                    filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


class LinearRegressionModel(object):
    def __init__(self):
        # read_csv里面的参数是csv在你电脑上的路径，此处csv文件放在notebook运行目录下面的CCPP目录里
        data = pd.read_csv('./CCPP/test.csv')
        # 读取前五行数据，如果是最后五行，用data.tail()
        x, y = data[['AT', 'V', 'AP', 'RH']], data['PE']
        self.x, self.y, self.data = x, y, data
        logging.info("初始化完成")
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

    @staticmethod
    def linear_regression_model(data_train, result_train, data_test, result_test):
        model = LinearRegression()
        model.fit(data_train, result_train)
        print(model.intercept_)
        print(model.coef_)
        result_predict = model.predict(data_test)
        logging.info("=============")
        """
        用scikit-learn计算MSE和RMSE
        均方差（Mean Squared Error, MSE）
        均方根差(Root Mean Squared Error, RMSE)
        """
        print("MSE:", metrics.mean_squared_error(result_test, result_predict))
        print("RMSE:", np.sqrt(metrics.mean_squared_error(result_test, result_predict)))

    def cross_validation(self):
        model = LinearRegression()
        x, y = self.x, self.y
        predicted = cross_val_predict(model, x, y, cv=10)
        print("MSE:", metrics.mean_squared_error(y, predicted))
        print("RMSE:", np.sqrt(metrics.mean_squared_error(y, predicted)))
        logging.info("-------------")
        return y, predicted

    @staticmethod
    def plot(x, y):
        fig, ax = plt.subplots()
        ax.scatter(x, y)
        ax.plot([x.min(), x.max()], [x.min(), x.max()], 'k--', lw=4)
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')
        plt.show()
        # plt.savefig("test.png")


class RidgeRegressionModel(LinearRegressionModel):
    @staticmethod
    def ridge_regression(data_train, result_train, data_test, result_test):
        """
        :param data_train: 训练集数据
        :param result_train: 训练集的结果
        :param data_test: 测试集数据
        :param result_test: 测试集结果
        :return:
        """

        """
        alpha：{float，array-like}，shape（n_targets）
        正则化强度; 必须是正浮点数。 正则化改善了问题的条件并减少了估计的方差。 
        较大的值指定较强的正则化。 Alpha对应于其他线性模型（如Logistic回归或LinearSVC）中的C^-1。 
        如果传递数组，则假定惩罚被特定于目标。 因此，它们必须在数量上对应。

        copy_X：boolean，可选，默认为True.如果为True，将复制X; 否则，它可能被覆盖。

        fit_intercept：boolean 是否计算此模型的截距。 如果设置为false，则不会在计算中使用截距（例如，数据预期已经居中）。

        max_iter：int，可选.共轭梯度求解器的最大迭代次数。 
        对于'sparse_cg'和'lsqr'求解器，默认值由scipy.sparse.linalg确定。对于'sag'求解器，默认值为1000。

        normalize：boolean，可选，默认为False.如果为真，则回归X将在回归之前被归一化。
        当fit_intercept设置为False时，将忽略此参数。当回归量归一化时，注意到这使得超参数学习更加鲁棒，
        并且几乎不依赖于样本的数量。相同的属性对标准化数据无效。然而，如果你想标准化，
        请在调用normalize = False训练估计器之前，使用preprocessing.StandardScaler处理数据。

        solver：{'auto'，'svd'，'cholesky'，'lsqr'，'sparse_cg'，'sag'，'saga'}用于计算的求解方法：
            'auto'根据数据类型自动选择求解器。

            'svd'使用X的奇异值分解来计算Ridge系数。对于奇异矩阵比'cholesky'更稳定。

            'cholesky'使用标准的scipy.linalg.solve函数来获得闭合形式的解。

            'sparse_cg'使用在scipy.sparse.linalg.cg中找到的共轭梯度求解器。作为迭代算法，
            这个求解器比大规模数据（设置tol和max_iter的可能性）的“cholesky”更合适。

            'lsqr'使用专用的正则化最小二乘常数scipy.sparse.linalg.lsqr。它是最快的，
            但可能不是在旧的scipy版本可用。它还使用迭代过程。

            'sag'使用随机平均梯度下降。它也使用迭代过程，并且当n_samples和n_feature都很大时，
            通常比其他求解器更快。注意，“sag”快速收敛仅在具有近似相同尺度的特征上被保证。
            可以使用sklearn.preprocessing的缩放器预处理数据。

        所有最后四个求解器支持密集和稀疏数据。但是，当fit_intercept为True时，只有'sag'支持稀疏输入。
        新版本0.17支持：随机平均梯度下降解算器。
        新版本0.19支持：SAGA解算器。

        tol：float解的精度。

        random_state：int seed，RandomState实例或None（默认）伪随机数生成器的种子，当混洗数据时使用。 
        仅用于'sag'求解器。
        """
        model = Ridge(alpha=1.0,
                      fit_intercept=True,
                      normalize=False,
                      copy_X=True,
                      max_iter=None,
                      tol=0.001,
                      solver='auto',
                      random_state=None)
        model.fit(data_train, result_train)
        result_predict = model.predict(data_test)
        logging.info("计算完成")
        """
        用scikit-learn计算MSE和RMSE
        均方差（Mean Squared Error, MSE）
        均方根差(Root Mean Squared Error, RMSE)
        """
        print("MSE:", metrics.mean_squared_error(result_test, result_predict))
        print("RMSE:", np.sqrt(metrics.mean_squared_error(result_test, result_predict)))


if __name__ == '__main__':
    s = RidgeRegressionModel()
    data_train, data_test, result_train, result_test = s.split_dataset()
    s.ridge_regression(data_train, result_train, data_test, result_test)
    # s.plot(real_result, predicted_result)
    # linear_regression_model(x_train, y_train)
    # 模型拟合测试集
