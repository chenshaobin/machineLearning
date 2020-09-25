#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
sns.set(context="notebook", style="whitegrid", palette="dark")
import matplotlib.pyplot as plt
# import tensorflow as tf
import numpy as np

data = pd.read_csv('ex1data1.txt', names=['population', 'profit'])
# data.head()
# data.info()
sns.lmplot('population', 'profit', data, fit_reg=False)
#plt.show()
"""
    # 多变量的假设 h 表示为：
    # ℎ𝜃(𝑥)=𝜃0+𝜃1𝑥1+𝜃2𝑥2+...+𝜃𝑛𝑥𝑛
    # 这个公式中有n+1个参数和n个变量，为了使得公式能够简化一些，引入 𝑥0=1 ，则公式转化为：
    # 此时模型中的参数是一个n+1维的向量，任何一个训练实例也都是n+1维的向量，特征矩阵X的维度是 m*(n+1),m为数据个数
    # 因此公式可以简化为： ℎ𝜃(𝑥)=𝜃𝑇𝑋 ，其中上标T代表矩阵转置。
"""


def get_X(df):
    # 读取特征population
    ones = pd.DataFrame({'ones': np.ones(len(df))})  # ones是m行1列的dataframe
    data = pd.concat([ones, df], axis=1)    # 根据列合并数据
    return data.iloc[:, :-1].values     # 返回ndarray,不是矩阵


def get_y(df):
    # 读取标签值
    return np.array(df.iloc[:, -1])     # df的最后一列


# print(get_y(data))
X = get_X(data)
y = get_y(data)
# print(y.shape, type(y))


def normalize_feature(df):
    # 特征缩放操作，对数据进行均值归一化
    return df.apply(lambda column: (column - column.mean()) / column.std())


"""
    # 代价函数的计算
    # 𝐽(𝜃)=1/2𝑚 * ∑𝑖=1𝑚(ℎ𝜃(𝑥(𝑖))−𝑦(𝑖))2
    # 其中：ℎ𝜃(𝑥)=𝜃𝑇𝑋=𝜃0𝑥0+𝜃1𝑥1+𝜃2𝑥2+...+𝜃𝑛𝑥𝑛
"""

# 参数向量𝜃
theta = np.zeros(X.shape[1])  # X.shape[1]=2,代表特征数n


def lr_cost(theta, X, y):
    """

    :param theta: 参数向量𝜃
    :param X: R(m*n), m 样本数, n 特征数
    :param y: R(m),profit
    :return: 代价函数值
    """
    m = X.shape[0]   # 样本数
    inner = X @ theta - y       # 矩阵乘以向量，等价于X.dot(theta),矩阵相乘
    """
        # 1*m @ m*1 = 1*1 in matrix multiplication
        # but you know numpy didn't do transpose in 1d array, so here is just a
        # vector inner product to itselves
    """
    square_sum = inner.T @ inner
    cost = square_sum / (2 * m)
    return cost


# print('cost:', lr_cost(theta, X, y))
# 计算代价函数的导数值

def gradient(theta, X, y):
    m = X.shape[0]
    inner = X.T @ (X @ theta - y)     # (m,n).T @ (m, 1) -> (n, 1)，X @ theta等价于X.dot(theta)
    return inner / m

"""
    # batch gradient decent（批量梯度下降）
    # 𝜃𝑗:=𝜃𝑗−𝛼∂∂𝜃𝑗𝐽(𝜃)
"""
def batch_gradient_decent(theta, X, y, epoch, alpha = 0.01):
    """
    :param theta:
    :param X:
    :param y:
    :param epoch: 批处理的轮数
    :param alpha:
    :return:
    """
    cost_data = [lr_cost(theta, X, y)]      # 比较随着迭代次数的增多，比较代价函数值的变化
    _theta = theta.copy()
    for i in range(epoch):
        _theta = _theta - alpha * gradient(_theta, X, y)
        cost_data.append(lr_cost(_theta, X, y))

    return _theta, cost_data

epoch = 600
final_theta, cost_data = batch_gradient_decent(theta, X, y, epoch)

# print('final_theta', final_theta)
# print('cost_data', cost_data)
# print('最终的代价函数值：', lr_cost(final_theta, X, y))

# 将代价数据可视化,比较随着迭代次数的增多，比较代价函数值的变化
"""
ax = sns.tsplot(cost_data, time=np.arange(epoch+1))     # 新版本没有tsplot（）
ax.set_xlabel('epoch')
ax.set_ylabel('cost')
plt.show()
"""

"""
b = final_theta[0]
m = final_theta[1]
plt.scatter(data.population, data.profit, label="Training data")
plt.plot(data.population, data.population*m + b, label="Prediction")    # 画出假设函数
plt.legend(loc=2)   # 左上角
plt.show()
"""
