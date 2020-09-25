#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')        #This shows an example of the "fivethirtyeight" styling, which tries to replicate the styles from FiveThirtyEight.com.
import scipy.optimize as opt     # 利用其中的opt.minimize方法寻找最小参数
from sklearn.metrics import classification_report  # 这个包是评价报告

df = pd.read_csv('ex2data2.txt', names=['test1', 'test2', 'accepted'])
# print(df)
# print(df.head())
# 画出数据分布

sns.set(context="notebook", style="ticks", font_scale=1.5)

sns.lmplot('test1', 'test2', hue='accepted', data=df, height=6, fit_reg=False, scatter_kws={"s": 50})
plt.title('Regularized Logistic Regression')
# plt.show()

def get_y(df):
    # 读取标签值
    return np.array(df.iloc[:, -1])     # df的最后一列

# 构建一个常用的逻辑函数（logistic function）为S形函数（Sigmoid function）
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 计算代价函数
def cost(theta, X, y):
    return np.mean(-y * np.log(sigmoid(X @ theta)) - (1 - y) * np.log(1 - sigmoid(X @ theta)))   # X @ theta与X.dot(theta)等价

# 梯度下降，进行向量化计算,计算导数值
def gradient(theta, X, y):
    return (1 / len(X)) * X.T @ (sigmoid(X @ theta) - y)

# 训练值预测和分析
def predict(x, theta):
    prob = sigmoid(x @ theta)
    return (prob >= 0.5).astype(int)    # >=0.5则表示预测正确

"""
    # 如果样本量多，逻辑回归问题很复杂，而原始特征只有x1,x2可以用多项式创建更多的特征x1、x2、x1x2、x1^2、x2^2、... X1^nX2^n。
    # 因为更多的特征进行逻辑回归时，得到的分割线可以是任意高阶函数的形状。
"""
# 特征映射函数
def feature_mapping(x1, x2, power, as_ndarray_flag=False):
    """
        # x1, x2 为数组
        # return mapped features as ndarray or dataframe
        # data = {}
        # # inclusive
        # for i in np.arange(power + 1):
        #     for p in np.arange(i + 1):
        #         data["f{}{}".format(i - p, p)] = np.power(x, i - p) * np.power(y, p)
    """
    data = {"f{}{}".format(i-p, p): np.power(x1, i - p) * np.power(x2, p) for i in np.arange(power+1) for p in np.arange(i+1)}
    if as_ndarray_flag:
        return pd.DataFrame(data).values    # 数组形式
    else:
        return pd.DataFrame(data)   # 还是DataFrame格式

x1 = np.array(df.test1)
# print(x1)
x2 = np.array(df.test2)
data = feature_mapping(x1, x2, power=6)     # 经过特诊映射后的特征数据
# print(data.shape)
# print(data.head())
# print(data.describe())

theta = np.zeros(data.shape[1])    # n*1的ndarray数组,一维
X = feature_mapping(x1, x2, power=6, as_ndarray_flag=True)   # 获取数据，数组形式
# print(X.shape)
y = get_y(df)
# print(y.shape)
# 计算正则化代价函数
def regularized_cost(theta, X, y, l=1):
    # do not penalize theta_0
    theta_1_to_n = theta[1:]
    regularized_term = (1 / (2 * len(X))) * np.power(theta_1_to_n, 2).sum()
    return cost(theta, X, y) + regularized_term

# 测试正则化代价函数
# print('init cost = {}'.format(regularized_cost(theta, X, y)))

# 计算正则化梯度
def regularized_gradient(theta, X, y, l =1):
    # do not penalize theta_0
    theta_1_to_n = theta[1:]
    regularized_theta = (l / len(X)) * theta_1_to_n
    regularized_term = np.concatenate([np.array([0]), regularized_theta])
    return gradient(theta, X, y) + regularized_term

# 测试正则化梯度
# print(regularized_gradient(theta, X, y))

# 参数拟合
res = opt.minimize(fun=regularized_cost, x0=theta, args=(X, y), method='Newton-CG', jac=regularized_gradient)
# print(res)
final_theta = res.x
y_pred = predict(X, final_theta)
analysis_result_report = classification_report(y, y_pred, target_names=['test1', 'test2'])
# print(analysis_result_report)

# 使用不同的l画出决策边界
