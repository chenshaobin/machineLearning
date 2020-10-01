#!/usr/bin/python
# -*- coding: utf-8 -*-


"""
    # 利用水库水位的变化来预测大坝的出水量。
    # 使用正则化线性回归算法，调整参数，比较不同的训练偏差和方差所产生的影响
    # 根据偏差和方差的学习曲线来调整参数
"""

import numpy as np
import scipy.io as sio
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    # 数据集已经分成训练集、交叉验证和测试集三个部分
    data = sio.loadmat('ex5data1.mat')
    # print('data:', data)
    # print('data[\'X\'].shape:', data['X'].shape)      # (12,1)
    # 将数据展开成一维数组返回
    return map(np.ravel, [data['X'], data['y'], data['Xval'], data['yval'], data['Xtest'], data['ytest']])

# data = load_data()
# print('data:', data)
X, y, Xval, yval, Xtest, ytest = load_data()
# print('X.shape:', X.shape)  # (12,)
data_text_dataFrame = pd.DataFrame({'water_level': X, 'flow': y})
# 画出测试数据集
"""
sns.lmplot('water_level', 'flow', data=data_text_dataFrame, fit_reg=False, height=5)
plt.show()
"""
# 将数据集转换为数组并添加偏置单元
X, Xval, Xtest = [np.insert(x.reshape(x.shape[0], 1), 0, np.ones(x.shape[0]), axis=1) for x in (X, Xval, Xtest)]
# print('X.shape:', X.shape)    #(12,2)

# 代价函数
def cost(theta, X, y):
    """
    :param theta:  R(n), linear regression parameters
    :param X:   R(m*n), m records, n features
    :param y:   R(m)
    :return: cost
    """
    m = X.shape[0]
    inner = X @ theta - y    # (m,)
    square_sum = inner.T @ inner
    cost = square_sum / (2 * m)

def regularized_cost(theta, X, y, l=1):
    m = X.shape[0]
    regularized_term = (1 / (2 * m)) * np.power(theta[1:], 2).sum()
    return cost(theta, X, y) + regularized_term

# 梯度计算
def gradient(theta, X, y):
    m = X.shape[0]
    innner = X.T @ (X @ theta - y)

    return innner / m

def regularized_gradient(theta, X, y, l=1):
    m = X.shape[0]
    regularized_term = theta.copy()
    regularized_term[0] = 0     # 偏置单元不进行正则化
    regularized_term = regularized_term * (1 / m)
    return gradient(theta, X, y) + regularized_term

# 拟合函数
def linear_regression(X, y, l=1):
    theta = np.ones(X.shape[1])
    res = opt.minimize(fun=regularized_cost, x0=theta, args=(X, y, l), method='TNC', jac=regularized_gradient, options={"disp":True})
    return res
