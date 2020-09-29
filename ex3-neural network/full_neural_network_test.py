#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as sio      # 读取mat文件
import matplotlib.pyplot as plt
import scipy.optimize as opt
import pandas as pd

data = sio.loadmat("ex3data1.mat")
"""
    # 图像数据data.get('X')X中表示为400维向量（其中有5,000个）。 400维“特征”是原始20 x 20图像中每个像素的灰度强度。
    # 图像数据data.get('y')表示图像中数字的数字类,共5000个。
"""
# 接下来对数据进行向量化，向量化代码可以利用线性代数，通常比迭代代码快一些。

def sigmoid(z):
    # 构建一个常用的逻辑函数（logistic function）为S形函数（Sigmoid function）
    return 1 / (1 + np.exp(-z))

def neural_cost(theta, X, y, learningRate):
    # 转换为矩阵乘法，也可以使用操作符@计算ndarray间的矩阵乘法
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first_item = np.multiply(-y, np.log(sigmoid(X * theta.T)))   # np.multiply计算逐元素相乘;(5000,400) * (1,400).T
    second_item = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / (2 * len(X))) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    return np.sum(first_item - second_item) / len(X) + reg      # 返回代价函数计算值

def gradient(theta, X, y, learningRate):
    # 向量化的梯度函数，使用正则化逻辑算法中的梯度下降法
    theta = np.matrix(theta)    # (1,401)
    X = np.matrix(X)    # (5000,401)
    Y = np.matrix(y)    # (1,5000)
    error = sigmoid(X * theta.T) - y    # （5000，1）
    # print('error.shape', error.shape)   # 验证
    grad = ((X.T * error) / len(X)).T + (learningRate / len(X)) * theta
    # 截距梯度没有进行正则化
    grad[0, 0] = np.sum(np.multiply(error, X[:, 0])) / len(X)   # X[:, 0]表示只取特征值x0，一共5000个
    return np.array(grad).ravel()       # 转换为一维数据格式

def one_vs_all(X, y, num_labels, learning_rate):
    """
    :param X:(5000,400)
    :param y:(5000,1)
    :param num_labels:  建立k维分类器，在这里k=10,由num_labels决定
    :param learning_rate:
    :return: all_theta
    """

    rows = X.shape[0]
    params = X.shape[1]
    all_theta = np.zeros((num_labels, params + 1))      # (num_labels, params + 1)
    X = np.insert(X, 0, values=np.ones(rows), axis=1)
    # 接下来将每个y值转换为10维的数组，只有一维的值为1，其它维的值为0
    # 数字是从1开始的
    for i in range(1, num_labels + 1):
        theta = np.zeros(params + 1)    # (401,)
        y_i = np.array([1 if label == i else 0 for label in y])     # 一维(5000,)，对每个y值进行矢量化
        # print('y_i_prep.shape:', y_i.shape)
        y_i = np.reshape(y_i, (rows, 1))    # 转换为二维数组(5000,1)
        # print('y_i_reshape.shape', y_i.shape)
        fmin = opt.minimize(fun=neural_cost, x0=theta, args=(X, y_i, learning_rate), method='TNC', jac=gradient)
        # print('fmin.x.shape:', fmin.x.shape)    # (401,)
        all_theta[i-1, :] = fmin.x

    return all_theta

# 各数据维度测试
"""
print('X.shape:', data.get('X').shape)
print('Y.shape:', data.get('y').shape)
y_test = np.array([1 if label == 1 else 0 for label in data.get('y')])
print('y_test.shape:', y_test.shape)
y_test_reshape = np.reshape(y_test, (5000, 1))
print('y_test_reshape:', y_test_reshape.shape)
"""
# 各数据维度测试end

# print("the classes of y:", np.unique(data.get('y')))    # 查看所有类别
all_theta = one_vs_all(data.get('X'), data.get('y'), 10, 1)
# print('all_theta:', all_theta.shape)

def predict_all(X, all_theta):
    # 预测图像标签
    rows = X.shape[0]
    params = X.shape[1]
    # num_labels = all_theta.shape[0]   # 标签类别总数目
    X = np.insert(X, 0, values=np.ones(rows), axis=1)
    X = np.matrix(X)
    all_theta = np.matrix(all_theta)
    h = sigmoid(X * all_theta.T)
    h_armax = np.argmax(h, axis=1)      # 取得每行中的最大值的索引
    h_armax = h_armax + 1   # 索引是0-9，而数字是1-10
    return h_armax

y_pred = predict_all(data['X'], all_theta)
result_compare = [1 if a == b else 0 for (a, b) in zip(y_pred, data['y'])]
accuracy = (sum(map(int, result_compare)) / float(len(result_compare)))
print("accuracy = {}%".format(accuracy * 100))

