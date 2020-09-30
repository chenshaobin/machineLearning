#!/usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import matplotlib
import scipy.optimize as opt
from sklearn.metrics import classification_report#这个包是评价报告

def load_data(path, transpose=True):
    """
    :param path:    文件名
    :param transpose: 转置矩阵的标志
    :return: 返回数据X和标签值
    """
    data = sio.loadmat(path)
    # print(data)
    y = data.get('y')   # (5000,1)的二维数组
    # print('y_origin:', y)
    # print('y_origin.shape:', y.shape)
    y = y.reshape(y.shape[0])   # 转换为（5000，）一维数组
    # print('y_change:', y)
    # print('y.shape_change', y.shape)
    X = data.get('X')       # (5000,400),二维数组
    # print('X:', X)
    # print(X.shape)
    if transpose:
        # for this dataset, you need a transpose to get the orientation right,坐标系的转换,显示图片的时候需要转置
        X = np.array([im.reshape((20, 20)).T for im in X])      # (5000, 20, 20)
        # print('X.shape:', X.shape)
        X = np.array([im.reshape(400) for im in X])
    return X, y

def load_weight(path):
    # 读取权重
    data = sio.loadmat(path)
    """
        theta_1 = data['Theta1']
        print(theta_1.shape)    # (25,401)
        theta_2 = data['Theta2']
        print(theta_2.shape)    # (10,26)
    """
    return data['Theta1'], data['Theta2']
    # print('data:', data)

X_show, y_show = load_data('ex4data1.mat')    # 用于图像显示的数据

def plot_100_image(X):
    # 随机显示100张图片
    size = int(np.sqrt(X.shape[1]))     # 每张图片是20 * 20的，但是在数据中是平展开来的，为一行400列
    sample_index = np.random.choice(np.arange(X.shape[0]), 100)    # 随机选取100个数据作为X的行数
    sample_images = X[sample_index, :]      # 获取这100张图的数据
    fig, ax = plt.subplots(nrows=10, ncols=10, sharex=True, sharey=True, figsize=(8, 8))
    for r in range(10):
        for c in range(10):
            ax[r, c].matshow(sample_images[10 * r + c].reshape((size, size)), cmap=matplotlib.cm.binary)    # 二进制图像数据
            plt.xticks(np.array([]))  # 去除x,y的坐标的刻度显示
            plt.yticks(np.array([]))
    plt.show()

# plot_100_image(X_show)

# 导入原始数据
X_raw, y_raw = load_data('ex4data1.mat', transpose=False)
X = np.insert(X_raw, 0, values=np.ones(X_raw.shape[0]), axis=1)     # 每一行增加一个偏置单元
# print('X.shape:', X.shape)      # (5000, 401)
# print('y_raw.shape:', y_raw.shape)  # (5000,)

def expand_y(y):
    # 对y进行扩展 （5000，）-> (5000, 10)
    res = []
    for i in y:
        y_array = np.zeros(10)
        y_array[i-1] = 1
        res.append(y_array)

    res = np.array(res)     # (5000, 10)
    # print('res.shape:', res.shape)
    return res

y = expand_y(y_raw)
# print('y:', y)
theta1, theta2 = load_weight("ex4weights.mat")      # theta1:(25,401), theta2:(10,26)

def serialize(a, b):
    # 序列化矩阵
    return np.concatenate((np.ravel(a), np.ravel(b)))   # 拼接成一维数组

def deserialize(seq):
    # 反序列化矩阵
    return seq[:25*401].reshape(25, 401), seq[25*401:].reshape(10, 26)

theta = serialize(theta1, theta2)
# print('theta.shape:', theta.shape)  # 25* 401+10*26=10285

def sigmoid(z):
    # 构建一个常用的逻辑函数（logistic function）为S形函数（Sigmoid function）
    return 1 / (1 + np.exp(-z))

def feed_forward(theta, X):
    # 前向传播
    theta1, theta2 = deserialize(theta)
    rows = X.shape[0]
    a1 = X
    z2 = a1 @ theta1.T  # 隐藏层的单元数与输入层一致，(5000, 401) @ (25,401).T = (5000, 25)
    a2 = sigmoid(z2)    # 得到隐藏层的激活项
    a2 = np.insert(a2, 0, values=np.ones(rows), axis=1)     # (5000,26)
    z3 = a2 @ theta2.T  # (5000,10)
    h = sigmoid(z3)  # (5000,10)
    return a1, z2, a2, z3, h

a1, z2, a2, z3, h = feed_forward(theta, X)
# print('h.shape:', h.shape)
# print('z2.shape:', z2.shape)

def cost(theta, X, y):
    # 简化版代价函数
    rows = X.shape[0]
    a1, z2, a2, z3, h = feed_forward(theta, X)
    accumulate = -np.multiply(y, np.log(h)) - np.multiply((1 - y), np.log(1 - h))
    return accumulate.sum() / rows

# print('cost:', cost(theta, X, y))

def regularized_cost(theta, X, y, l=1):
    # 在正则化代价函数中，偏置单元是不参与计算的
    theta1, theta2 = deserialize(theta)
    rows = X.shape[0]
    reg_theta1 = (l / (2 * rows)) * np.power(theta1[:, 1:], 2).sum()
    reg_theta2 = (l / (2 * rows)) * np.power(theta2[:, 1:], 2).sum()
    return cost(theta, X, y) + reg_theta1 + reg_theta2

# print('regularized_cost:', regularized_cost(theta, X, y))

# 以下是反向传播过程
def sigmoid_gradient(z):
    # 导数
    return np.multiply(sigmoid(z), 1 - sigmoid(z))


def gradient(theta, X, y):
    # theta gradient
    # return: 返回代价函数关于theta的导数值
    theta1, theta2 = deserialize(theta)  # (25, 401),(10,26)
    rows = X.shape[0]
    delta1 = np.zeros(theta1.shape)
    delta2 = np.zeros(theta2.shape)
    a1, z2, a2, z3, h = feed_forward(theta, X)

    for i in range(rows):
        # 累计数据的参数误差，最后计算平均值
        a1i = a1[i, :]      # (1,401)，一维数组
        z2i = z2[i, :]      # (1,25)，一维数组
        a2i = a2[i, :]      # (1,26)，一维数组
        hi = h[i, :]        # (1,10)，一维数组
        yi = y[i, :]        # (1,10),一维数组
        # print('yi.shape:', yi.shape)
        d3i = hi - yi       # (1,10)
        z2i = np.insert(z2i, 0, values=np.ones(1))  # 添加偏置单元，（1，26）
        d2i = np.multiply(theta2.T @ d3i, sigmoid_gradient(z2i))    # (10,26).T @ (10,) -> (26,)
        # print("测试d2i.shape:", d2i.shape)    # (26,)
        delta2 += np.matrix(d3i).T @ np.matrix(a2i)    # (1,10).T @ (1,26) -> (10,26)
        delta1 += np.matrix(d2i[1:]).T @ np.matrix(a1i) #(1,25).T @ (1,401) ->(25,401),这里需要移除偏置单元

    delta2 = delta2 / rows
    delta1 = delta1 / rows

    return serialize(delta1, delta2)
"""
# 测试维度
delta1, delta2 = deserialize(gradient(theta, X, y))
print('delta1.shape:', delta1.shape)
print('delta2.shape:', delta2.shape)
"""

def expand_array(arr):
    # 将数组扩展为方阵,元素不变
    """
    # input:
    [1, 2, 3]
    # output:
    [[1, 2, 3],
     [1, 2, 3],
     [1, 2, 3]]
    """
    return np.array(np.matrix(np.ones(arr.shape[0])).T @ np.matrix(arr))

def regularized_gradient(theta, X, y, l):
    # add regularization to the gradient,即代价函数的导数值
    rows = X.shape[0]
    delta1, delta2 =deserialize(gradient(theta, X, y))
    theta1, theta2 = deserialize(theta)
    # 偏置单元不进行正则化
    theta1[:, 0] = 0
    reg_trem_delta1 = (l / rows) * theta1
    delta1 = delta1 + reg_trem_delta1

    theta2[:, 0] = 0
    reg_trem_delta2 = (l / rows) * theta2
    delta2 = delta2 + reg_trem_delta2

    return serialize(delta1, delta2)

def gradient_checking(theta, X, y, epsilon, regularized=False):
    # 进行梯度检测
    def a_numeric_grad(plus, minus, regularized= False):
        # 梯度检测中代价函数导数的一种计算方法, 与反向传播算法得到的进行比较
        if regularized:
            # 代价函数考虑正则化
            return (regularized_cost(plus, X, y) - cost(minus, X, y)) / (epsilon * 2)
        else:
            return (cost(plus, X, y) - cost(minus, X, y)) / (epsilon * 2)

    theta_matrix = expand_array(theta)  # 扩展成（10285，10285）
    epsilon_matrix = np.identity(len(theta)) * epsilon   # np.identity()形成单位方阵
    plus_matrix = theta_matrix + epsilon_matrix
    minus_matrix = theta_matrix - epsilon_matrix
    # 计算代价函数的导数
    numeric_grad = np.array([a_numeric_grad(plus_matrix[i], minus_matrix[i], regularized) for i in range(len(theta))])
    # 利用反向传播算法分析得到的梯度
    analytic_grad = regularized_gradient(theta, X, y) if regularized else gradient(theta, X, y)
    """
        # If you have a correct implementation, and assuming you used EPSILON = 0.0001
        # the diff below should be less than 1e-9
        # this is how original matlab code do gradient checking
        # np.linalg.norm 求二范数
    """
    diff = np.linalg.norm(numeric_grad - analytic_grad) / np.linalg.norm(numeric_grad + analytic_grad)
    print('If your backpropagation implementation is correct,'
          '\nthe relative difference will be smaller than 10e-9 (assume epsilon=0.0001).\nRelative Difference: {}\n'.format(diff))

#gradient_checking(theta, X, y, epsilon=0.0001)

# 模型训练
def random_init(size):
    # 随机产生size个参数值
    return np.random.uniform(-0.12, 0.12, size)

def nn_training(X, y):
    # 训练参数
    init_theta = random_init(10285)     # (25,401) + (10,26)
    res = opt.minimize(fun=regularized_cost, x0=init_theta, args=(X, y, 1),
                       method='TNC', jac=regularized_gradient, options={'maxiter': 300})
    return res

res = nn_training(X, y)
# print('res:', res)
_, y_answer = load_data('ex4data1.mat')
final_theta = res.x

def show_accuracy(theta, X, y):
    _, _, _, _, h = feed_forward(theta, X)
    y_prep = np.argmax(h, axis=1) + 1
    print(classification_report(y, y_prep))

# show_accuracy(final_theta, X, y_answer)

def plot_hidden_layer(theta):
    final_theta, _ = deserialize(theta)
    hidden_layer = final_theta[:, 1:]   # 不考虑偏置单元
    fig, ax_array = plt.subplots(nrows=5, ncols=5, sharey=True, sharex=True, figsize=(5, 5))

    for r in range(5):
        for c in range(5):
            ax_array[r, c].matshow(hidden_layer[5 * r + c].reshape((20, 20)),
                                   cmap=matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
    plt.show()

# plot_hidden_layer(final_theta)
