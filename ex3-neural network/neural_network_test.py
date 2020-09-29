#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as sio      # 读取mat文件
import matplotlib
import scipy.optimize as opt
from sklearn.metrics import classification_report   #这个包是评价报告
"""
    # ex3weights.mat文件是已经训练好的权重值，
    # 本例子利用该数据进行前向传播得到预测结果值
"""
def load_data(path, transpose=True):
    """
    :param path:    文件名
    :param transpose: 转置矩阵的标志
    :return: 返回数据X和标签值
    """
    data = sio.loadmat(path)
    # print(data)
    y = data.get('y')   # (5000,1)的二维数组
    # print(y)
    # print(y.shape)
    y = y.reshape(y.shape[0])   # 转换为（5000，）一维数组
    # print(y)
    # print(y.shape)
    X = data.get('X')       # (5000,400),二维数组
    # print('X:', X)
    # print(X.shape)
    if transpose:
        # for this dataset, you need a transpose to get the orientation right,坐标系的转换
        X = np.array([im.reshape((20, 20)).T for im in X])
        X = np.array([im.reshape(400) for im in X])
    return X, y

def load_weight(path):
    data = sio.loadmat(path)
    """
        theta_1 = data['Theta1']
        print(theta_1.shape)    # (25,401)
        theta_2 = data['Theta2']
        print(theta_2.shape)    # (10,26)
    """
    return data['Theta1'], data['Theta2']
    # print('data:', data)

def sigmoid(z):
    # 构建一个常用的逻辑函数（logistic function）为S形函数（Sigmoid function）
    return 1 / (1 + np.exp(-z))

theta1, theta2 = load_weight("ex3weights.mat")
X, y = load_data('ex3data1.mat', transpose=False)       # 使用原始数据，不进行转置
X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)     # X:(5000, 401), y:(5000,)

# 前向传播算法预测结果，注意：在进行矩阵计算的过程当中，要注意各数据的维度
a1 = X
z2 = a1 @ theta1.T      # 隐藏层的单元数与输入层一致，(5000, 401) @ (25,401).T = (5000, 25)，得到的数据当中没有偏置单元，需要自行添加
z2 = np.insert(z2, 0, values=np.ones(z2.shape[0]), axis=1)     #(5000,26)
a2 = sigmoid(z2)    # 得到隐藏层的激活项
z3 = a2 @ theta2.T   #(5000,10)
a3 = sigmoid(z3)    # 得到第三层的激活项，即最终的输出结果
# print(a3.shape)
y_pred = np.argmax(a3, axis=1) + 1      # 得到每一行中的最大值的索引值，numpy的索引是从0开始的，所以需要加1，才是最终的结果值
# print(y_pred.shape)

# 准确率的计算，虽然人工神经网络是非常强大的模型，但训练数据的准确性并不能完美预测实际数据，很容易出现过拟合现象。
analyse_result = classification_report(y, y_pred)
print(analyse_result)






