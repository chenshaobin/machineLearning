#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    # 识别手写数字（from 0 to 9）
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio      # 读取mat文件
import matplotlib
import scipy.optimize as opt
from sklearn.metrics import classification_report   #这个包是评价报告

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
        # for this dataset, you need a transpose to get the orientation right,坐标系的转换,显示图片的时候需要转置
        X = np.array([im.reshape((20, 20)).T for im in X])
        X = np.array([im.reshape(400) for im in X])
    return X, y

x, y = load_data('ex3data1.mat')

def plot_an_image(image):
    # 画出一张数据图像
    fig, ax =plt.subplots(figsize=(2, 2))
    ax.matshow(image.reshape((20, 20)), cmap=matplotlib.cm.binary)  # 以矩阵的形式显示数组，二进制图像数据
    plt.xticks(np.array([]))    # 去除x,y的坐标的刻度显示
    plt.yticks(np.array([]))

# 随机选择一张照片显示
"""
pick_one = np.random.randint(0, 5000)
plot_an_image(x[pick_one, :])   # 选择显示的某行数据
plt.show()
print("this picture should be {}".format(y[pick_one]))
"""

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

# plot_100_image(x)
raw_x, raw_y = load_data('ex3data1.mat')
# print(raw_y)
# 数据处理
# add intercept=1 for x0
X = np.insert(raw_x, 0, values=np.ones(raw_x.shape[0]), axis=1)     # 在原来的数据上插入了第一列（全部为1）
# print(X.shape)
"""
    # y have 10 categories here. 1..10, they represent digit 0 as category 10 because matlab index start at 1
    # I'll ditit 0, index 0 again
"""
y_matrix = []
for k in range(1, 11):
    y_matrix.append((raw_y == k).astype(int))   # 将值转换为0,1表示，1代表为某个分类,将y进行向量化扩展，一共有10种分类
# 最后一行k=10，把它看成0，即把最后一行放到第一行
y_matrix = [y_matrix[-1]] + y_matrix[:-1]
# print(y_matrix)
y = np.array(y_matrix)
# print(y)
# print(y.shape)  # 扩展到（10,5000）

# 训练一维模型

def sigmoid(z):
    # 构建一个常用的逻辑函数（logistic function）为S形函数（Sigmoid function）
    return 1 / (1 + np.exp(-z))

def cost(theta, X, y):
    # 计算代价函数
    return np.mean(-y * np.log(sigmoid(X @ theta)) - (1 - y) * np.log(1 - sigmoid(X @ theta)))   # X @ theta与X.dot(theta)等价

def regularized_cost(theta, X, y, l=1):
    # 计算正则化代价函数
    # do not penalize theta_0
    theta_1_to_n = theta[1:]
    regularized_term = (l / (2 * len(X))) * np.power(theta_1_to_n, 2).sum()
    return cost(theta, X, y) + regularized_term


def gradient(theta, X, y):
    # 梯度下降，进行向量化计算,计算导数值
    return (1 / len(X)) * X.T @ (sigmoid(X @ theta) - y)

def regularized_gradient(theta, X, y, l =1):
    # 计算正则化梯度
    # do not penalize theta_0
    theta_1_to_n = theta[1:]
    regularized_theta = (l / len(X)) * theta_1_to_n
    regularized_term = np.concatenate([np.array([0]), regularized_theta])
    return gradient(theta, X, y) + regularized_term

def logistic_regression(X, y, l=1):
    """
    :param X: feature matrix, (m, n+1) # with incercept x0=1
    :param y: target vector, (m, )
    :param l: lambda constant for regularization
    :return: final_theta
    """
    theta = np.zeros(X.shape[1])
    res = opt.minimize(fun=regularized_cost, x0=theta, args=(X, y, l), method='TNC', jac=regularized_gradient)  # options中的disp设置为True则直接打印结果: options={'disp': True}
    final_theta = res.x
    return final_theta

def predict(x, theta):
    # 训练值预测和分析
    prob = sigmoid(x @ theta)
    return (prob >= 0.5).astype(int)    # >=0.5则表示预测正确

"""
final_theta = logistic_regression(X, y[0])  # 仅仅训练判别一种数字，这里只是判别数字0
# print('final_theta.shape:', final_theta.shape)  # (401,)
y_prep = predict(X, final_theta)
print('Accurancy={}'.format(np.mean(y[0] == y_prep)))
"""

# 训练一维模型end

# 训练k维模型：同时训练10种数字，取出现概率最大的那个作为结果
"""
k_theta = np.array([logistic_regression(X, y[k]) for k in range(10)])   # (10, 401)
# print(k_theta.shape)
# 进行预测
prob_matrix = sigmoid(X @ k_theta.T)    # (5000,401) * (10,401).T = （5000,10）
np.set_printoptions(suppress=True)  # 设置打印浮点数
# print(prob_matrix)
y_prep_k = np.argmax(prob_matrix, axis=1)   # 找到每一行中最大值的索引，即为训练得到的数字,0代表数字10
# print(y_prep_k)
y_answer = raw_y.copy()     # 复制初始结果
y_answer[y_answer == 10] = 0    # 将数字10转换为0
analyze_result = classification_report(y_answer, y_prep_k)
print(analyze_result)

# 训练k维模型end
"""