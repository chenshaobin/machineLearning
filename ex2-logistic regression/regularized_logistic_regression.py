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
"""
sns.set(context="notebook", style="ticks", font_scale=1.5)

sns.lmplot('test1', 'test2', hue='accepted', data=df, height=6, fit_reg=False, scatter_kws={"s": 50})
plt.title('Regularized Logistic Regression')
plt.show()
"""
# 画出数据分布end
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
    regularized_term = (l / (2 * len(X))) * np.power(theta_1_to_n, 2).sum()
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


def regularized_logistic_regression(power, l):
    """
        # 将数据读取、特征映射、利用梯度下降计算最优参数统合在一个函数当中
        #   power: int
        #   l: int
        #   return：final_theta
    """

    df = pd.read_csv('ex2data2.txt', names=['test1', 'test2', 'accepted'])
    x1 = np.array(df.test1)
    x2 = np.array(df.test2)
    y = get_y(df)
    X = feature_mapping(x1, x2, power=6, as_ndarray_flag=True)  # 获取数据，数组形式
    theta = np.zeros(data.shape[1])  # n*1的ndarray数组,一维
    # 尝试其它method方法,感觉区别不是特别大; args参数中应该带l，
    res = opt.minimize(fun=regularized_cost, x0=theta, args=(X, y, l), method='TNC', jac=regularized_gradient)
    final_theta  = res.x
    return final_theta

def find_decision_boundary(density, power, theta, threshold):
    """
    # 找到所有满足  𝑋×𝜃=0  的x
    # 创建一个足够密集的x、y网格，利用参数theta，找到𝑋×𝜃足够小于0的特征，并利用其中的两组数据作为决策边界函数的x,y

    :param density: 决定x、y取值的密集度
    :param power: 决定多项式的幂
    :param theta: 参数
    :param threshold: 阈值设置
    :return: 用于画出决策边界的x、y
    """
    t1 = np.linspace(-1, 1.5, density)
    t2 = np.linspace(-1, 1.5, density)
    cordinates = [(x, y) for x in t1 for y in t2]
    x_cord, y_cord = zip(*cordinates)
    mapped_cord = feature_mapping(x_cord, y_cord, power)    # return a dataFrame
    inner_product = mapped_cord.values @ theta
    decision = mapped_cord[np.abs(inner_product) < threshold]       # 找到𝑋×𝜃足够小于0的映射特征数据,这里的数据提取需要再斟酌斟酌
    # print(decision)     # 测试
    return decision.f10, decision.f01   # 因为是二位平面，则选择幂为1的数据，即x1,x2

def draw_boundary(power, l):
    """
    :param power: polynomial power for mapped feature
    :param l: 常数，作为λ值
    :return: 图像
    """
    density = 1000
    threshold = 2 * 10 ** -3
    final_theta = regularized_logistic_regression(power, l)
    x, y = find_decision_boundary(density, power, final_theta, threshold)
    df = pd.read_csv('ex2data2.txt', names=['test1', 'test2', 'accepted'])
    sns.lmplot('test1', 'test2', hue='accepted', data=df, height=8, fit_reg=False, scatter_kws={"s": 40})
    plt.scatter(x, y, c='r', s=8)      # 画出散点图，红色
    plt.title('Decision boundary')
    plt.show()

draw_boundary(power=6, l=1)
# draw_boundary(power=6, l=0)     # 过拟合
# draw_boundary(power=6, l=100)      # l过大，欠拟合效果
