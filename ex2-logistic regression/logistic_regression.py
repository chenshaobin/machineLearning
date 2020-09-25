#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')        #This shows an example of the "fivethirtyeight" styling, which tries to replicate the styles from FiveThirtyEight.com.
import scipy.optimize as opt     # 利用其中的opt.minimize方法寻找最小参数
from sklearn.metrics import classification_report  # 这个包是评价报告

# 读取数据
data = pd.read_csv('ex2data1.txt', names=['exam1', 'exam2', 'admitted'])
# print(data.head())
# print(data.shape)
# print(data.describe())
sns.set(context="notebook", style="darkgrid", palette=sns.color_palette("RdBu", 2))
# sns.lmplot('exam1', 'exam2', hue='admitted',  data=data, height=6, fit_reg=False, scatter_kws={"s": 40})
# plt.show()

def get_X(df):
    # 读取特征population
    ones = pd.DataFrame({'ones': np.ones(len(df))})  # ones是m行1列的dataframe
    # print(ones)
    data = pd.concat([ones, df], axis=1)    # 根据列合并数据
    return data.iloc[:, :-1].values     # 返回ndarray,不是矩阵

def get_y(df):
    # 读取标签值
    return np.array(df.iloc[:, -1])     # df的最后一列

# print(get_y(data))
X = get_X(data)
# print(X.shape)
# print(len(X))
y = get_y(data)
# print(y.shape, type(y))

# 构建一个常用的逻辑函数（logistic function）为S形函数（Sigmoid function）
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 画出S形函数
fig, ax = plt.subplots()
ax.plot(np.arange(-10, 10, step=0.01), sigmoid(np.arange(-10, 10, step=0.01)))
ax.set_ylim(-0.1, 1.1)
ax.set_xlabel('z', fontsize=18)
ax.set_ylabel('g(z)', fontsize=18)
ax.set_title('sigmoid function', fontsize=18)
# plt.show()

theta = np.zeros(X.shape[1])    # n*1的ndarray数组,一维
# print(theta)
# print(theta.shape)
# 计算代价函数
def cost(theta, X, y):
    return np.mean(-y * np.log(sigmoid(X @ theta)) - (1 - y) * np.log(1- sigmoid(X @ theta)))   # X @ theta与X.dot(theta)等价

test_cost = cost(theta, X, y)   # 测试初始代价函数值
# print(test_cost)
# 梯度下降，进行向量化计算,计算导数值
def gradient(theta, X, y):
    return (1 / len(X)) * X.T @ (sigmoid(X @ theta) - y)

test_gradient = gradient(theta, X, y)   # 测试初始梯度值（导数）
# print(test_gradient)
# 梯度下降法寻找最优theta参数值
res = opt.minimize(fun=cost, x0=theta, args=(X, y), method='Newton-CG', jac=gradient)
# print(res)      # 查看结果信息

# 训练值预测和分析
def predict(x, theta):
    prob = sigmoid(x @ theta)
    return (prob >= 0.5).astype(int)    # >=0.5则表示预测正确

final_theta = res.x     # 最终的训练结果参数
y_pred = predict(X, final_theta)
analysis_result_report = classification_report(y, y_pred, target_names=['exam1', 'exam2'])
# print(analysis_result_report)

# 计算决策边界，即线性方程,推导一下方程h(x)即可确定
coef = -(res.x / res.x[2])
print(coef)
# print(data.describe())     # 查看数据分布情况
equation_x = np.arange(130, step=0.1)
equation_y = coef[0] + coef[1] * equation_x

sns.set(context="notebook", style="ticks", font_scale=1.5)
sns.lmplot('exam1', 'exam2', hue='admitted', data=data, height=6, fit_reg=False, scatter_kws={"s": 25})

plt.plot(equation_x, equation_y, 'gray')
plt.xlim(0, 130)
plt.ylim(0, 130)
plt.title("Decision Boundary")
# plt.show()
