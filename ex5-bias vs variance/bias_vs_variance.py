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
    return cost

def regularized_cost(theta, X, y, l=1):
    m = X.shape[0]
    regularized_term = (l / (2 * m)) * np.power(theta[1:], 2).sum()
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
    regularized_term = regularized_term * (l / m)
    return gradient(theta, X, y) + regularized_term

# 拟合函数
def linear_regression(X, y, l=1):
    theta = np.ones(X.shape[1])
    res = opt.minimize(fun=regularized_cost, x0=theta, args=(X, y, l), method='TNC', jac=regularized_gradient, options={"disp":False})
    return res

final_theta = linear_regression(X, y, l=0).get('x')

# 画出原始数据的散点图和决策边界
"""
b = final_theta[0]      # 获得直线的截距，即偏置单元
m = final_theta[1]      # 获得直线的斜率

plt.scatter(X[:, 1], y, label="Training data")      # 画出原始数据的散点图
plt.plot(X[:, 1], X[:, 1] * m + b, label="Preduction" )     # 画出决策边界
plt.legend(loc=2)
plt.show()
"""

def show_learning_curve_nonRegularized(X, y, Xval, yval, l=0):
    """
        # 误差函数没有正则化时的学习函数分析
        # 比较训练集和交叉验证集
    """
    training_cost, cv_cost = [], []
    m = X.shape[0]
    for i in range(1, m + 1):
        res = linear_regression(X[:i, :], y[:i], l)
        tc = regularized_cost(res.x, X[:i, :], y[:i], l)
        cv = regularized_cost(res.x, Xval, yval, l)
        training_cost.append(tc)
        cv_cost.append(cv)

    plt.plot(np.arange(1, m+1), training_cost, label='training cost')
    plt.plot(np.arange(1, m+1), cv_cost, label='cv cost')
    plt.legend(loc=1)
    plt.show()

# show_learning_curve_nonRegularized(X, y, Xval, yval)      # 拟合不好，模型较差

# 通过多项式回归拟合出更好的模型

def normalize_feature(df):
    # 归一化数据
    # 输入数据为dataFrame格式
    # 对每一行数据进行归一化，每一列数据
    return df.apply(lambda column: (column - column.mean()) / column.std())

def poly_features(x, power, as_ndarray=False):
    """
        # 通过多项式扩展特征值
        # 当一个训练集X的大小为m×1被传递到函数中，函数应该返回一个m×p矩阵
        # 如果as_ndarray=True,则返回数组形式的数据
    """
    data = {'f{}'.format(i): np.power(x, i) for i in range(1, power + 1)}
    df = pd.DataFrame(data)
    # print('poly_features:')
    # print(df)

    return df.values if as_ndarray else df
def prepare_poly_data(*args, power):
    # 产生多组多项式的数据，比如X, Xval, or Xtest等
    def prepare(x):
        df = poly_features(x, power=power)
        # print('X_poly_features:')
        # print(df)
        df_to_ndarray =normalize_feature(df).values
        # print('X_normalize:')
        # print(df_to_ndarray)
        return np.insert(df_to_ndarray, 0, np.ones(df_to_ndarray.shape[0]), axis=1)     # 每一行添加偏置单元

    return [prepare(x) for x in args]


X, y, X_val, y_val, X_test, y_test = load_data()

# print(poly_features(X, power=3, as_ndarray=True))
# prepare_poly_data(X, power=3)
X_ploy, X_val_poly, X_test_poly = prepare_poly_data(X, X_val, X_test, power=8)      # 特征值加上偏置单元已经扩展到八个

# 画出决策边界，多项式

def plot_learning_curve(X, y, Xval, yval, l=0):
    training_cost, cv_cost = [], []
    m = X.shape[0]

    for i in range(1, m + 1):
        # regularization applies here for fitting parameters
        res = linear_regression(X[:i, :], y[:i], l=l)

        # remember, when you compute the cost here, you are computing
        # non-regularized cost. Regularization is used to fit parameters only
        tc = cost(res.x, X[:i, :], y[:i])
        cv = cost(res.x, Xval, yval)

        training_cost.append(tc)
        cv_cost.append(cv)

    plt.plot(np.arange(1, m + 1), training_cost, label='training cost')
    plt.plot(np.arange(1, m + 1), cv_cost, label='cv cost')
    plt.legend(loc=1)
    plt.xlabel('lambda')
    plt.ylabel('cost')
    plt.show()

# plot_learning_curve(X_ploy, y, X_val_poly, y_val, l=0)    # 过拟合

# plot_learning_curve(X_ploy, y, X_val_poly, y_val, l=1)     # 模型比较合适
# plot_learning_curve(X_ploy, y, X_val_poly, y_val, l=100)    # 正则化项太大，变成欠拟合状态

# 找到合适的正则化项
def find_suitable_lambda(X_ploy, X_val_poly, y, y_val):
    l_candidate = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]     # l的选值
    training_cost, cv_cost = [], []
    for l in l_candidate:
        res = linear_regression(X_ploy, y, l)
        tc = cost(res.x, X_ploy, y)
        cv = cost(res.x, X_val_poly, y_val)
        training_cost.append(tc)
        cv_cost.append(cv)

    l_cv_cost_min = l_candidate[np.argmin(cv_cost)]
    print('交叉训练误差最小对应的l值：', l_cv_cost_min)
    # 画出对应不同l值的学习曲线
    plt.plot(l_candidate, training_cost, label='training')
    plt.plot(l_candidate, cv_cost, label='cross validation')
    plt.legend(loc=1)
    plt.xlabel('lambda')
    plt.ylabel('cost')
    plt.show()

# find_suitable_lambda(X_ploy, X_val_poly, y, y_val)

def find_suit_lambda(X_ploy, y, X_test_poly, y_test):
    # 根据测试数据计算误差来确定lambda值
    l_candidate = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]  # l的选值
    test_cost = []
    for l in l_candidate:
        theta = linear_regression(X_ploy, y, l).x
        tc = cost(theta, X_test_poly, y_test)
        test_cost.append(tc)

    l_test_cost_min = l_candidate[np.argmin(test_cost)]
    print('测试集误差最小对应的l值：', l_test_cost_min)



find_suit_lambda(X_ploy, y, X_test_poly, y_test)
