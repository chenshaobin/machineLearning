#!/usr/bin/python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context="notebook", style="white")

import numpy as np
import pandas as pd
import scipy.io as sio

data_origin = sio.loadmat('data/ex7data1.mat')
# print(data_origin.keys())
# 初始数据显示
X = data_origin.get('X')
# print('X.shape:', X.shape)  # (50,2)
sns.lmplot('X1', 'X2', data=pd.DataFrame(X, columns=['X1', 'X2']), fit_reg=False)
# plt.show()

def normalize(X):
    # 对每一类特征值进行归一化数据，for each column, X-mean / std
    X_copy = X.copy()
    m, n = X_copy.shape

    for col in range(n):
        X_copy[:, col] = (X_copy[:, col] - X_copy[:, col].mean()) / X_copy[:,  col].std()

    return X_copy

def covariance_matrix(X):
    """
        # 数据协方差矩阵的计算
        :param X: (m*n)
        :return: (n*n)
    """
    m = X.shape[0]
    return (X.T @ X) /m

def pca(X):
    """

    :param X: ndarray(m, n)
    :return: U ndarray(n, n): principle components
    """
    X_norm = normalize(X)
    Sigma = covariance_matrix(X_norm)
    # print('Sigma.shape:', Sigma.shape)
    # 奇异值分解
    U, S, V = np.linalg.svd(Sigma)
    # print('U.shape:', U.shape)
    # print('S.shape:', S.shape)
    # print('V.shape:', V.shape)
    return U, S, V

def project_data(X, U, k):
    """
    :param X: 归一化后的数据，（m,n）
    :param U: 奇异值分解后的U，（n,n）
    :param k: 需要降到的低维的维数
    :return: 返回降维后的数据
    """
    m, n = X.shape
    if k > n:
        raise ValueError('k should be lower dimension of n')

    return X @ U[:, :k]

def recover_data(Z, U):
    # 低维数据恢复到高维数据
    m, n = Z.shape
    if n >= U.shape[0]:
        raise ValueError('Z dimension is >= U, you should recover from lower dimension to higher')

    return Z @ U[:, :n].T

# 归一化后的数据显示
X_norm = normalize(X)
sns.lmplot('X1', 'X2', data=pd.DataFrame(X_norm, columns=['X1', 'X2']), fit_reg=False)
# plt.show()

Sigma = covariance_matrix(X_norm)   #(n, n)
# print(Sigma)
U, S, V = pca(X)
# print(U)    # (n, n)

# 数据降维，降高维数据投射到低维数据
Z = project_data(X_norm, U, 1)
# print('Z.shape:', Z.shape)  #(50,1)
# print('Z[:10]', Z[:10])
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 4))
sns.regplot('X1', 'X2', data=pd.DataFrame(X_norm, columns=['X1', 'X2']), fit_reg=False, ax=ax1)  # 用线性回归模型拟合数据
ax1.set_title('Original dimension')

sns.rugplot(Z, ax=ax2)      # 用于绘制出一维数组中数据点实际的分布位置情况
ax2.set_xlabel('Z')
ax2.set_title('Z dimension')
# plt.show()

X_recover = recover_data(Z, U)
print('X_recover.shape:', X_recover.shape)
fig2, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(13, 4))

sns.rugplot(Z, ax=ax1)
ax1.set_title('Z dimension')
ax1.set_xlabel('Z')

sns.regplot('X1', 'X2', data=pd.DataFrame(X_recover, columns=['X1', 'X2']), fit_reg=False, ax=ax2)
ax2.set_title("2D projection from Z")

sns.regplot('X1', 'X2', data=pd.DataFrame(X_norm, columns=['X1', 'X2']), fit_reg=False, ax=ax3)
ax3.set_title('Original dimension')
plt.show()
