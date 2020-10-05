#!/usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context="notebook", style="white")

import numpy as np
import pandas as pd
import scipy.io as sio

data_origin = sio.loadmat('data/ex7faces.mat')
# print(data_origin.keys())
# 在展示图像之前，需要对图像数据做坐标上的转换
X = np.array([x.reshape(32, 32).T.reshape(1024) for x in data_origin.get('X')])     #(5000, 1024)
# print('X.shape:', X.shape)

def plot_n_image(X, n):
    """
        #显示n张照片
        :param X: 已经做好坐标转换后的照片数据
        :param n: 一个可以开方的数
        :return: none
    """

    pic_size = int(np.sqrt(X.shape[1]))      # 图片大小，32X32
    grid_size = int(np.sqrt(n))  # 网格的行列数，
    first_n_images = X[:n, :]
    fig, ax_array = plt.subplots(nrows=grid_size, ncols=grid_size, sharey=True, sharex=True, figsize=(8, 8))
    for r in range(grid_size):
        for c in range(grid_size):
            ax_array[r, c].imshow(first_n_images[grid_size * r + c].reshape((pic_size, pic_size)))
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
    plt.show()

def normalize(X):
    # 对每一类特征值进行归一化数据，for each column, X-mean / std
    X_copy = X.copy()
    m, n = X_copy.shape
    # 对每一种特征值进行归一化操作
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

# plot_n_image(X, n=64)

# 运行PCA算法，获取特征向量
U, S, V = pca(X)
print('U.shape:', U.shape)
# plot_n_image(U, n=64)
# 1024维降维到100维
Z = project_data(X, U, k=100)
# plot_n_image(Z, n=64)
# 恢复数据
X_recover = recover_data(Z, U)
plot_n_image(X_recover, n=64)
