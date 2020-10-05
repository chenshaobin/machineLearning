#!/usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context="notebook", style="white")

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.decomposition import PCA

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

sk_pca = PCA(n_components=100)   # 保留100个主成分
Z = sk_pca.fit_transform(X)
# print(Z.shape)
X_recover = sk_pca.inverse_transform(Z)
# plot_n_image(X, n=64)
# plot_n_image(X_recover, n=64)
