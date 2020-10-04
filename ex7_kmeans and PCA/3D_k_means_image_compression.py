#!/usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from skimage import io
import matplotlib as mpl
# 防止中文乱码，设置字符集
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

pic = io.imread('data/text.bmp') / 255
# print(pic.shape)
# print(pic)
# io.imshow(pic)
# plt.show()

data = pic.reshape(512 * 512, 3)
# print('data:', data)

def combine_data(data, C):
    # 为DataFrame数据添加一个新的列数据
    data_buff = data.copy()
    data_buff['C'] = C
    return data_buff

def random_init_centroids(data, k):
    """
    # 随机选取k个簇中心
    :param data: DataFrame
    :param k: 簇的个数
    :return: k个簇，类型为ndarray
    """
    return data.sample(k).values

def find_closest_centroids(x, centroids):
    """
        #find the right cluster for x with respect to shortest distance
        :param x:   ndarry (n,)
        :param centroids: (k, n)
        :return: the index of centroid
    """
    # np.apply_along_axis(),沿给定的轴对1-D切片应用一个函数,distance的计算针对每一行数据
    # np.linalg.norm()  用于返回矩阵或者向量的范数
    distance = np.apply_along_axis(func1d=np.linalg.norm, axis=1, arr=centroids - x)
    return np.argmin(distance)

def assign_cluster(data, centroids):
    """
    #为每个样本分配到对应的簇中去
    :param data: 样本数据,DataFrame
    :param centroids: k个簇
    :return: 返回每个样本所属簇的索引值，类型为ndarry
    """
    # np.apply_along_axis(),沿给定的轴对1-D切片应用一个函数
    return np.apply_along_axis(func1d=lambda x: find_closest_centroids(x, centroids), axis=1, arr=data.values)

def new_centroids(data, C):
    """
        # 重新计算簇中心
        :param data: 样本数据，格式为DataFrame
        :param C:   簇索引值，根据它来对样本进行分组计算均值
        :return: 返回新的簇的坐标，ndarray
    """
    data_with_cluster_index = combine_data(data, C)
    return data_with_cluster_index.groupby('C', as_index=False).mean().sort_values(by='C').drop('C', axis=1).values

def cost(data, centroids, cluster_index):
    # 计算代价函数
    m = data.shape[0]
    data_centroids = centroids[cluster_index]   # 按照簇索引找到各样本对应的簇中心
    distance = np.apply_along_axis(func1d=np.linalg.norm, axis=1, arr=data.values - data_centroids)
    return distance.sum() / m

def k_means_iter(data, k, epoch=100, tol=0.0001):
    """
        # 只进行一次初始化参数，然后根据代价函数最小化完成最终的分类
        :param data: 样本数据,格式为DataFrame
        :param k: 簇的个数
        :param epoch: 设置的迭代次数，即根据首次分类后重新计算簇中心的次数
        :param tol: 迭代停止的精度设置
        :return: 样本所属簇的索引， 簇的坐标数据，最佳的代价函数值
    """
    centroids = random_init_centroids(data, k)  #只初始化一次
    cost_process = []   # 存放迭代时的代价函数值
    for i in range(epoch):
        print('running epoch {}'.format(i))
        cluster_index = assign_cluster(data, centroids)     # 得到各样本的簇中心索引
        centroids = new_centroids(data, cluster_index)  # 重新计算簇中心
        cost_process.append(cost(data, centroids, cluster_index))

        if len(cost_process) > 1:
            # 只要后面的代价函数比前一个的代价函数小得多的话就停止迭代
            if(np.abs(cost_process[-1] - cost_process[-2])) / cost_process[-1] < tol:
                break

    return cluster_index, centroids, cost_process[-1]

def k_means(data, k, epoch=100, n_init=10):
    """
    # 多次初始化簇中心点，然后根据代价函数最小化完成最终的分类
    :param data: 样本数据,格式为DataFrame
    :param k: 簇的个数
    :param epoch: 设置的迭代次数,即根据首次分类后重新计算簇中心的次数
    :param n_init: 初始化簇中心点的次数
    :return: 返回最小代价函数的样本所属簇的索引， 簇的坐标数据，最佳的代价函数值
    """
    k_means_data = np.array([k_means_iter(data, k, epoch) for _ in range(n_init)])
    least_cost_k_means_data_index = np.argmin(k_means_data[:, -1])
    return k_means_data[least_cost_k_means_data_index]

# print(pd.DataFrame(data))
best_cluster_index, best_centroids, least_cost = k_means(pd.DataFrame(data), k=16, epoch=10, n_init=3)
compress_pic = best_centroids[best_cluster_index].reshape(512, 512, 3)
# print(best_centroids.shape)
# print(best_cluster_index.shape)
fig, ax = plt.subplots(1, 2)
ax[0].set_title("原图像")
ax[1].set_title("压缩后的图像")
ax[0].imshow(pic)
ax[1].imshow(compress_pic)
plt.show()
