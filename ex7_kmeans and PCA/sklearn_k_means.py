#!/usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.cluster import KMeans


def data_load_plot(path):
    data_origin = sio.loadmat(path)
    # print(data_origin.keys())
    data = pd.DataFrame(data_origin.get('X'), columns=['X1', 'X2'])
    # print(data.shape)
    # print(data.head())
    # 画出数据分布图
    """
    sns.set(context="notebook", style='dark')
    sns.lmplot('X1', 'X2', data=data, fit_reg=False)
    plt.show()
    """
    return data
def combine_data(data, C):
    # 为DataFrame数据添加一个新的列数据
    data_buff = data.copy()
    data_buff['C'] = C
    return data_buff
# load_plot('data/ex7data1.mat')

data = data_load_plot('data/ex7data2.mat')
sk_k_means = KMeans(n_clusters=3)
sk_k_means.fit(data)
sk_cluster_index = sk_k_means.predict(data)
data_with_cluster_index = combine_data(data, sk_cluster_index)
sns.lmplot('X1', 'X2', hue='C', data=data_with_cluster_index, fit_reg=False)
plt.show()
