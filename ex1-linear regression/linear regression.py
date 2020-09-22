#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
sns.set(context="notebook", style="whitegrid", palette="dark")
import matplotlib.pyplot as plt
# import tensorflow as tf
import numpy as np

data = pd.read_csv('ex1data1.txt', names=['population', 'profit'])
data.head()
data.info()
sns.lmplot('population', 'profit', data, fit_reg=False)
#plt.show()
"""
    # 多变量的假设 h 表示为：
    # ℎ𝜃(𝑥)=𝜃0+𝜃1𝑥1+𝜃2𝑥2+...+𝜃𝑛𝑥𝑛
    # 这个公式中有n+1个参数和n个变量，为了使得公式能够简化一些，引入 𝑥0=1 ，则公式转化为：
    # 此时模型中的参数是一个n+1维的向量，任何一个训练实例也都是n+1维的向量，特征矩阵X的维度是 m*(n+1),m为数据个数
    # 因此公式可以简化为： ℎ𝜃(𝑥)=𝜃𝑇𝑋 ，其中上标T代表矩阵转置。
"""

def get_X(df):
    # 读取特征population
    ones = pd.DataFrame({'ones': np.ones(len(df))})  # ones是m行1列的dataframe
    data = pd.concat([ones, df], axis=1)    # 根据列合并数据
    return data.iloc[:, :-1].values     # 返回ndarray,不是矩阵


def get_y(df):
    # 读取标签值
    return np.array(df.iloc[:, -1])     # df的最后一列


# print(get_y(df))
X = get_X(data)
y = get_y(data)
# print(y.shape, type(y))
def normalize_feature(df):
    # 特征缩放操作，对数据进行均值归一化
    return df.apply(lambda column:(column - column.mean()) / column.std())


