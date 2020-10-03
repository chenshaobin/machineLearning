#!/usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.io as sio

def gaussian_kernel(x1, x2, sigma):
    # 创建高斯核函数
    return np.exp(- np.power(x1 - x2, 2).sum() / (2 * sigma ** 2))

data_origin = sio.loadmat('data/ex6data2.mat')
# print(data_origin)
data = pd.DataFrame(data_origin['X'], columns=['X1', 'X2'])
data['y'] = data_origin.get('y')
# print(data.head())

# 画出原始数据
"""
sns.set(context="notebook", style="dark", palette=sns.diverging_palette(240, 50, n=2))    # sns.diverging_palette:在两种HUSL颜色之间做一个调色板
sns.lmplot('X1', 'X2', hue='y', data=data, height=5, fit_reg=False, scatter_kws={'s': 10})  # 'X1', 'X2','y'为data中的columns
plt.show()
"""
# 使用基于libsvm的sklearn.svm.SVC来构建高斯核函数的SVM
svc = svm.SVC(C=100, kernel='rbf', gamma=10, probability=True)
svc.fit(data[['X1', 'X2']], data['y'])
accuracy = svc.score(data[['X1', 'X2']], data['y'])
print('accuracy:', accuracy)
predict_prob = svc.predict_proba(data[['X1', 'X2']])[:, 0]    # 计算X中样本可能结果的概率
# print(predict_prob)
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(data['X1'], data['X2'], s=30, c=predict_prob, cmap="RdBu")
plt.show()
