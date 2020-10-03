#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sklearn.svm
import seaborn as sns
import scipy.io as sio
import matplotlib.pyplot as plt

data_origin = sio.loadmat('data/ex6data1.mat')
# print(data_origin.keys())
data = pd.DataFrame(data_origin.get('X'), columns=['X1', 'X2'])
data['y'] = data_origin.get('y')
# print(data)
# 画出原始数据
"""
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(data['X1'], data['X2'], s=40, c=data['y'], cmap='RdBu')
ax.set_title('Row datas')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
plt.show()
"""
# print('data[[\'X1\', \'X2\']]:',data[['X1', 'X2']])

def linear_SVM_train(data, c=1):
    svm_demo = sklearn.svm.LinearSVC(C=c, loss='hinge')
    svm_demo.fit(data[['X1', 'X2']], data['y'])     # 训练模型
    accuracy = svm_demo.score(data[['X1', 'X2']], data['y'])   # 准确率
    print('accuracy:', accuracy)
    data['SVM Confidence'] = svm_demo.decision_function(data[['X1', 'X2']])     # 每个样本的预测信心分数，
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(data['X1'], data['X2'], s=40, c=data['SVM Confidence'], cmap='RdBu')
    ax.set_title('SVM (C={}) Decision Confidence'.format(c))
    data.head()
    plt.show()

# linear_SVM_train(data, c=1)
linear_SVM_train(data, c=100)


