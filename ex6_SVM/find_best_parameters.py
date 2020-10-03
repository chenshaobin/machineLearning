#!/usr/bin/python
# -*- coding: utf-8 -*-

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

import numpy as np
import pandas as pd
import scipy.io as sio

# 数据预处理
data_origin = sio.loadmat('data/ex6data3.mat')
# print(data_origin.keys())
# print(data_origin.get('X').shape)   # (211,2)
# print(data_origin.get('Xval').shape)    #(200,2)
data_train = pd.DataFrame(data_origin.get('X'), columns=['X1', 'X2'])
data_train['y'] = data_origin.get('y')

data_cross = pd.DataFrame(data_origin.get('Xval'), columns=['X1', 'X2'])
data_cross['y'] = data_origin.get('yval')
# print(data_train.head())

# 由训练数据训练得到的参数，使用交叉数据集来选择最佳的C参数和gamma参数(人工选择)
candidate = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]   # 备选参数值
combination_C_sigma = [(C, gamma) for C in candidate for gamma in candidate]    # C参数和gamma参数的组合
accuracy_set = []
for C, gamma in combination_C_sigma:
    svc = svm.SVC(C=C, gamma=gamma)
    svc.fit(data_train[['X1', 'X2']], data_train['y'])
    accuracy_set.append(svc.score(data_cross[['X1', 'X2']], data_cross['y']))

high_accuracy = accuracy_set[np.argmax(accuracy_set)]
best_param = combination_C_sigma[np.argmax(accuracy_set)]
# print('manual:high_accuracy:{},best_param:{}'.format(high_accuracy, best_param))

# 根据得到的最佳参数，构建一个文本报告，显示主要的分类度量
best_svc = svm.SVC(C=0.3, gamma=100)
best_svc.fit(data_train[['X1', 'X2']], data_train['y'])
y_pred = best_svc.predict(data_cross[['X1', 'X2']])
result_report = metrics.classification_report(data_cross['y'], y_pred)
# print(result_report)

# 使用网格搜索,从训练集中寻找最佳参数
parameters = {'C': candidate, 'gamma': candidate}
svc = svm.SVC()
clf = GridSearchCV(estimator=svc, param_grid=parameters, n_jobs=-1)
clf.fit(data_train[['X1', 'X2']], data_train['y'])
best_param_clf = clf.best_params_
best_accuracy = clf.best_score_
print('GridSearchCV:high_accuracy:{},best_param:{}'.format(best_accuracy, best_param_clf))
y_pred_clf = clf.predict(data_cross[['X1', 'X2']])
result_report_clf = metrics.classification_report(data_cross['y'], y_pred_clf)
print(result_report_clf)

