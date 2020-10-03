#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    # 需要把每封电子邮件转换成一个特征向量
    # 邮件垃圾分类
    #如何从电子邮件构建这样的特征向量。
    # 只使用电子邮件的正文(不包括电子邮件标题)
    # 处理电子邮件经常使用的一种方法是
    # “规范化”这些值，比如使所有的url都得到相同的处理，所有的数字都得到相同的处理，等等。
"""

from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

import scipy.io as sio
data_origin = sio.loadmat('data/spamTrain.mat')
print(data_origin)
