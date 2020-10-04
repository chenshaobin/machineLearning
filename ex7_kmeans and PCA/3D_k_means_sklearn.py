#!/usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.cluster import KMeans
from skimage import io


pic = io.imread('data/text.bmp') / 255
# print(pic.shape)
# print(pic)
# io.imshow(pic)
# plt.show()
data = pic.reshape(512 * 512, 3)
model = KMeans(n_clusters=16, n_init=100)

model.fit(data)
centroids = model.cluster_centers_
# print('centroids.shape:', centroids.shape)
model_cluster_index = model.predict(data)
# print('model_cluster_index.shape:', model_cluster_index.shape)
compressed_pic = centroids[model_cluster_index].reshape(512, 512, 3)    # 图片转换为原来的格式，方便显示

fig, ax = plt.subplots(1, 2)
ax[0].imshow(pic)
ax[1].imshow(compressed_pic)
plt.show()
