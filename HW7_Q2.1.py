# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 23:03:12 2020

@author: Aayushi Agarwal
"""

 # clustering dataset
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
table = pd.read_excel("Q1-Data_HW6.xlsx")


x1 = table.iloc[:,1].values
x2 = table.iloc[:,2].values
       

# create new plot and data
plt.plot()
X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
colors = ['b', 'g', 'c','y']
markers = ['o', 'v', 's','o']

# KMeans algorithm 
K = 4
kmeans_model = KMeans(n_clusters=K).fit(X)

print(kmeans_model.cluster_centers_)
centers = np.array(kmeans_model.cluster_centers_)

plt.plot()
plt.title('k means centroids')

for i, l in enumerate(kmeans_model.labels_):
    plt.plot(x1[i], x2[i], color=colors[l],marker=markers[l],ls='None')
    plt.xlim([3, 8])
    plt.ylim([1, 5])

plt.scatter(centers[:,0], centers[:,1], marker="x", color='k')
plt.show()