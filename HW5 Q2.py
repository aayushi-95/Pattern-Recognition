# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 19:14:38 2020

@author: Aayushi Agarwal
"""

import numpy as np
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt

X = np.array([[1,3],[2,3],[-1,1],[-2,0.5]])
Y = np.array([1,1,-1,-1])
h = .02

clf = Perceptron(n_iter=100).fit(X, Y)

xx, yy = np.meshgrid(np.arange(-6.0, 8.0, h),
                     np.arange(-1.5,6.0, h))

# Plot the decision boundary.
fig, ax = plt.subplots()
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])


Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, cmap=plt.cm.magma)
ax.axis('off')

# Test vector check
plt.scatter(1,-1, marker='+', linewidths=20, color='blue')
ax.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.summer_r)

ax.set_title('Perceptron')