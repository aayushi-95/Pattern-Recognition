# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 21:16:05 2020

@author: Aayushi Agarwal
"""

import numpy as np
import pandas as pd

#Read values from excel sheet and extract 3 column values in 3 variables
table = pd.read_excel("Q1-Data_HW6.xlsx")

x = table.iloc[:,1].values
y = table.iloc[:,2].values
L = table.iloc[:,3].values


dist_x = x - 4.9
dist_y = y - 6.2

#Calculating Euclidian distance

dist = np.sqrt(np.square(dist_x) + np.square(dist_y))

#Combine with label
dist= (dist,L)

ans = sorted(dist, key=lambda t:t[0])

#Calculate value and label of 1-NN and 3-NN
ans1 = ans[0][0],ans[1][0]
ans3 = ans[0][0:3],ans[1][0:3]

print('Value and decision according to 1-nearest neighbor classifier is: ',ans1[0], ans1[1])
print('')
print('Value and decision according to 3-nearest neighbor classifier is: ',ans3[0], ans3[1])
