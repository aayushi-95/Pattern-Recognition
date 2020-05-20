# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 10:25:13 2020

@author: Aayushi Agarwal
"""
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sn
#from sklearn.cluster import KMeans
import scipy.linalg as la
import pandas as pd
import numpy as np

table = pd.read_excel("data2.xlsx")


Data = {'X': table.iloc[:,0].values,
        'Y': table.iloc[:,1].values
       }
df = DataFrame(Data,columns=['X','Y'])

muVector = np.mean(df, axis=0)
print(muVector)
covMatrix = pd.DataFrame.cov(df)
print (covMatrix)
sn.heatmap(covMatrix, annot=True, fmt='g')
plt.show()

eigvals, eigvecs = la.eig(covMatrix)
print(eigvals)
print(eigvecs)