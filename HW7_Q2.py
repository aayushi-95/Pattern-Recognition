from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import distance
import pandas as pd
import numpy as np


#Read the data and fit it for k-means cluster model
table = pd.read_excel("Q1-Data_HW6.xlsx")


Data = {'x': table.iloc[:,1].values,
        'y': table.iloc[:,2].values
       }
df = DataFrame(Data,columns=['x','y'])


#Test Vector
X=[4.9,6.2] 
df1 = pd.DataFrame({
    'x1': [4.9],
    'x2': [6.2]
})


#Divide into 3 clusters and find cluster center
kmeans = KMeans(n_clusters=4).fit(df)

#Find nearest cluster label of test vector
labels = kmeans.predict(df1)
centroids = kmeans.cluster_centers_
print(centroids)

#Plot the clusters and respective centroids
plt.xlim([4, 7.5])
plt.ylim([1.5, 5])
plt.scatter(df['x'], df['y'], c= kmeans.labels_.astype(float), s=50, alpha=1.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.show()

#end

#check = kmeans.predict([[4.9,0],[0,6.2]])
#predict_me = np.array(X)
#predict_me = predict_me.reshape(-1, len(predict_me))
#prediction = kmeans.predict(predict_me)
#print(prediction)
#cluster_index = kmeans.labels_[table[4.9,6.2].index]
#print(distance.cosine(table[4.9,6.2], centroids[cluster_index]))


