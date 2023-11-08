#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import pandas as pd 
from sklearn import datasets 
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt # Load the Iris dataset 
iris= datasets.load_iris() 
data = pd.DataFrame(iris.data, columns=iris.feature_names)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

data_scaled = scaler.fit_transform(data)

k = 3 # Number of clusters

kmeans =KMeans (n_clusters=k, random_state=42) 
kmeans.fit(data_scaled) # Fit the model to the standardized data
KMeans (n_clusters=3, random_state=42)

data['Cluster'] = kmeans.labels_

# Plot the clusters

plt.scatter(data_scaled[:,0], data_scaled[:, 1],c=data['Cluster'], cmap='viridis')
plt.scatter (kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')

plt.title('K-Means Clustering of Iris Data')

plt.xlabel("Sepal Length (cm)")

plt.ylabel("Sepal Width (cm)")

plt.legend()

plt.show()


# In[ ]:




