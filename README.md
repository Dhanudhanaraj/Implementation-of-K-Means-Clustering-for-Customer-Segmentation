# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:

To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:

1. Hardware – PCs
  
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Choose the number of clusters (K): Decide how many clusters you want to identify in your data. This is a hyperparameter that you need to set in advance.

2.Initialize cluster centroids: Randomly select K data points from your dataset as the initial centroids of the clusters.

3.Assign data points to clusters: Calculate the distance between each data point and each centroid. Assign each data point to the cluster with the closest centroid. This step is typically done using Euclidean distance, but other distance metrics can also be used.

4.Update cluster centroids: Recalculate the centroid of each cluster by taking the mean of all the data points assigned to that cluster.

5.Repeat steps 3 and 4: Iterate steps 3 and 4 until convergence. Convergence occurs when the assignments of data points to clusters no longer change or change very minimally.

6.Evaluate the clustering results: Once convergence is reached, evaluate the quality of the clustering results. This can be done using various metrics such as the within-cluster sum of squares (WCSS), silhouette coefficient, or domain-specific evaluation criteria.

7.Select the best clustering solution: If the evaluation metrics allow for it, you can compare the results of multiple clustering runs with different K values and select the one that best suits your requirements

## Program:
```

Program to implement the K Means Clustering for Customer Segmentation.
Developed by:Dhanumalya.D
Register Number:212222230030  

```
```

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("Mall_Customers (1).csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.cluster import KMeans
wcss=[]

for i in range (1,11):
    kmeans=KMeans(n_clusters = i,init="k-means++")
    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel("No. of clusters")
plt.ylabel("wcss")
plt.title("Elbow matter")

km=KMeans(n_clusters=5)
km.fit(data.iloc[:,3:])

y_pred=km.predict(data.iloc[:,3:])
y_pred

data["cluster"]=y_pred
df0=data[data["cluster"]==0]
df1=data[data["cluster"]==1]
df2=data[data["cluster"]==2]
df3=data[data["cluster"]==3]
df4=data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="black",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="blue",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="green",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="magenta",label="cluster4")
plt.legend()
plt.title("Customer Segmets")
```
## Output:
### data.head():
![Screenshot from 2023-10-30 23-33-56](https://github.com/Dhanudhanaraj/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119218812/341b54c6-a8d0-4699-94d5-a5f0b1d784f0)

### data.info():
![Screenshot from 2023-10-30 23-34-07](https://github.com/Dhanudhanaraj/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119218812/aae5f707-ea79-4811-bb21-be41ff1784bd)

### Null Values:
![Screenshot from 2023-10-30 23-34-18](https://github.com/Dhanudhanaraj/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119218812/1b909cd6-f6be-4efc-b998-137bfb41c9c8)

### Elbow Graph:
![Screenshot from 2023-10-30 23-34-32](https://github.com/Dhanudhanaraj/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119218812/4ec044b3-4f84-4db4-b955-a9543e586b9c)

### K-Means Cluster Formation:
![Screenshot from 2023-10-30 23-48-11](https://github.com/Dhanudhanaraj/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119218812/653c9ca1-e9c3-4b64-a6e1-62332edda410)


###   Predicted Value:
![Screenshot from 2023-10-30 23-35-24](https://github.com/Dhanudhanaraj/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119218812/d114a782-8e4a-4ffa-8260-35a2cfcc87e5)

### Final Graph:
![Screenshot from 2023-10-30 23-35-39](https://github.com/Dhanudhanaraj/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119218812/5306741b-0524-4b61-bc19-759830c1b33d)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
