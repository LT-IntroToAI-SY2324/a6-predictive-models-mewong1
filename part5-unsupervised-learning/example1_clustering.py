import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#imports the data
data = pd.read_csv("part5-unsupervised-learning/old_faithful_data.csv")
data = data[["eruptions", "waiting"]].values

#standardizes the data
scaler = StandardScaler().fit(data)
x_std = scaler.transform(data)

#sets the value of k and creates kmeans model
k = 2
km = KMeans(n_clusters=k).fit(x_std)

#returns centroid x, y values in a 2D array
centroids = km.cluster_centers_

#returns the cluster labels of each data point in the x_std data set
labels = km.labels_

#sets the size of the scatterplot
plt.figure(figsize=(5,4))

#plots the data points for each cluster
for i in range(k):
    cluster = data[labels == i]
    # print(cluster)
    plt.scatter(cluster[:,0], cluster[:,1])

#plots the centriods
# plt.scatter(, marker='X', s=100,
            # c='r', label='centroid')
print(centroids)
cent = scaler.inverse_transform(centroids)
plt.scatter(cent[:,0], cent[:,1], marker='X', s=100,
            c='r', label='centroid')
#labels the axes
plt.xlabel('Length of Eruption (minutes)')
plt.ylabel('Time Between Eruptions(minutes)')
plt.show()