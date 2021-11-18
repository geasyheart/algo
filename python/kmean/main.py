from sklearn.cluster import KMeans
import numpy as np
data = [3, 3, 2, 10, 11, 12, 20, 21, 22]
data = np.array(data).reshape(-1, 1)
kmean = KMeans(n_clusters=3)
kmean.fit(data)
print('h')