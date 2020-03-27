import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math

def myKMeans(P, C, point):

	# Choose random centroids
	centroids = random.sample(range(1, P), C)
	centroid_points = point[centroids, :]

	# Assign points to the cluster, whose centroid has the smallest distance from the point
	# Stop when there are no new assignments
	cluster_assign = np.zeros(P)
	previous_cluster_assign = np.ones(P)
	while np.not_equal(cluster_assign, previous_cluster_assign).any():
		previous_cluster_assign = cluster_assign
		for i in range(P):
			for j in range(C):
				distance = math.sqrt(math.pow(abs(centroid_points[j, 0] - point[i, 0]), 2) + math.pow(abs(centroid_points[j, 1] - point[i, 1]), 2))
				if j == 0:
					min = distance
					cluster_assign[i] = j
				else:
					if distance < min:
						min = distance
						cluster_assign[i] = j

		# Calculate the new centroids
		for j in range(C):
			centroid_points[j, 0] = np.mean(point[cluster_assign == j, 0])
			centroid_points[j, 1] = np.mean(point[cluster_assign == j, 1])

	# Return result
	return cluster_assign

# Create dataset in space
P = input("Enter the number of points in space:\n")
point = np.random.random((P, 2))
C = input("Enter the number of clusters:\n")

# Visualize dataset
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.scatter(point[:, 0], point[:, 1])
ax1.set_title("Points in space")

# Apply Scikit learn K-means
km = KMeans(n_clusters = C, init = 'random', n_init = 1)
scikit_kmeans_colors = km.fit_predict(point)

# Visualize Scikit learn K-means results
ax2.scatter(point[:, 0], point[:, 1], c = scikit_kmeans_colors)
ax2.set_title("Scikit learn K-means")

# Apply my K-means
my_kmeans_colors = myKMeans(P, C, point)

# Visualize my K-means results
ax3.scatter(point[:, 0], point[:, 1], c = my_kmeans_colors)
ax3.set_title("My K-means")
fig.suptitle("K-means")
plt.show()