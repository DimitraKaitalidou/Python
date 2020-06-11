# Copyright 2020, Dimitra S. Kaitalidou, All rights reserved

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import metrics
import sys

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from itertools import repeat
from xlrd import open_workbook

"""
====================================================================================================
Apply the K-Means algorithm
====================================================================================================
"""

# Initialize parameters
print "Initializing parameters..."
number_of_clusters = [i for i in range(5, 30)]
sc = []
layer = int(sys.argv[1])
subset_name = []
number_of_rows = 0

# Read the nucleotide sequences
if layer == 1:
	acts = np.load("Activations_model2p_layer1_21.npy")
elif layer == 2:
	acts = np.load("Activations_model2p_layer2_21.npy")
else:
	acts = np.load("Activations_model2p_layer3_21.npy")

k, l, m, n = acts.shape
print acts.shape
new_acts = np.zeros((k, l * n))
kmeans_colors = np.zeros((k, len(number_of_clusters)))

for x in range(k):
	for y in range(l):
		for z in range(n):
			new_acts[x, y * n + z] = acts[x, y, 0, z]

# Perform K-Means
print "Performing K-Means..."

for i in range(len(number_of_clusters)):
	kmeans_colors[:, i] = KMeans(n_clusters = number_of_clusters[i], max_iter = 1000, random_state = 170).fit_predict(new_acts) # compute cluster centers and predict cluster index for each sample
	sc.append(metrics.silhouette_score(new_acts, kmeans_colors[:, i], metric = 'euclidean'))

# Find the best number of clusters
max = sc[0]
best_cl = number_of_clusters[0]
best_cl_index = 0
for i in range(len(number_of_clusters)):
	if sc[i] > max:
		max = sc[i]
		best_cl = number_of_clusters[i]
		best_cl_index = i

print "Best number of clusters based on the Silhouette score is: " +str(best_cl)
np.savetxt("kmeans_layer" +str(layer)+ "_cl" +str(best_cl)+ ".csv", kmeans_colors[:, best_cl_index], delimiter = ",", comments = "")
count_nb_of_seqs = np.zeros((1, best_cl))
for i in range(k):
	count_nb_of_seqs[0, int(kmeans_colors[i, best_cl_index])] = count_nb_of_seqs[0, int(kmeans_colors[i, best_cl_index])] + 1

print "The number of sequences per cluster is: " 
print count_nb_of_seqs

"""
====================================================================================================
Create the confusion matrices
====================================================================================================
"""

# Find the indices of the clustered data
kmeans_cluster_of_indices = [[] for i in repeat(None, best_cl)]
for i in range(best_cl):
	for j in range(k):
		if kmeans_colors[j, best_cl_index] == i:
			kmeans_cluster_of_indices[i].append(j)

# Read the ground truth
book = open_workbook("labels21.xlsx")
for sheet in book.sheets():
	number_of_rows = sheet.nrows
	for rowidx in range(1, sheet.nrows):
		if (sheet.cell(rowidx, 2).value).encode("utf-8") not in subset_name:
			subset_name.append((sheet.cell(rowidx, 2).value).encode("utf-8"))

subset_name_unicode = [unicode(i, "utf-8") for i in subset_name]				
count = np.zeros((best_cl, len(subset_name)))

for k in range(best_cl):
	for l in range(len(kmeans_cluster_of_indices[k])):
		for sheet in book.sheets():
			cell_value = (sheet.cell(kmeans_cluster_of_indices[k][l] + 1, 2).value).encode('utf-8') 
			ix = subset_name.index(cell_value)
			count[k, ix] = count[k, ix] + 1

# Save normalized confusion matrix & non normalized confusion matrix
norm_count = []
count_transpose = map(list, zip(*count))
for i in count_transpose:
    a = 0
    tmp_count = []
    a = sum(i, 0)
    for j in i:
        tmp_count.append(float(j)/float(a))
    norm_count.append(tmp_count)
norm_count_transpose = map(list, zip(*norm_count))

cluster_names = [str(i) for i in reversed(range(best_cl))]

df_cm_norm = pd.DataFrame(norm_count_transpose, index = cluster_names, columns = subset_name_unicode)
df_cm = pd.DataFrame(count, index = cluster_names, columns = subset_name_unicode)
plt.figure(figsize = (10, 10))
sn.heatmap(df_cm_norm, annot = True, cmap = "Reds", fmt = '.1f')
plt.yticks(rotation = 0)
plt.xticks(rotation = 90)
plt.savefig("kmeans_cl" +str(best_cl)+ "_layer" +str(layer)+ "_norm.png")
plt.figure(figsize = (10, 10))
sn.heatmap(df_cm, annot = True, cmap = "Reds", fmt = '.0f')
plt.yticks(rotation = 0)
plt.xticks(rotation = 90)
plt.savefig("kmeans_cl" +str(best_cl)+ "_layer" +str(layer)+ ".png")
