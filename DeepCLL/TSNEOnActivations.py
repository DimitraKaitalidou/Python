# Copyright 2020, Dimitra S. Kaitalidou, All rights reserved

from time import time
import numpy as np
import matplotlib.pyplot as plt
import create_gif as gif

from xlrd import open_workbook
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold
plt.rcParams["font.family"] = "Times New Roman"

# Initialize parameters
rows_all = []
rows_00000 = []
rows_00001 = [] 
rows_00012 = [] 
rows_00027 = [] 
rows_00189 = [] 
rows_00758 = [] 
rows_00144 = [] 
rows_00148 = [] 
rows_00128 = [] 
rows_00133 = [] 
rows_00002 = [] 
rows_00792 = [] 
rows_00063 = [] 
rows_01590 = [] 
rows_00079 = [] 
rows_00213 = [] 
rows_00570 = [] 
rows_00097 = [] 
rows_00238 = [] 

book = open_workbook("bio.xlsx")

for sheet in book.sheets():

    for rowidx in range(sheet.nrows):

        row = sheet.row(rowidx)

        for colidx, cell in enumerate(row):

            if cell.value == "CLUSTER-0-0000" :

                rows_00000.append(rowidx - 1) # minus 1 because there' s a label row in bio.xlsx
                rows_all.append(rowidx - 1)

            elif cell.value == "CLUSTER-0-0001" :

                rows_00001.append(rowidx - 1)
                rows_all.append(rowidx - 1)

            elif cell.value == "CLUSTER-0-0012" :

                rows_00012.append(rowidx - 1)
                rows_all.append(rowidx - 1)

            elif cell.value == "CLUSTER-0-0027" :

                rows_00027.append(rowidx - 1)
                rows_all.append(rowidx - 1)

            elif cell.value == "CLUSTER-0-0189" :

                rows_00189.append(rowidx - 1)
                rows_all.append(rowidx - 1)

            elif cell.value == "CLUSTER-0-0758" :

                rows_00758.append(rowidx - 1)
                rows_all.append(rowidx - 1)

            elif cell.value == "CLUSTER-0-0144" :

                rows_00144.append(rowidx - 1)
                rows_all.append(rowidx - 1)

            elif cell.value == "CLUSTER-0-0148" :

                rows_00148.append(rowidx - 1)
                rows_all.append(rowidx - 1)

            elif cell.value == "CLUSTER-0-0128" :

                rows_00128.append(rowidx-1)
                rows_all.append(rowidx - 1)

            elif cell.value == "CLUSTER-0-0133" :

                rows_00133.append(rowidx - 1)
                rows_all.append(rowidx - 1)

            elif cell.value == "CLUSTER-0-0002" :

                rows_00002.append(rowidx - 1)
                rows_all.append(rowidx - 1)

            elif cell.value == "CLUSTER-0-0792" :

                rows_00792.append(rowidx - 1)
                rows_all.append(rowidx - 1)

            elif cell.value == "CLUSTER-0-0063" :

                rows_00063.append(rowidx - 1)
                rows_all.append(rowidx - 1)

            elif cell.value == "CLUSTER-0-1590" :

                rows_01590.append(rowidx - 1)
                rows_all.append(rowidx - 1)

            elif cell.value == "CLUSTER-0-0079" :

                rows_00079.append(rowidx - 1)
                rows_all.append(rowidx - 1)

            elif cell.value == "CLUSTER-0-0213" :

                rows_00213.append(rowidx-1)
                rows_all.append(rowidx - 1)

            elif cell.value == "CLUSTER-0-0570" :

                rows_00570.append(rowidx - 1)
                rows_all.append(rowidx - 1)

            elif cell.value == "CLUSTER-0-0097" :

                rows_00097.append(rowidx - 1)
                rows_all.append(rowidx - 1)

            elif cell.value == "CLUSTER-0-0238" :

                rows_00238.append(rowidx - 1)
                rows_all.append(rowidx - 1)

activations = np.load("activations/model A/act.npy")
k, l, m, n = activations.shape

act_cl = activations[rows_all, :, :, :]
print(len(rows_all))

total = np.zeros(shape = (l * len(rows_all), n))
cs1 = np.zeros(shape = (k, 3))
cs2 = np.zeros(shape = (len(rows_all), 3))
cs3 = np.zeros(shape = (l * len(rows_all), 3))

for i in range(0, len(rows_all)):

	for j in range(0, l):
	
		total[i * l + j, :] = act_cl[i, j, 0, :]

# Create the colors for the best clustering representation
cs1[rows_00000, :] = [0.00, 0.00, 1.00] # blue
cs1[rows_00001, :] = [1.00, 0.00, 0.00] # red
cs1[rows_00012, :] = [0.00, 1.00, 0.00] # green
cs1[rows_00027, :] = [0.00, 0.00, 0.00] # black
cs1[rows_00189, :] = [1.00, 1.00, 0.00] # yellow
cs1[rows_00758, :] = [1.00, 0.00, 1.00] # magenda
cs1[rows_00144, :] = [0.00, 1.00, 1.00] # cyan
cs1[rows_00148, :] = [0.98, 0.80, 0.91] # classic rose
cs1[rows_00128, :] = [1.00, 0.86, 0.35] # mustard
cs1[rows_00133, :] = [0.13, 0.70, 0.67] # light sea green
cs1[rows_00002, :] = [0.50, 0.50, 0.00] # olive green
cs1[rows_00792, :] = [0.50, 0.50, 0.00] # pale robin egg blue
cs1[rows_00063, :] = [0.70, 0.75, 0.71] # ash grey
cs1[rows_01590, :] = [1.00, 0.44, 0.37] # bittersweet
cs1[rows_00079, :] = [0.36, 0.54, 0.66] # airforce blue
cs1[rows_00213, :] = [0.43, 0.21, 0.10] # auburn (~brown)
cs1[rows_00570, :] = [0.58, 0.00, 0.83] # dark violet
cs1[rows_00097, :] = [0.55, 0.00, 0.00] # dark red
cs1[rows_00238, :] = [1.00, 0.49, 0.00] # amber (~orange)

cs2 = cs1[rows_all, :]

for i in range(0, len(rows_all)):

	for j in range(0, l):

		cs3[i * l + j, :] = cs2[i, :]

# Comment or uncomment in order to enable the 2D or the 3D representation

"""
# Perform t-distributed stochastic neighbor embedding 2D.
t0 = time()
tsne = manifold.TSNE(n_components = 2, init = "pca", random_state = 0)
trans_data = tsne.fit_transform(total).T
t1 = time()
print("t-SNE: %.2g sec" % (t1 - t0))

s = plt.scatter(trans_data[0], trans_data[1], color = cs3, cmap = plt.cm.rainbow)
plt.title("t-SNE (%.2g sec) of 20 ground truth clusters" % (t1 - t0))
plt.axis("tight")
"""

# Perform t-distributed stochastic neighbor embedding 3D
Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection = "3d")
t0 = time()
tsne = manifold.TSNE(n_components = 3, init = "pca", random_state = 0)
trans_data = tsne.fit_transform(total).T
t1 = time()
print("t-SNE: %.2g sec" % (t1 - t0))

ax.scatter(trans_data[0], trans_data[1], trans_data[2], color = cs3)
plt.title("Model A")
plt.axis("off")
angles = np.linspace(0, 360, 21)[:-1] # take 20 angles between 0 and 360
gif.rotanimate(ax, angles, 'movieA.gif', delay = 20) # create an animated gif (20ms between frames) 

plt.show()