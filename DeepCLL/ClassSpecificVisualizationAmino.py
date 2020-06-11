# Copyright 2020, Dimitra S. Kaitalidou, All rights reserved

import numpy as np
from Bio import motifs
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC, generic_protein
from xlrd import open_workbook
from itertools import repeat

# Initialize parameters
seqs = np.load("sequences7-9.npy")
k, l, m = seqs.shape
stack1 = [[] for i in repeat(None, 100)]
stack2 = [[] for i in repeat(None, 100)]

# Read and scan the input files
book1 = open_workbook("aminoacid_clustering.xlsx")
for sheet1 in book1.sheets():
	for rowidx1 in range(1, sheet1.nrows):
		seq_l = np.empty((1, l), dtype = str)
		for s in range(l):
			if np.any(seqs[rowidx1 - 1][s][:]):
				if seqs[rowidx1 - 1][s][0] == 1:
					seq_l[0][s] = "A"
				elif seqs[rowidx1 - 1][s][1] == 1:
					seq_l[0][s] = "R"
				elif seqs[rowidx1 - 1][s][2] == 1:
					seq_l[0][s] = "Q"
				elif seqs[rowidx1 - 1][s][3] == 1:
					seq_l[0][s] = "W"
				elif seqs[rowidx1 - 1][s][4] == 1:
					seq_l[0][s] = "L"
				elif seqs[rowidx1 - 1][s][5] == 1:
					seq_l[0][s] = "V"
				elif seqs[rowidx1 - 1][s][6] == 1:
					seq_l[0][s] = "P"
				elif seqs[rowidx1 - 1][s][7] == 1:
					seq_l[0][s] = "Y"
				elif seqs[rowidx1 - 1][s][8] == 1:
					seq_l[0][s] = "F"
				elif seqs[rowidx1 - 1][s][9] == 1:
					seq_l[0][s] = "D"
				elif seqs[rowidx1 - 1][s][10] == 1:
					seq_l[0][s] = "E"
				elif seqs[rowidx1 - 1][s][11] == 1:
					seq_l[0][s] = "H"
				elif seqs[rowidx1 - 1][s][12] == 1:
					seq_l[0][s] = "C"
				elif seqs[rowidx1 - 1][s][13] == 1:
					seq_l[0][s] = "M"
				elif seqs[rowidx1 - 1][s][14] == 1:
					seq_l[0][s] = "S"
				elif seqs[rowidx1 - 1][s][15] == 1:
					seq_l[0][s] = "T"
				elif seqs[rowidx1 - 1][s][16] == 1:
					seq_l[0][s] = "K"
				elif seqs[rowidx1 - 1][s][17] == 1:
					seq_l[0][s] = "N"
				elif seqs[rowidx1 - 1][s][18] == 1:
					seq_l[0][s] = "I"
				elif seqs[rowidx1 - 1][s][19] == 1:
					seq_l[0][s] = "G"
			else:
				seq_l[0][s] = "X"

		sequence_l = Seq(seq_l.tostring(), IUPAC.extended_protein)
		stack1[int(sheet1.cell(rowidx1, 3).value)].append(sequence_l) 
		stack2[int(sheet1.cell(rowidx1, 4).value)].append(sequence_l)

stack1 = [x for x in stack1 if x != []]
stack2 = [x for x in stack2 if x != []]

# Create the logo sequences
print "Visualizations from the 1st layer..."
for i in range(len(stack1)):
	m = motifs.create(stack1[i])
	m.weblogo("km_amino_layer1_cluster_" + str(i) + ".png", color_scheme = "color_classic")

print "Visualizations from the 2nd layer..."
for j in range(len(stack2)):
	m = motifs.create(stack2[j])
	m.weblogo("km_amino_layer2_cluster_" + str(j) + ".png", color_scheme = "color_classic")
