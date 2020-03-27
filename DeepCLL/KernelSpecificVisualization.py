# Copyright 2020, Dimitra S. Kaitalidou, All rights reserved

import numpy as np
import sys

from Bio import motifs
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
from xlrd import open_workbook
from scipy import signal
from itertools import repeat

# Initialize parameters
number_of_layers = len(sys.argv) - 2 # the number of parameters determines the layer the motifs come from
stack = [[] for i in repeat(None, 32)]
count = np.zeros((1, 32))

# Read the parameters of the trained model
if number_of_layers == 3:

	filename = str(sys.argv[2])
	kernels1 = np.load(filename)
	bn_mean_layer1 = np.loadtxt("model_parameters/bn1_mean.txt")
	bn_std_layer1 = np.loadtxt("model_parameters/bn1_std.txt")
	bn_beta_layer1 = np.loadtxt("model_parameters/bn1_beta.txt")
	bn_gamma_layer1 = np.loadtxt("model_parameters/bn1_gamma.txt")
	bias_layer1 = np.loadtxt("model_parameters/bias1.txt")

	filename = str(sys.argv[3])
	kernels2 = np.load(filename)
	bn_mean_layer2 = np.loadtxt("model_parameters/bn2_mean.txt")
	bn_std_layer2 = np.loadtxt("model_parameters/bn2_std.txt")
	bn_beta_layer2 = np.loadtxt("model_parameters/bn2_beta.txt")
	bn_gamma_layer2 = np.loadtxt("model_parameters/bn2_gamma.txt")
	bias_layer2 = np.loadtxt("model_parameters/bias2.txt")

	filename = str(sys.argv[4])
	kernels3 = np.load(filename)
	bias_layer3 = np.loadtxt("model_parameters/bias3.txt")
	print "Kernels from convolutional layer 3.\n" # the number of parameters determines the layer the motifs come from
	
elif number_of_layers == 2:

	filename = str(sys.argv[2])
	kernels1 = np.load(filename)
	bn_mean_layer1 = np.loadtxt("model_parameters/bn1_mean.txt")
	bn_std_layer1 = np.loadtxt("model_parameters/bn1_std.txt")
	bn_beta_layer1 = np.loadtxt("model_parameters/bn1_beta.txt")
	bn_gamma_layer1 = np.loadtxt("model_parameters/bn1_gamma.txt")
	bias_layer1 = np.loadtxt("model_parameters/bias1.txt")

	filename = str(sys.argv[3])
	kernels2 = np.load(filename)
	bn_mean_layer2 = np.loadtxt("model_parameters/bn2_mean.txt")
	bn_std_layer2 = np.loadtxt("model_parameters/bn2_std.txt")
	bn_beta_layer2 = np.loadtxt("model_parameters/bn2_beta.txt")
	bn_gamma_layer2 = np.loadtxt("model_parameters/bn2_gamma.txt")
	bias_layer2 = np.loadtxt("model_parameters/bias2.txt")
	print "Kernels from convolutional layer 2.\n" # the number of parameters determines the layer the motifs come from

elif number_of_layers == 1:

	filename = str(sys.argv[2])
	kernels1 = np.load(filename)
	bn_mean_layer1 = np.loadtxt("model_parameters/bn1_mean.txt")
	bn_std_layer1 = np.loadtxt("model_parameters/bn1_std.txt")
	bn_beta_layer1 = np.loadtxt("model_parameters/bn1_beta.txt")
	bn_gamma_layer1 = np.loadtxt("model_parameters/bn1_gamma.txt")
	bias_layer1 = np.loadtxt("model_parameters/bias1.txt")
	print "Kernels from convolutional layer 1.\n" # the number of parameters determines the layer the motifs come from

else:

	sys.exit("No input file with kernels given!")

k, l, m = kernels1.shape
threshold = float(sys.argv[1])

# Read and scan the input file
book = open_workbook("bio.xlsx")

for sheet in book.sheets():

	for rowidx in range(1, sheet.nrows):

		cell_value = str(sheet.cell(rowidx, 22).value) # column W contains the CDR3 sequences
		cell_value_padded_l = cell_value.ljust(125, "n") # letter "N" for any nucleotide, all sequences should be of the same length
		seq_l = list(cell_value_padded_l)
		res = np.zeros((4, 125), dtype = np.uint8)

		# Create one - hot encoding of the dna sequence
		for s in range(125):

			if seq_l[s] == "a":
				res[0][s] = 1
			elif seq_l[s] == "t":
				res[1][s] = 1
			elif seq_l[s] == "c":
				res[2][s] = 1		
			elif seq_l[s] == "g":
				res[3][s] = 1
			elif seq_l[s] == "r": # puRine (a, g)
				res[0][s] = 1
				res[3][s] = 1
			elif seq_l[s] == "y": # pYrimidine (c, t)
				res[1][s] = 1
				res[2][s] = 1	
			
		# Apply convolution for as many layers as needed
		rep_res = np.tile(res, (k, 1))
		a, b =  rep_res.shape	
		rep_res = np.reshape(rep_res, (k, 4, b))
		activations1 = np.zeros((k, 125)) # the activations after convolution of layer 1 are of length 123 but we initialize with length 125 to skip zero padding after batch normalization
				
		if number_of_layers > 1:

			slope_layer1 = np.divide(bn_gamma_layer1, np.sqrt(bn_std_layer1 + 0.001)) # 0.001 is the default value for the constant epsilon
			bias_layer1 = np.add(bn_beta_layer1, -np.multiply(slope_layer1, bn_mean_layer1))
		
			for x in range(k):

				activations1[x, 0:123] = signal.convolve2d(rep_res[x, :, :], kernels1[x, :, :], mode = "valid") # convolution
				activations1[x, 0:123] = activations1[x, 0:123] + bias_layer1[x]
				activations1[x, 0:123] = np.maximum(activations1[x, 0:123], 0) # rectified linear unit
				activations1[x, 0:123] = np.add(np.multiply(activations1[x, 0:123], slope_layer1), bias_layer1) # batch normalization

			activations2 = np.zeros((k, 123)) # the activations after convolution of layer 2 are of length 123, zero padding is perfomed after max pooling

			if number_of_layers > 2:

				maxpooling2 = np.zeros((k, 47)) # the activations after max pooling of layer 2 are of length 41 but we initialize with length 47 to skip zero padding 
				slope_layer2 = np.divide(bn_gamma_layer2, np.sqrt(bn_std_layer2 + 0.001)) # 0.001 is the default value for the constant epsilon
				bias_layer2 = np.add(bn_beta_layer2, -np.multiply(slope_layer2, bn_mean_layer2))

				for y in range(k):
	
					temp = np.reshape(kernels2[y, :, :,:], (k, 3))
					activations2[y, :] = signal.convolve2d(activations1, temp, mode = "valid") # convolution
					activations2[y, :] = activations2[y, :] + bias_layer2[y]
					activations2[y, :] = np.maximum(activations2[y, :], 0) # rectified linear unit
					activations2[y, :] = np.add(np.multiply(activations2[y, :], slope_layer2), bias_layer2) # batch normalization

					for ki in range(41):

						imax = np.argmax(activations2[y, ki * 3:ki * 3 + 3])
						maxpooling2[y, ki] = activations2[y, imax]
			
				activations3 = np.zeros((k, 45))

				# Find maximum activations for layer 3 and build the stack
				for z in range(k):

					temp = np.reshape(kernels3[z, :, :, :], (k, 3))
					activations3[z, :] = signal.convolve2d(maxpooling2, temp, mode = "valid") # convolution
					activations3[z, :] = activations3[z, :] + bias_layer3[z]
					for i in range(37):

						if activations3[z, i] > threshold:
		
							imax = np.argmax(activations3[z, 0:37]) # indices larger than 36 can not be projected on a 13-length part of the initial sequence
							str_value = str(cell_value_padded_l[imax * 3:imax * 3 + 13]).upper()
							sequence = Seq(str_value, IUPAC.ambiguous_dna)
							stack[z].append(sequence)
							count[0, z] = count[0, z] + 1
							break
								
			else:

				# Find maximum activations for layer 2 and build the stack
				for y in range(k):
	
					temp = np.reshape(kernels2[y, :, :, :], (k, 3))
					activations2[y, :] = signal.convolve2d(activations1, temp, mode = "valid") # convolution
					activations2[y, :] = activations2[y, :] + bias_layer2[y]
					for i in range(123):

						if activations2[y, i] > threshold:
		
							imax = np.argmax(activations2[y, :])
							str_value = str(cell_value_padded_l[imax:imax + 5]).upper()
							sequence = Seq(str_value, IUPAC.ambiguous_dna)
							stack[y].append(sequence)
							count[0, y] = count[0, y] + 1
							break

		else:
			
			# Find maximum activations for layer 1 and build the stack
			for x in range(k):

				activations1[x, 0:123] = signal.convolve2d(rep_res[x, :, :], kernels1[x, :, :], mode = "valid") # convolution
				activations1[x, 0:123] = activations1[x, 0:123] + bias_layer1[x]
				for i in range(123):

					if activations1[x, i] > threshold:
		
						imax = np.argmax(activations1[x, 0:123])
						str_value = str(cell_value_padded_l[imax:imax + 3]).upper()
						sequence = Seq(str_value, IUPAC.ambiguous_dna)
						stack[x].append(sequence)
						count[0, x] = count[0, x] + 1
						break

# Create the sequence logos
for i in range(k):
	print "Number of sequences above activation threshold for kernel " + str(i) + ": " + str(count[0, i]) + "\n" 
	if count[0, i] != 0:
		m = motifs.create(stack[i])
		print m.counts
		m.weblogo("layer" + str(number_of_layers) + "_threshold" + str(sys.argv[1]) + "_kernel" + str(i) + ".png", color_scheme = "color_classic")