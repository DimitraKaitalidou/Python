# DeepCLL
This project contains the most important scripts that were developed in the context of the master thesis with title: “Deep learning techniques for extraction and visualization of immunogenic characteristics from patients with chronic lymphocytic leukemia”. The full text of the master thesis is available [here](https://www.researchgate.net/publication/326405523_Deep_learning_techniques_for_extraction_and_visualization_of_immunogenic_characteristics_from_patients_with_chronic_lymphocytic_leukemia) and the presentation is available [here](https://docs.google.com/presentation/d/10t-XL0bSHfuboE6vJGLVRTmNVyGcrN7ocYGqGqsxLf0/edit?usp=sharing).

## Summary
In this master thesis, an automatic clustering method of leukemic patients (Chronic Lymphocytic Leukemia - CLL) is being proposed, which is based on Deep Learning models and a clustering algorithm, in order to learn functional relationships from the biological data without the need to define them a priori. The method was applied on real data and it discovered over 90% of the classification of the previous non automatic widely accepted method, which is based on the Teiresias algorithm. Additionally, to increase the validity of the method, it was evaluated with the means of many visualization methods and a comparative analysis.

## Data
The data were biosequences, i.e., nucleotide or aminoacid sequences of CLL patients. Because the sequences varied significantly in length (21 to 72 for nucleotide sequences and according to the triplet code, 7 to 24 for aminoacid sequences), they were divided into groups of one length (same length) and groups of three lengths (similar length). The Python scripts were applied to the data after they were divided into groups. Unfortunately, the biosequences cannot be publicly available in the context of this project.

## Model description
Details regarding the model are provided in the master thesis report and the presentation. Concisely:
- The CNN model used in the thesis was based on the SegNet model. Multiple architectures were evaluated before choosing the final one. For nucleotide sequences the model uses three layers and for aminoacid sequences two layers
- The K-Means algorithm was applied on the activations of the CNN model
- The method was verified using class specific and kernel specific visualizations with sequence logos and confusion matrices

## Implementations
The following major implementations were performed:
1.	TSNEOnActivations.py: 
    - Scope: Application of the t-SNE method to select the most appropriate CNN model
    - Description: The script receives as input activations from a layer of the CNN model and applies the t-SNE dimensionality reduction in order to bring closer similar data points and create more distance between dissimilar data points. Then, colors are applied to the clusters known from the Teiresias method in order to decide which model is more appropriate
2.	KMeansOnActivations.py:
    - Scope: Application of the K-Means algorithm on the activations of a layer of the CNN in order to cluster them
    - Description: The script reads the activations and performs the K-Means algorithm using different number of clusters. Then, it calculates the Silhouette coefficient to decide the optimal number of clusters. The script saves an excel file with the cluster assignments, which are used as input for class specific visualization. Finally, the clusters produced by the K-Means algorithm are depicted in comparison to the Teiresias clusters using confusion matrices
3.	Visualization using sequence logos:
    - KernelSpecificVisualization.py:
        - Scope: Visualize the kernels that the CNN model learnt
        - Description: The script receives as input the kernels and batch normalization parameters for the 3 layers. The visualization can be performed after each convolution depending on the execution parameters of the script. Here is how it works: Given a kernel K and a sequence S with which the kernel has been convoluted, if an activation is above the threshold T, find the position of maximum activation and use that to find the part of the initial sequence that the maximum activation corresponds to. Do that for all sequences, align the parts that correspond to maximum activations and use that alignment to create the sequence logo. This sequence logo shows the motif that kernel K is looking for in the sequences. In order to execute the script, the command must be in the following format: 
<br/> ```python KernelSpecificVisualization.py T 3x32_2p_layer1_filters.npy 3x32_2p_layer2_filters.npy 3x32_2p_layer3_filters.npy```
    - ClassSpecificVisualization{Amino/Nucleo}.py:
        - Scope: Visualize the clusters that the K-Means algorithm produced 
        - Description:  The script receives as input the group of sequences of the same or similar length and their clusters after the K-Means algorithm. Then, the sequences of each cluster are aligned and the sequence logo of each cluster is produced

### Important note
The data used during the master thesis are not publicly available and the scripts cannot be executed without them. This project was created only for reviewing purposes.

## Python libraries and dependencies
1.	Peter J. A. Cock, Tiago Antao, Jeffrey T. Chang, Brad A. Chapman, Cymon J. Cox, Andrew Dalke, Iddo Friedberg, Thomas Hamelryck, Frank Kauff, Bartek Wilczynski, Michiel J. L. de Hoon, Biopython: freely available Python tools for computational molecular biology and bioinformatics, Bioinformatics, Volume 25, Issue 11, 1 June 2009, Pages 1422–1423, https://doi.org/10.1093/bioinformatics/btp163
2.	Travis E. Oliphant. A guide to NumPy, USA: Trelgol Publishing, (2006).
3.	https://xlrd.readthedocs.io/en/latest/index.html
4.	https://docs.python.org/3/library/itertools.html
5.	https://docs.python.org/3/library/sys.html
6.	Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, CJ Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E.A. Quintero, Charles R Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors. (2020) SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, in press.
7.	Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
8.	McKinney W, others. Data structures for statistical computing in python. In: Proceedings of the 9th Python in Science Conference. 2010. p. 51–6.
9.	https://seaborn.pydata.org/
10.	John D. Hunter. Matplotlib: A 2D Graphics Environment, Computing in Science & Engineering, 9, 90-95 (2007), DOI:10.1109/MCSE.2007.55
11.	https://docs.python.org/3/library/time.html
12.	https://zulko.wordpress.com/2012/09/29/animate-your-3d-plots-with-pythons-matplotlib/
13.	François Chollet et al. Keras. https://github.com/fchollet/keras, 2015.

## References
1.	Peter J. A. Cock, Tiago Antao, Jeffrey T. Chang, Brad A. Chapman, Cymon J. Cox, Andrew Dalke, Iddo Friedberg, Thomas Hamelryck, Frank Kauff, Bartek Wilczynski, Michiel J. L. de Hoon, Biopython: freely available Python tools for computational molecular biology and bioinformatics, Bioinformatics, Volume 25, Issue 11, 1 June 2009, Pages 1422–1423, https://doi.org/10.1093/bioinformatics/btp163
2.	Travis E. Oliphant. A guide to NumPy, USA: Trelgol Publishing, (2006).
3.	https://xlrd.readthedocs.io/en/latest/index.html
4.	https://docs.python.org/3/library/itertools.html
5.	https://docs.python.org/3/library/sys.html
6.	Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, CJ Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E.A. Quintero, Charles R Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors. (2020) SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, in press.
7.	Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
8.	McKinney W, others. Data structures for statistical computing in python. In: Proceedings of the 9th Python in Science Conference. 2010. p. 51–6.
9.	https://seaborn.pydata.org/
10.	John D. Hunter. Matplotlib: A 2D Graphics Environment, Computing in Science & Engineering, 9, 90-95 (2007), DOI:10.1109/MCSE.2007.55
11.	https://docs.python.org/3/library/time.html
12.	https://zulko.wordpress.com/2012/09/29/animate-your-3d-plots-with-pythons-matplotlib/
13.	François Chollet et al. Keras. https://github.com/fchollet/keras, 2015.
