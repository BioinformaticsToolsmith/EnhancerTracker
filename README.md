# EnhancerSniffer
"Enhancer Sniffer is a computation tool for assessing the similarity among three sequences with respect to their enhancer activites." 

# Steps for EnhancerSniffer: 

Download FANTOM5 dataset https://zenodo.org/record/556775
Download human genome dataset ... 

Store FANTOM5 dataset in Data/FANTOM as F5.hg38.enhancers.expression.usage.matrix
Store human genome dataset in Data/HG38 as HG38.fa

Run ProcessHG38.ipynb. Ensure that the following directories exist beforehand: Data/HG38/Scaffolds and Data/HG38/Chromosomes

Download Red (Girgis, H.Z. Red: an intelligent, rapid, accurate tool for detecting repeats de-novo on the genomic scale. BMC Bioinformatics 16, 227 (2015). https://doi.org/10.1186/s12859-015-0654-5)

Ensure the directory Data/RED_HG38/ exists beforehand.
Run Red on the following directory: Data/HG38/Chromosomes and Data/HG38/Scaffolds.
Use Red as so: red -gnm Data/HG38/Chromosomes/ -dir Data/HG38/Scaffolds/ -rpt Data/RED_HG38/ -frm 2

FantomData.ipynb
Clean data of FANTOM5 depending on length of sequence wanted and split into separate training, validation, and testing datasets.

Ensure the following directories exists beforehand: Data/Datasets, Data/Datasets/LR, Data/Datasets/LNR, Data/Datasets/LGR, and Data/Datasets/LGNR
CreateControls.ipynb
Create control datasets that contain random sequences: LR, LNR, LGR, and LGNR.

SplitControlsToSets.ipynb
Split each of the control datasets into separate training, validation, and testing datasets depending on the ratio needed.

FindSimilarEnhancers.ipynb
Creates a pickle file containing an array of similar enhancers (active in same tissue) for each enhancer.

Ensure that the following directory exists beforehand: Data/Datasets/Enhancer
CreateEnhancerOnlyDataset.ipynb
Creates an enhancer dataset taken from the processed FANTOM5 dataset.

CreateControlDatasets.ipynb
Assembles sequence triplets for each of the control datasets (LR, LNR, LGR, and LGNR). 

Ensure the following directory exists beforehand: Data/Datasets/All
CombineControls.ipynb
Combine all of the control datasets into one temporary dataset (not including enhancer dataset). 

CreateCompositeDataset.ipynb
Assembles sequence triplets of the composite dataset.
File lengths were taken from the split datasets (training, validation, and training) in CombineControls.ipynb.
Provide them as a list of tuples, where each tuple represents a mode of training, validation, or testing.
It should be as so: [(train_1, train_2, train_3, train_4), (valid_1, valid_2, valid_3, valid_4), (test_1, test_2, test_3, test_4)]

FastaReverseComplement.ipynb
Converts the fasta files (training, validation, and testing) of the composite dataset, control datasets, and the enhancer dataset to reverse complement, creating a new file for each. 

Ensure the following directory exists beforehand: Data/Datasets/All/Models
Ensemble.ipynb
Takes in a dataset (either a control, enhancer, or composite dataset) and the three dataset splits (training, validation, and testing) of sequences, reverse sequences, similar triplets, and dissimilar triplets, depending on the dataset used.
Creates a plot on the metrics on total models (determines the amount of models to use). 
The Ensemble outputs a consensus between the models. 

MonteCarloDropoutEnsemble.ipynb
Takes in a dataset (either a control, enhancer, or composite dataset) and the three dataset splits (training, validation, and testing) of sequences, reverse sequences, similar triplets, and dissimilar triplets, depending on the dataset used.
Creates a plot on the metrics on total models (determines the amount of models to use).
The Monte Carlo Dropout Ensemble outputs multiple different predictions on the same model. 

HierarchicalClassifier.ipynb
Takes in a dataset (either a control, enhancer, or composite dataset) and the three dataset splits (training, validation, and testing) of fantom matrixes, sequences, reverse sequences, similar triplets, and dissimilar triplets, depending on the dataset used.
The hierarchical classifier has two outputs: the main output is a multi-label and the secondary output is a single-label.
The main output determines which tissues a potential enhancer is active in. 
The secondary output determines whether a sequence is an enhancer. 
 
Ensure the following directories exists beforehand: Data/Triplets, Data/GNM, Data/GNM/60, Data/GNM/70, Data/GNM/80, and Data/GNM/90
TripletConfidenceGenerator.ipynb
An interactive jupyter notebook that selects similar triplets based on confidence and saves to FASTA files.
