# EnhancerTracker

Copyright (C) 2023 Rolando Garcia, Anthony B. Garza, Luis M. Solis, Mark S. Halfon, and Hani Z. Girgis

Academic use: Affero General Public License version 1.

Any restrictions to use for profit or non-academics: Alternative commercial license is required.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

Please contact Dr. Hani Z. Girgis (hzgirgis@buffalo.edu) if you need more information.

EnhancerTracker is a computation tool for assessing the similarity among three sequences taking into account their enhancer activities. This repository contains multiple notebooks and scripts in Python that are used for our tool. EnhancerTracker uses an ensemble of deep convolutional neural network classifiers. These classifiers are trained on the FANTOM5 Project dataset (transcribed human enhancers). 

## Files

-create_dir.sh

Creates the multiple directories that are required for running EnhancerTracker.

-FantomData.ipynb:

Clean data of FANTOM5 depending on length of sequence wanted and split into separate training, validation, and testing datasets.

-CreateControls.ipynb:

Create control datasets that contain random sequences: LR, LNR, LGR, and LGNR.

-SplitControlsToSets.ipynb:

Split each of the control datasets into separate training, validation, and testing datasets depending on the ratio needed.

-FindSimilarEnhancers.ipynb:

Creates a pickle file containing an array of similar enhancers (active in same tissue) for each enhancer.

-CreateEnhancerOnlyDataset.ipynb:

Creates an enhancer dataset taken from the processed FANTOM5 dataset.

-CreateControlDatasets.ipynb:

Assembles sequence triplets for each of the control datasets (LR, LNR, LGR, and LGNR). 

CombineControls.ipynb:

Combine all of the control datasets into one temporary dataset (not including enhancer dataset). 

-CreateCompositeDataset.ipynb:

Assembles sequence triplets of the composite dataset.
File lengths were taken from the split datasets (training, validation, and training) in CombineControls.ipynb.
Provide them as a list of tuples, where each tuple represents a mode of training, validation, or testing.
It should be as so: [(train_1, train_2, train_3, train_4), (valid_1, valid_2, valid_3, valid_4), (test_1, test_2, test_3, test_4)]

-FastaReverseComplement.ipynb:

Converts the fasta files (training, validation, and testing) of the composite dataset, control datasets, and the enhancer dataset to reverse complement, creating a new file for each. 

-Ensemble.ipynb:

Takes in a dataset (either a control, enhancer, or composite dataset) and the three dataset splits (training, validation, and testing) of sequences, reverse sequences, similar triplets, and dissimilar triplets, depending on the dataset used.
Creates a plot on the metrics on total models (determines the amount of models to use). 
The Ensemble outputs a consensus between the models. 

-MonteCarloDropoutEnsemble.ipynb:

Takes in a dataset (either a control, enhancer, or composite dataset) and the three dataset splits (training, validation, and testing) of sequences, reverse sequences, similar triplets, and dissimilar triplets, depending on the dataset used.
Creates a plot on the metrics on total models (determines the amount of models to use).
The Monte Carlo Dropout Ensemble outputs multiple different predictions on the same model. 

-HierarchicalClassifier.ipynb:

Takes in a dataset (either a control, enhancer, or composite dataset) and the three dataset splits (training, validation, and testing) of fantom matrixes, sequences, reverse sequences, similar triplets, and dissimilar triplets, depending on the dataset used.
The hierarchical classifier has two outputs: the main output is a multi-label and the secondary output is a single-label.
The main output determines which tissues a potential enhancer is active in. 
The secondary output determines whether a sequence is an enhancer. 

-TripletConfidenceGenerator.ipynb:

An interactive jupyter notebook that selects similar triplets based on confidence and saves to FASTA files.
Outputs triplet permutations or non-permutated triplets.
A region of two different sizes from the human genome are centered around the third part of a triplet.
Triplet permutations use a 10,000 bp region and non-permutated triplets use a 100,000 bp region.

## Tool

-EnhancerTracker.py:
Our tool EnhancerTracker.
Takes in a triplet permutation generated from TripletConfidenceGenerator.ipynb. 
Outputs predicted enhancer regions, confidence scores for regions, and segment regions for each window size.

## Requirements
Download FANTOM5 dataset https://zenodo.org/record/556775

Download human genome dataset ... 

Download Red (Girgis, H.Z. Red: an intelligent, rapid, accurate tool for detecting repeats de-novo on the genomic scale. BMC Bioinformatics 16, 227 (2015). https://doi.org/10.1186/s12859-015-0654-5)

Download BEDtools (Quinlan, A. R., & Hall, I. M. (2010). BEDTools: a flexible suite of utilities for comparing genomic features. Bioinformatics, 26(6), 841-842.) 

EnhancerTracker requires a tensorflow with gpu support.  

## To Run Tool

1. Navigate to the folder EnhancerTracker was cloned and perform the following command

2. Unzip "Models_29.zip" and make sure it contains 29 models.

3. Run EnhancerTracker.py:
   > python3 EnhancerTracker.py two_enhancers.fa sequence.fa output_folder/

## To Run our Tests: 
1. Store FANTOM5 dataset in Data/FANTOM as F5.hg38.enhancers.expression.usage.matrix
   
2. Store human genome dataset in Data/HG38 as HG38.fa
   
3. Run create_dir.sh

Ensure that the following directories exist beforehand: Data/HG38/Scaffolds and Data/HG38/Chromosomes

4. Run ProcessHG38.ipynb. 

Ensure the directory Data/RED_HG38/ exists beforehand.

5. Run Red on the following directory: Data/HG38/Chromosomes and Data/HG38/Scaffolds. (Use Red as so: red -gnm Data/HG38/Chromosomes/ -dir Data/HG38/Scaffolds/ -rpt Data/RED_HG38/ -frm 2)

6. Run FantomData.ipynb

Ensure the following directories exists beforehand: Data/Datasets, Data/Datasets/LR, Data/Datasets/LNR, Data/Datasets/LGR, and Data/Datasets/LGNR

7. Run CreateControls.ipynb

8. Run SplitControlsToSets.ipynb

9. Run FindSimilarEnhancers.ipynb
   
Ensure that the following directory exists beforehand: Data/Datasets/Enhancer

10. Run CreateEnhancerOnlyDataset.ipynb

11. Run CreateControlDatasets.ipynb

Ensure the following directory exists beforehand: Data/Datasets/All

12. Run CombineControls.ipynb

13. Run CreateCompositeDataset.ipynb

14. Run FastaReverseComplement.ipynb
    
Ensure the following directory exists beforehand: Data/Datasets/All/Models

15. Run Ensemble.ipynb
    
16. Run MonteCarloDropoutEnsemble.ipynb

17. Run HierarchicalClassifier.ipynb
    
Ensure the following directories exists beforehand: Data/Triplets, Data/GNM, Data/GNM/60, Data/GNM/70, Data/GNM/80, and Data/GNM/90

18. Run TripletConfidenceGenerator.ipynb and generate either triplet permutations or non-permutated triplets (used for Ensemble.ipynb).














