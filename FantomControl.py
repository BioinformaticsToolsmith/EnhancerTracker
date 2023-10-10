import pandas as pd
import time
from multiprocessing import Pool, Manager
from itertools import repeat
import itertools
import numpy as np
import gc
import random
import pickle as pk
import os
from Bio import SeqIO
import Fantom

class ControlDriver(Fantom.Driver):
    
    def __init__(self, fantom_file, pickle_file, triplet_sim_file, triplet_dis_file, fasta_dir, sequence_file, control_file, cores, iteration):
        assert os.path.exists(control_file), f'{control_file} does not exist!'
        self.control_file = control_file
        
        super().__init__(fantom_file, pickle_file, triplet_sim_file, triplet_dis_file, fasta_dir, sequence_file, cores, iteration)
        self.crm_len = len(self.df_clean)
        self.control_len = len(list(SeqIO.parse(control_file, 'fasta')))        
        
    def make_negative_list(self):
        '''
        Creating a negative list from the control
        
        '''

        r = np.random.choice(range(self.crm_len, self.crm_len + self.control_len), size = self.iteration).astype(np.uintc)       
        return r
        
    def fill_anchor_block(self, i_list, is_similar):
        '''
        Creating the anchor, positive, and negative tensors and filling the anchor block with these tensors.
        Function that will be utilized for multiprocessing.
        '''
        block = np.zeros((len(i_list), 3, self.iteration), dtype=np.uintc)
        
        for index, anchor in enumerate(i_list):
            
            if index % 100 == 0:
                print(f'Processed {index} out of {len(i_list)}')
            
            # Fill the anchor's channels
            block[index, 0, :] = anchor

            # Fill the positive's channels
            anchor_sim_array = self.crm_list[anchor]
            is_replace = True if len(anchor_sim_array) < self.iteration else False
            positive_array = np.random.choice(anchor_sim_array, size=self.iteration, replace = is_replace).astype(np.uintc)
            block[index, 1, :] = positive_array
            
            # Fill the negative list or similar list
            if is_similar:
                # block[index, 2, :] = self.make_similar_list(positive_array)
                block[index, 2, :] = self.make_similar_list(anchor, anchor_sim_array, positive_array)

            else:
                block[index, 2, :] = self.make_negative_list()
        
        return block
        
        
    def write_sequences(self, fasta_header = "HG38_", chrom_list = ["chr" + str(x) for x in range(1, 23)] + ["chrX", "chrY"]):
        fasta_header = f"{self.fasta_dir}/{fasta_header}"                
        fasta_dict = {f"{chrom}": str(SeqIO.read(f'{fasta_header}{chrom}.fa', 'fasta').seq) for chrom in chrom_list}
        
        with open(self.sequence_file, 'w') as file:
            for label in list(self.df_clean.index):
                colon = label.find(":")
                dash = label.find("-")

                chrom = label[:colon]
                start = int(label[colon + 1:dash])
                end = int(label[dash + 1:])
                seq = fasta_dict[chrom][start:end]
                
                file.write(f">{label}\n")
                file.write(seq.upper() + "\n")
       
            del fasta_dict
            gc.collect()
            for rec in SeqIO.parse(self.control_file, 'fasta'):
                chrom = rec.description.split(':')[1]
                start, end = rec.description.split(':')[2].split()[0].split('-')
                label = f"{chrom}:{start}-{end}"

                file.write(f">{label}\n")
                file.write(str(rec.seq).upper() + "\n")
                       
           
        
if __name__ == "__main__":
    train_fantom_file = "../Data/FANTOM/train.usage.matrix"
    train_pickle_file = "../Data/FANTOM/train_crm.pickle"
    train_triplet_sim_file = "../Data/Datasets/LNR/train_triplet_sim.npy"
    train_triplet_dis_file = "../Data/Datasets/LNR/train_triplet_dis.npy"
    train_sequence_file = "../Data/Datasets/LNR/train_sequences.fa"
    train_control_file = "../Data/Datasets/LNR/train_length_no_repeats.fa"

    
    valid_fantom_file = "../Data/FANTOM/valid.usage.matrix"
    valid_pickle_file = "../Data/FANTOM/valid_crm.pickle"
    valid_triplet_sim_file = "../Data/Datasets/LNR/valid_triplet_sim.npy"
    valid_triplet_dis_file = "../Data/Datasets/LNR/valid_triplet_dis.npy"
    valid_sequence_file = "../Data/Datasets/LNR/valid_sequences.fa"
    valid_control_file = "../Data/Datasets/LNR/valid_length_no_repeats.fa"

    
    test_fantom_file = "../Data/FANTOM/test.usage.matrix"
    test_pickle_file = "../Data/FANTOM/test_crm.pickle"
    test_triplet_sim_file = "../Data/Datasets/LNR/test_triplet_sim.npy"
    test_triplet_dis_file = "../Data/Datasets/LNR/test_triplet_dis.npy"
    test_sequence_file = "../Data/Datasets/LNR/test_sequences.fa"
    test_control_file = "../Data/Datasets/LNR/test_length_no_repeats.fa"

    
    
    fasta_dir = "../Data/HG38/Chromosomes/"
    
    
    ControlDriver(train_fantom_file, train_pickle_file, train_triplet_sim_file, train_triplet_dis_file, fasta_dir, train_sequence_file, train_control_file, 8, 500).run()
    
    ControlDriver(valid_fantom_file, valid_pickle_file, valid_triplet_sim_file, valid_triplet_dis_file, fasta_dir, valid_sequence_file, valid_control_file, 8, 500).run()

    ControlDriver(test_fantom_file, test_pickle_file, test_triplet_sim_file, test_triplet_dis_file, fasta_dir, test_sequence_file, test_control_file, 8, 500).run()

