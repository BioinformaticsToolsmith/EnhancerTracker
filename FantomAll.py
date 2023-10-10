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

class AllDriver(Fantom.Driver):    
    def __init__(self, fantom_file, pickle_file, triplet_sim_file, triplet_dis_file, fasta_dir, sequence_file, control_file, file_length_list, cores, iteration):
        assert os.path.exists(control_file), f'{control_file} does not exist!'
        assert len(file_length_list), len(file_length_list)
        
        self.control_file = control_file
        self.file_length_list = file_length_list
        self.file_count = len(self.file_length_list)
        
        super().__init__(fantom_file, pickle_file, triplet_sim_file, triplet_dis_file, fasta_dir, sequence_file, cores, iteration)
        self.crm_len = len(self.df_clean)
        self.control_len = len(list(SeqIO.parse(control_file, 'fasta'))) 
        
        
        self.file_ranges = []
        s = 0
        for i in range(self.file_count):
            self.file_ranges.append(range(self.crm_len + s, self.crm_len + s + self.file_length_list[i]))
            s += self.file_length_list[i]
            
        
    def make_negative_list(self, positive_array, anchor_sim_array):
        '''
        Creating a negative list from all of the arrays (anchor, positive, and negative) minus the union of both the positive and the anchor arrays.
        
        '''
        r = np.zeros(self.iteration)
        random_range_list = [np.random.choice(self.file_ranges[x], size = self.iteration, replace = False).astype(np.uintc) for x in range(self.file_count)]
        for x, positive_i in enumerate(positive_array):
            random_choice = random.choice(range(self.file_count + 1))
            
            # if random.random() < 0.2:
            if random_choice == 0:
                # Similar instances to the positive
                positive_sim_array = self.crm_list[positive_i]

                # Union of the positive and anchor similar sets
                union_array = np.union1d(anchor_sim_array, positive_sim_array)

                # Set of instances that neither positive or anchor are similar to, i.e., negative
                negative_array = np.setdiff1d(self.all_index_array, union_array)

                assert len(negative_array) > 0, "The negative array has no negatives!" 
                r[x] = np.random.choice(negative_array)
            else:
                r[x] = random_range_list[random_choice - 1][x]

            
        return r
    
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