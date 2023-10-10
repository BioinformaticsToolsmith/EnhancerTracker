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

class Driver():    
    def __init__(self, fantom_file, pickle_file, triplet_sim_file, triplet_dis_file, fasta_dir, sequence_file, cores, iteration):
        """
        fantom_file - path to fantom matrix
        pickle_file - path to pickle file; does not have to exist and can be made; if exists, can be laoded
        triplet_sim_file - output of the similar triplet tensor
        triplet_dis_file - output of the dissimilar triplet tensor
        fasta_dir - fasta files for each chromosome
        sequence_file - Output each enhancer in a multi-fasta format file
        """
        self.fantom_file = fantom_file
        self.pickle_file = pickle_file
        self.triplet_sim_file = triplet_sim_file
        self.triplet_dis_file = triplet_dis_file
        self.sequence_file = sequence_file
        self.fasta_dir = fasta_dir
        self.crm_dict = None
        
        assert os.path.exists(self.fantom_file)
        # assert os.path.exists(self.fasta_dir)
        
        self.iteration = iteration
        
        self.cores = cores # Time: 524.9272162914276
                        # Time: 786.3049099445343
        
        
        #nrows = 1000
        self.df_clean = pd.read_csv(self.fantom_file, sep='\t', index_col = 0).astype(np.uintc) # ,nrows=nrows

#         # Drop regions that do not show activity in any of the cell types
#         row_sum_list = df.sum(axis=1)
#         self.df_clean = df[row_sum_list > 0]
#         self.df_clean = self.df_clean.astype(np.uintc)   
        
#         label_list = list(self.df_clean.index)
        
#         print(len(self.df_clean))  
#         keep_list = []
#         for label in label_list:
#             colon = label.find(":")
#             dash = label.find("-")
            
#             start = int(label[colon + 1:dash])
#             end = int(label[dash + 1:])
#             size = end-start
#             if size >= 100 and size <= 1000:
#                 keep_list.append(label)
                   
#         self.df_clean = self.df_clean.loc[keep_list]        
#         print(len(self.df_clean))
        
        # Get row names
        row_names = self.df_clean.index.tolist()

        self.crm_index_table = {}
        for index, a_crm in enumerate(row_names):
            self.crm_index_table[a_crm] = index
            
        self.all_index_array = np.arange(0, len(self.crm_index_table), dtype=np.uintc)
        
        group_by = int(len(row_names) / self.cores) + 1
        self.partition_list = [row_names[ x * group_by : min((x + 1) * group_by, len(row_names))] for x in range(self.cores)]
        
    def make_crm_dict(self):
        '''
        Make a dictionary of CRM objects that will be run using multiprocessing.
        Then, saving it into a file for memory efficiency.
        '''
        time_start = time.time()
        with Pool(processes=self.cores) as p:
            element_list = p.map(self.make_crm_dict_parallel, self.partition_list)
        time_end = time.time()
        print('Time: ', time_end-time_start)
        self.crm_dict = None
        
        self.crm_dict = {}
        for dictionary in element_list:
            self.crm_dict.update(dictionary)
        
        with open(self.pickle_file, 'wb') as f:
            pk.dump(self.crm_dict, f)
            
        self.crm_list = list(self.crm_dict.values())
        del self.crm_dict
        gc.collect()
     
    def make_crm_dict_parallel(self, name_list):
        '''
        Make a dictionary of CRM objects. Each object knows its similar and dissimilar regions.
        Function for multiprocessing. 
        '''
        # A table of the following format: index -> a np array of ushort
        r = {}
        counter = 0
        l = len(name_list)
        for name in name_list:
            sim_crm_mask = np.any(self.df_clean.values[:, None] & self.df_clean.loc[name].values, axis=2).any(axis=1)        
            sim_list = self.df_clean.index[sim_crm_mask].tolist()
            r[self.crm_index_table[name]] = np.array([self.crm_index_table[a_crm] for a_crm in sim_list], dtype=np.uintc)

            if counter % 100 == 0:
                print(f'{counter} out of {l}')
            
            counter += 1
        print(f'{l} out of {l}')

        return r
    
    def read_crm_dict(self):
        '''
        Loads the dictionary of CRM objects.
        '''
        assert os.path.exists(self.pickle_file)
        with open(self.pickle_file, 'rb') as f:
            self.crm_dict = pk.load(f)
            
        self.crm_list = list(self.crm_dict.values())
        del self.crm_dict
        gc.collect()
            
            
    # def get_similar(self, is_less):
    #     def get_similar_replace():
    #         pass
    #     def get_similar_without():
    #         pass
    #     return get_similar_replace if is_less else get_similar_without
    
    def make_negative_list(self, positive_array, anchor_sim_array):
        '''
        Creating a negative list from all of the arrays (anchor, positive, and negative) minus the union of both the positive and the anchor arrays.
        
        '''
        r = np.zeros(self.iteration)

        for x, positive_i in enumerate(positive_array):
            # Similar instances to the positive
            positive_sim_array = self.crm_list[positive_i]
            
            # Union of the positive and anchor similar sets
            union_array = np.union1d(anchor_sim_array, positive_sim_array)

            # Set of instances that neither positive or anchor are similar to, i.e., negative
            negative_array = np.setdiff1d(self.all_index_array, union_array)

            assert len(negative_array) > 0, "The negative array has no negatives!" 
            r[x] = np.random.choice(negative_array)
            
        return r
    
#     def make_similar_list(self, positive_array):
#         similar_array = positive_array.copy()
#         np.random.shuffle(similar_array)
#         for i in range(len(similar_array)):
#             while similar_array[i] == positive_array[i]:
#                 similar_array[i] = np.random.choice(positive_array, size=1, replace = True)
                
#         return similar_array
    
    def make_similar_list(self, anchor, anchor_sim_array, positive_array):
        # similar_array = positive_array.copy()
        # np.random.shuffle(similar_array)
        # for i in range(len(similar_array)):
        #     while similar_array[i] == positive_array[i]:
        #         similar_array[i] = np.random.choice(positive_array, size=1, replace = True)
        
        # Loop through the positive array
        similar_array = np.zeros(positive_array.shape)
        ar = self.df_clean.iloc[anchor].values

        for i,p in enumerate(positive_array):
            positive_sim_array = self.crm_list[p]
            pr = self.df_clean.iloc[p].values

            intersect_array = np.intersect1d(anchor_sim_array, positive_sim_array, assume_unique=True)
            # randomize the intersect array
            np.random.shuffle(intersect_array)


            found = False
            for val in intersect_array:
                sr = self.df_clean.iloc[val].values

                # check for overlapping columns of ar, pr, and sr
                if np.any(ar & pr & sr):
                    similar_array[i] = val
                    found = True
                    break

            assert found, "No similar region found!"
                
        return similar_array
    
    
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
                block[index, 2, :] = self.make_negative_list(block[index, 1, :], anchor_sim_array)
        
        return block
    
    # Time: 2388.9494280815125
    def make_triplet_tensor(self, is_similar):
        '''
        Creating triplet tensors with the use of multiprocessing which will then be used to concatenate them all together to form
        a triplet matrix.
        '''
        assert self.crm_list != None, "crm_list doesn't exist! Call make_crm_dict() or read_crm_dict()!"
        
        start = time.time()
        with Pool(processes=self.cores) as p:
            
            group_by = int(len(self.crm_list) / self.cores) + 1
            i_list = [list( range( x * group_by, min((x + 1) * group_by, len(self.crm_list)) ) ) for x in range(self.cores)]
            anchor_block_list = p.starmap(self.fill_anchor_block, zip(i_list, repeat(is_similar)))
        
        
        triplet_matrix = np.concatenate(anchor_block_list)
        print(time.time() - start)
                
        if is_similar:
            np.save(self.triplet_sim_file, triplet_matrix)
        else:
            np.save(self.triplet_dis_file, triplet_matrix)
            
        return triplet_matrix
    
    
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
            
            
                   
    def run(self, is_write = True, fasta_header = "HG38_", chrom_list = ["chr" + str(x) for x in range(1, 23)] + ["chrX", "chrY"]):
        # self.make_crm_dict()
        self.read_crm_dict()
        self.make_triplet_tensor(True)
        self.make_triplet_tensor(False)
        
        if is_write:
            self.write_sequences(fasta_header, chrom_list)

        