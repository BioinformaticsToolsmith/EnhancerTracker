#
# Read sequence data 
# Convert a sequence to 2D matrix
#

import numpy as np
from Bio import SeqIO
import os


class FantomToOneHotConverter:
    def __init__(self, a_file, mask_value, max_value):
        assert os.path.exists(a_file), f'The sequence file {a_file} does not exist.'
        
        self.mask_value = mask_value
        self.max_value = max_value
        
        seq_list = [str(x.seq) for x in SeqIO.parse(a_file, "fasta")]

        self.base_num_table = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
        self.seq_matrix = np.zeros((len(seq_list), 4, self.max_value), dtype=np.ubyte)
        
        for i in range(len(seq_list)):
            self.seq_matrix[i, ...] = self.convert_seq_to_one_hot(seq_list[i])  
                               
    def convert_seq_to_one_hot(self, a_seq):
        '''
        Convert a sequence of nucleotides to a matrix where each column is one hot representation of a nucleotide      

        Input: is a DNA sequence
        Output: is a 4-by-max_length matrix
        
        To do: Handle uncertain nucleotides
        '''        
        matrix = self.mask_value * np.ones((4, self.max_value), dtype=np.ubyte)
        for i in range(len(a_seq)):
            if a_seq[i] in self.base_num_table:
                matrix[self.base_num_table[a_seq[i]], i] = 1
                
        assert np.any(matrix), a_seq
                
        return matrix
    
class FantomToOneHotConverterRCAddition:
    def __init__(self, a_file, mask_value, max_value):
        assert os.path.exists(a_file), f'The sequence file {a_file} does not exist.'
        
        self.mask_value = mask_value
        self.max_value = max_value
                
        seq_list = [x.seq for x in SeqIO.parse(a_file, "fasta")]

        self.base_num_table = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
        self.seq_matrix = np.zeros((len(seq_list) * 2, 4, self.max_value), dtype=np.ubyte)
        
        for i in range(0, len(seq_list)):
            self.seq_matrix[i, ...] = self.convert_seq_to_one_hot(str(seq_list[i]))
        for i in range(len(seq_list), len(seq_list) * 2):
            self.seq_matrix[i, ...] = self.convert_seq_to_one_hot(str(seq_list[i - len(seq_list)].reverse_complement()))

            
    def convert_seq_to_one_hot(self, a_seq):
        '''
        Convert a sequence of nucleotides to a matrix where each column is one hot representation of a nucleotide      

        Input: is a DNA sequence
        Output: is a 4-by-max_length matrix
        
        To do: Handle uncertain nucleotides
        '''        
        matrix = self.mask_value * np.ones((4, self.max_value), dtype=np.ubyte)
        for i in range(len(a_seq)):
            if a_seq[i] in self.base_num_table:
                matrix[self.base_num_table[a_seq[i]], i] = 1
                
        assert np.any(matrix), a_seq
                
        return matrix
    

        
        
    
                
class FantomToDecimalConverter:
    def __init__(self, train_file, valid_file, test_file, mask_value):
        assert os.path.exists(train_file), f'Train File {train_file} does not exists.'
        assert os.path.exists(valid_file), f'Valid File {valid_file} does not exists.'
        assert os.path.exists(test_file), f'Test File {test_file} does not exists.'

        # Sequence files
        self.train_file  = train_file
        self.valid_file  = valid_file
        self.test_file   = test_file

        self.mask_value = mask_value
        
        train_list = [str(x.seq) for x in SeqIO.parse(train_file, "fasta")]
        valid_list = [str(x.seq) for x in SeqIO.parse(valid_file, "fasta")]
        test_list = [str(x.seq) for x in SeqIO.parse(test_file, "fasta")]
                
        train_max_len = len(max(train_list, key = lambda x : len(x)))
        valid_max_len = len(max(valid_list, key = lambda x : len(x)))
        test_max_len = len(max(test_list, key = lambda x : len(x)))
        self.max_len = max([train_max_len, valid_max_len, test_max_len])
        
        self.distance_ratio_table = {'A': 0.6, 'C': 0.0, 'G': 1, 'T': 0.225}
        self.reverse_table = {v: k for k, v in self.distance_ratio_table.items()}

        train_len = len(train_list)
        valid_len = len(valid_list)
        test_len = len(test_list)
        
        self.train_x = np.zeros((train_len, 1, self.max_len), dtype=np.float16)
        for i in range(train_len):
            self.train_x[i, ...] = self.convert_seq_to_decimal(train_list[i])  
                       
        self.valid_x = np.zeros((valid_len, 1, self.max_len), dtype=np.float16)
        for i in range(valid_len):
            self.valid_x[i, ...] = self.convert_seq_to_decimal(valid_list[i])  
                       
        self.test_x = np.zeros((test_len, 1, self.max_len), dtype=np.float16)
        for i in range(test_len):
            self.test_x[i, ...] = self.convert_seq_to_decimal(test_list[i]) 
        
            
    def convert_seq_to_decimal(self, a_seq):
        decimal_sequence = -1.0 * np.ones((1, self.max_len), dtype=np.float16)

        for i, nucleotide in enumerate(a_seq):
            if nucleotide in self.distance_ratio_table: #valid_bases
                decimal_sequence[0, i] = self.distance_ratio_table[nucleotide]
                
        return decimal_sequence
        
        
    def convert_seq_to_one_hot(self, a_seq):
        '''
        Convert a sequence of nucleotides to a matrix where each column is one hot representation of a nucleotide      

        Input: is a DNA sequence
        Output: is a 1-by-max_length matrix
        
        To do: Handle uncertain nucleotides
        '''        
        matrix = self.mask_value * np.ones((4, self.max_len))
        for i in range(len(a_seq)):
            if a_seq[i] in self.base_num_table:
                matrix[self.base_num_table[a_seq[i]], i] = 1
                
        return matrix
    
    def split(self):
        return self.train_x, self.valid_x, self.test_x
        

class SeqToMatrixConverter():
    def __init__(self, file_name, mask_value):
        assert os.path.exists(file_name), f'File {file_name} does not exists.'
        
        self.file_name  = file_name
        self.mask_value = mask_value
        
        # Read FASTA file
        rec_list = list(SeqIO.parse(self.file_name, "fasta"))
        
        # Extract sequences and their corresponding labels and 
        # determine the maximum length
        
        l = len(rec_list)
        self.x_list = [None] * l
        self.y = np.zeros((l,1), dtype=int)
        self.max_len = -1
        for i in range(l):
            a_seq = str(rec_list[i].seq)
            self.x_list[i] = a_seq
            self.y[i] = rec_list[i].id.split('_')[1]

            if(len(a_seq) > self.max_len):
                self.max_len = len(a_seq)
                
        self.base_num_table = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        
        self.x = np.zeros((l, 4, self.max_len), dtype=np.ubyte)
        for i in range(l):
            self.x[i, ...] = self.convert_seq_to_one_hot(self.x_list[i])
                
    def convert_seq_to_one_hot(self, a_seq):
        '''
        Convert a sequence of nucleotides to a matrix where each column is one hot representation of a nucleotide      

        Input: is a DNA sequence
        Output: is a 4-by-max_length matrix
        
        To do: Handle uncertain nucleotides
        '''        
        matrix = self.mask_value * np.ones((4, self.max_len), dtype=np.ubyte)
        for i in range(len(a_seq)):
            if a_seq[i] in self.base_num_table:
                matrix[self.base_num_table[a_seq[i]], i] = 1
                
        return matrix
    
    def collect_set_indexes(self, a_label_list):
        '''
        Collect indexes with labels that are in the provided list
        '''
        index_list = []
        for i in range(len(self.y)):
            if self.y[i] in a_label_list:
                index_list.append(i)
        return index_list
    
    def split(self):
        '''
        Divid x and y into three sets with 60%, 20%, and 20% of the original lists.
        '''
        label_list = np.unique(self.y)
        train_limit = int(0.6 * len(label_list))
        valid_limit = int(0.8 * len(label_list))
        
        assert len(label_list[0:train_limit]) >= 2, 'Not enough training labels'
        assert len(label_list[train_limit:valid_limit]) >= 2, 'Not enough validation labels'
        assert len(label_list[valid_limit:]) >= 2, 'Not enough testing labels'
        
        train_index_list = self.collect_set_indexes(label_list[0:train_limit])
        valid_index_list = self.collect_set_indexes(label_list[train_limit:valid_limit])
        test_index_list  = self.collect_set_indexes(label_list[valid_limit:])
        
        train_x = self.x[train_index_list] 
        train_y = self.y[train_index_list]
        valid_x = self.x[valid_index_list]
        valid_y = self.y[valid_index_list]
        test_x  = self.x[test_index_list]
        test_y  = self.y[test_index_list]
        
        return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)