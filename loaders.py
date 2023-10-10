from tensorflow import keras
import numpy as np
import os
import random

random.seed(12)
                       
#######################################################################
# @@@@@@@@@@@@@@@@@@@@ PreMade Triplet Classifier @@@@@@@@@@@@@@@@@@@@@
#######################################################################

class PremadeTripletClassifierSequence(keras.utils.Sequence):
    def __init__(self, x_in, triplet_sim_file, triplet_dis_file, batch_size=5, reverse_x_in = None, shuffle=True):
        '''
        A dataset for triplets: anchor, positive, and negative
        '''
        assert os.path.exists(triplet_sim_file), f'This similar triplet file {triplet_sim_file} does not exist.'
        assert os.path.exists(triplet_dis_file), f'This dissimilar triplet file {triplet_dis_file} does not exist.'

        # Initialization
        self.batch_size  = batch_size
        self.x = x_in
        self.triplet_sim_file = triplet_sim_file
        self.triplet_dis_file = triplet_dis_file
        self.reverse_x = reverse_x_in
        self.shuffle = True
        

        
        triplet_sim_tensor = np.load(triplet_sim_file)
        triplet_dis_tensor = np.load(triplet_dis_file)
        sim_label_array = np.ones(len(triplet_sim_tensor))
        dis_label_array = np.zeros(len(triplet_dis_tensor))
        
        self.triplet_tensor = np.concatenate((triplet_sim_tensor, triplet_dis_tensor))
        #self.original_label_array = np.concatenate((sim_label_array, dis_label_array))
        self.label_array = np.concatenate((sim_label_array, dis_label_array))
        
        # permutation = np.random.permutation(len(self.triplet_tensor))
        # self.triplet_tensor = self.triplet_tensor[permutation]
        # self.label_array = self.label_array[permutation]
        
        #input(f'{self.triplet_tensor[:5]} ----> {self.label_array[:5]}')

        self.datalen = len(self.triplet_tensor)
        
        assert self.datalen > 0, 'Invalid triplet tensor of size 0.'
        
        _, _, self.channel = self.triplet_tensor.shape
        
        
        self.epoch_num = np.random.choice(list(range(0, self.channel)))
        if self.shuffle:
            self.set_matrix_and_labels()
        
    def set_matrix_and_labels(self):
        # self.matrix = np.copy(self.triplet_tensor[:, :, self.epoch_num])
        # permutation = np.random.permutation(len(self.matrix))
        # self.matrix = self.matrix[permutation]
        # self.label_array = np.copy(self.original_label_array)[permutation]
        
        permutation = np.random.permutation(len(self.triplet_tensor))
        self.triplet_tensor = self.triplet_tensor[permutation]
        self.label_array = self.label_array[permutation]
        self.matrix = self.triplet_tensor[:, : , self.epoch_num]
        
        self.epoch_num = (self.epoch_num + 1) % self.channel
        
    def __getitem__(self, index):    
        # Determine batch start and end
        batch_start = index*self.batch_size
        batch_end   = (index+1)*self.batch_size
        if batch_end > self.datalen:
            batch_end = self.datalen
        batch_size = batch_end - batch_start
            
        # Allocate tensors to hold x and y for the batch
        _, num_row, num_col = self.x.shape
        x_tensor = np.zeros((batch_size, num_row, num_col, 3))
        y_tensor = np.zeros((batch_size, 1))
        
        # Collect images into the tensors    
        x_tensor[:, :, :, 0] = self.x[self.matrix[batch_start:batch_end, 0], :, :]
        x_tensor[:, :, :, 1] = self.x[self.matrix[batch_start:batch_end, 1], :, :]
        x_tensor[:, :, :, 2] = self.x[self.matrix[batch_start:batch_end, 2], :, :]
        y_tensor[:] = np.expand_dims(self.label_array[batch_start:batch_end], axis = 1)
        

        if self.reverse_x is not None:
            for i in range(batch_size):
                for j in range(3):
                    if random.random() < 0.5:
                        x_tensor[i, :, :, j] = self.reverse_x[self.matrix[batch_start+i, j], :, :]
            
            
        #permutation = np.random.permutation(batch_size)

        
        #return x_tensor[permutation], y_tensor[permutation]
        return x_tensor, y_tensor


        
        
    def __len__(self):
        # Denotes the number of batches per epoch
        return self.datalen // self.batch_size

    def on_epoch_end(self):
        if self.shuffle: 
            self.set_matrix_and_labels()

#############################################################################################################################################
#########@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@     Premade Pair Sequence     @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@###############
#############################################################################################################################################
        
class PremadePairSequence(keras.utils.Sequence):
    def __init__(self, x_in, triplet_file, batch_size=5, reverse_x_in = None):
        '''
        A dataset for triplets: anchor, positive, and negative
        '''
        assert os.path.exists(triplet_file), f'This triplet file {triplet_file} does not exist.'
        # Initialization
        self.batch_size  = batch_size
        self.x = x_in
        self.reverse_x = reverse_x_in
        
        self.triplet_tensor = np.load(triplet_file)
        self.datalen = len(self.triplet_tensor)
        
        assert self.datalen > 0, 'Invalid triplet tensor of size 0.'
        
        _, _, self.channel = self.triplet_tensor.shape
        
        self.epoch_num = np.random.choice(list(range(0, self.channel)))
                
        self.set_matrix()
   
    def set_matrix(self):        
        permutation = np.random.permutation(len(self.triplet_tensor))
        self.triplet_tensor = self.triplet_tensor[permutation]
        self.matrix = self.triplet_tensor[:, : , self.epoch_num]
        
        self.epoch_num = (self.epoch_num + 1) % self.channel

    #
    #
    # Start here
    #
    #
    def __getitem__(self, index):    
        # Determine batch start and end
        batch_start = index*self.batch_size
        batch_end   = (index+1)*self.batch_size
        if batch_end > self.datalen:
            batch_end = self.datalen
        batch_size = batch_end - batch_start
        batch_half = batch_size // 2
     
            
        # Allocate tensors to hold x and y for the batch
        _, num_row, num_col = self.x.shape
        x_tensor = np.zeros((batch_size, num_row, num_col, 2))
        y_tensor = np.zeros((batch_size, 1))
        
        # Collect images into the tensors    
        # x_tensor[:, :, :, 0] = self.x[self.matrix[batch_start:batch_end, 0], :, :]
        # x_tensor[:, :, :, 1] = self.x[self.matrix[batch_start:batch_end, 1], :, :]
        # y_tensor[:] = np.expand_dims(self.label_array[batch_start:batch_end], axis = 1)
        
        x_tensor[:batch_half, :, :, 0] = self.x[self.matrix[batch_start:(batch_start+batch_half), 0], :, :]
        x_tensor[:batch_half, :, :, 1] = self.x[self.matrix[batch_start:(batch_start+batch_half), 1], :, :]
        
        x_tensor[batch_half:, :, :, 0] = self.x[self.matrix[(batch_start+batch_half):batch_end, 0], :, :]
        x_tensor[batch_half:, :, :, 1] = self.x[self.matrix[(batch_start+batch_half):batch_end, 2], :, :]
        
        y_tensor[:batch_half] = 1
        
        if self.reverse_x is not None:
            for i in range(batch_size):
                for j in range(2):
                    if random.random() < 0.5:
                        x_tensor[i, :, :, j] = self.reverse_x[self.matrix[batch_start+i, j], :, :]
    
        rand_perm = np.random.permutation(batch_size)
        
        return x_tensor[rand_perm], y_tensor[rand_perm]
    
    def reverse_complement(self, x):
        # cut_x = len(x) - np.argmax(np.any(x[::-1, :, :, :], axis=(1, 2, 3))) - 1
        # cut_x = x[:last_nonzero_index_i + 1, :, :, :]
        sum_array = np.sum(x, axis=0)
        x_slice = x[:, :(np.max(np.nonzero(sum_array)) + 1)]
        x_slice = x_slice[:, ::-1]
        
        nb = np.argmax(x_slice, axis=0)
        nb = self.nb_complement_array[nb]
        
        r = np.zeros((4, len(nb)))
        r[nb, range(len(nb))] = 1
        col_count = 1000 - x_slice.shape[1]
        r = np.pad(r, ((0, 0), (0, col_count)), 'constant')
        
        return r
    
    def __len__(self):
        # Denotes the number of batches per epoch
        return self.datalen // self.batch_size

    def on_epoch_end(self):
        # Grab new triplet indexes at the end of each batch per epoch        
        # self.epoch_num = (self.epoch_num + 1) % self.channel
        # self.matrix = self.triplet_tensor[:, : , self.epoch_num]
        # np.random.shuffle(self.matrix)
        self.set_matrix()
