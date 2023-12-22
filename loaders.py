from tensorflow import keras
import numpy as np
import os
import random

random.seed(12)

class TripletSequence(keras.utils.Sequence):
    '''
    The skeleton code of the sequence is based on code from: https://stackoverflow.com/questions/70230687/how-keras-utils-sequence-works
    '''
    def __init__(self, x_in, y_in, batch_size=5, can_shuffle=True, output='x'):
        '''
        A dataset for triplets: anchor, positive, and negative
        '''
        # Initialization
        self.batch_size  = batch_size
        self.can_shuffle = can_shuffle
        self.x = x_in
        self.y = y_in
        self.label_list = np.unique(self.y)
        self.output = output
        assert self.output in ['x', 'y', 'label'], f'Invalid output type: {self.output}, valid output are x, y, and label.'
        
        self.datalen = len(y_in)
        self.indexes = np.arange(self.datalen)
        if self.can_shuffle:
            np.random.shuffle(self.indexes)
            
        self.index_table = {}
        self.make_pstv_and_ngtv_indexes()
        self.make_triplet_matrix()
    
    def make_pstv_and_ngtv_indexes(self):
        '''
        Make two lists: (1) a list of label indexes and () a list of all other labels
        '''
        for label in self.label_list:
            assert label in self.y, f'Label {label} is not a valid class.'
            pstv_array = np.where(self.y == label)[0]
            ngtv_array = np.where(self.y != label)[0]

            self.index_table[label] = (pstv_array, ngtv_array)
    
    def make_triplet_indexes(self, label):
        '''
        Return three index arrays per a label: (1) the anchor indexes, (2) the positive indexes, snf (3) the negative indexes.
        '''
        assert label in self.label_list, f'Label {label} is not a valid class.'
        
        pstv_array, ngtv_array = self.index_table[label]
        
        a_array = np.copy(pstv_array)
        np.random.shuffle(a_array)
        
        p_array = np.copy(pstv_array)
        np.random.shuffle(p_array)
        
        n_array = np.copy(ngtv_array)
        np.random.shuffle(n_array)
        n_array = n_array[0:len(pstv_array)]
        
        assert len(a_array) == len(p_array), f'The anchor and the positive arrays must have the same length.'
        assert len(p_array) == len(n_array), f'The negative and the positive arrays must have the same length: {len(p_array)} {len(n_array)}.'
        
        assert self.y[a_array[0]] == label, 'The anchor must have the desired label.'
        assert self.y[p_array[0]] == label, 'The positive must have the desired label.'
        assert self.y[n_array[0]] != label, 'The negative must not have the desired label.'
        
        return a_array, p_array, n_array
    
    def make_triplet_matrix(self):
        '''
        Make a matrix where its first column is the anchor indexes, the second column is the positive  indexes, and
        the third column is the negative indexes. 
        This matrix is shuffled.
        '''
        self.matrix = np.ones((self.datalen, 3), dtype=int) * -1
        
        next_start = 0
        for a_label in self.label_list:
            a_array, p_array, n_array = self.make_triplet_indexes(a_label)
            next_end = next_start + len(a_array)
            self.matrix[next_start:next_end, 0] = a_array
            self.matrix[next_start:next_end, 1] = p_array
            self.matrix[next_start:next_end, 2] = n_array
            next_start = next_end
            
        np.random.shuffle(self.matrix)
        
        assert len(np.where(self.matrix == -1)[0]) == 0, 'Something wrong with the triplet matrix.'
            
            
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
        
        # Collect images into the tensors    
        x_tensor[:, :, :, 0] = self.x[self.matrix[batch_start:batch_end, 0], :, :]
        x_tensor[:, :, :, 1] = self.x[self.matrix[batch_start:batch_end, 1], :, :]
        x_tensor[:, :, :, 2] = self.x[self.matrix[batch_start:batch_end, 2], :, :]
            
        if self.output == 'x': 
            return x_tensor, x_tensor
        elif self.output == 'y':
            y_tensor = np.zeros((batch_size, 1))
            return x_tensor, y_tensor
        else:
            # Collect labels into the tensors
            y_tensor = np.zeros((batch_size, 2))
            y_tensor[:, 0] = np.squeeze(self.y[self.matrix[batch_start:batch_end, 0]]) # The anchor and the positive labels
            y_tensor[:, 1] = np.squeeze(self.y[self.matrix[batch_start:batch_end, 2]]) # The negative labels
            return x_tensor, y_tensor
            
    def __len__(self):
        # Denotes the number of batches per epoch
        return self.datalen // self.batch_size

    def on_epoch_end(self):
        # Make new triplet indexes at the end of each epoch
        if self.can_shuffle:
            self.make_triplet_matrix()
            
            
##################################################################################################
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ PREMADETRIPLETSEQUENCE @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
##################################################################################################
            
class PremadeTripletSequence(keras.utils.Sequence):
    def __init__(self, x_in, triplet_file, batch_size=5):
        '''
        A dataset for triplets: anchor, positive, and negative
        '''        
        assert os.path.exists(triplet_file), f'This triplet file {triplet_file} does not exist.'
        # Initialization
        self.batch_size  = batch_size
        self.x = x_in
        
        self.triplet_tensor = np.load(triplet_file)
        self.datalen = len(self.triplet_tensor)
        
        assert self.datalen > 0, 'Invalid triplet tensor of size 0.'
        
        _, _, self.channel = self.triplet_tensor.shape
        
        self.epoch_num = np.random.choice(list(range(0, self.channel)))
        self.matrix = self.triplet_tensor[:, :, self.epoch_num]
        
        # np.random.shuffle(self.matrix)
                
    def __getitem__(self, index):    
        # Determine batch start and end
        batch_start = index*self.batch_size
        batch_end   = (index+1)*self.batch_size
        if batch_end > self.datalen:
            batch_end = self.datalen
        batch_size = batch_end - batch_start
            
        # Allocate tensors to hold x and y for the batch
        _, num_row, num_col = self.x.shape
        x_tensor = np.zeros((batch_size, num_row, num_col, 3), dtype=np.ubyte)
        
        # Collect images into the tensors    
        x_tensor[:, :, :, 0] = self.x[self.matrix[batch_start:batch_end, 0], :, :]
        x_tensor[:, :, :, 1] = self.x[self.matrix[batch_start:batch_end, 1], :, :]
        x_tensor[:, :, :, 2] = self.x[self.matrix[batch_start:batch_end, 2], :, :]

        return x_tensor, x_tensor

    def __len__(self):
        # Denotes the number of batches per epoch
        return self.datalen // self.batch_size

    def on_epoch_end(self):
        # Grab new triplet indexes at the end of each batch per epoch        
        self.epoch_num = (self.epoch_num + 1) % self.channel
        self.matrix = self.triplet_tensor[:, : , self.epoch_num]
        # np.random.shuffle(self.matrix)
                
#######################################################################
# @@@@@@@@@@@@@@@@@@@@ PreMade Triplet Classifier @@@@@@@@@@@@@@@@@@@@@
#######################################################################

class PremadeTripletClassifierSequence(keras.utils.Sequence):
    def __init__(self, x_in, triplet_sim_file, triplet_dis_file, batch_size=5, reverse_x_in = None):
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
    
#     def reverse_complement(self, x):
#         # cut_x = len(x) - np.argmax(np.any(x[::-1, :, :, :], axis=(1, 2, 3))) - 1
#         # cut_x = x[:last_nonzero_index_i + 1, :, :, :]
#         sum_array = np.sum(x, axis=0)
#         x_slice = x[:, :(np.max(np.nonzero(sum_array)) + 1)]
#         x_slice = x_slice[:, ::-1]
        
#         nb = np.argmax(x_slice, axis=0)
#         nb = self.nb_complement_array[nb]
        
#         r = np.zeros((4, len(nb)))
#         r[nb, range(len(nb))] = 1
        
#         return r

        
        
    def __len__(self):
        # Denotes the number of batches per epoch
        return self.datalen // self.batch_size

    def on_epoch_end(self):
        self.set_matrix_and_labels()
        #         # Grab new triplet indexes at the end of each batch per epoch        
        #         self.epoch_num = (self.epoch_num + 1) % self.channel
        #         permutation = np.random.permutation(len(self.triplet_tensor))
        #         self.triplet_tensor = self.triplet_tensor[permutation]
        #         self.label_array = self.label_array[permutation]
        #         self.matrix = self.triplet_tensor[:, : , self.epoch_num]

        #         permutation = np.random.permutation(len(self.triplet_tensor))
        #         self.matrix = self.matrix[permutation]
        #         self.label_array = self.label_array[permutation]        

#         def __getitem__(self, index):    
#         # Determine batch start and end
#         batch_start = index*self.batch_size
#         batch_end   = (index+1)*self.batch_size
#         if batch_end > self.datalen:
#             batch_end = self.datalen
#         batch_size = batch_end - batch_start
#         batch_half = batch_size // 2
#         batch_mid  = batch_start + batch_half
            
#         # Allocate tensors to hold x and y for the batch
#         _, num_row, num_col = self.x.shape
#         x_tensor = np.zeros((batch_size, num_row, num_col, 3))
        
#         # Collect images into the tensors    
#         x_tensor[:, :, :, 0] = self.x[self.matrix[batch_start:batch_end, 0], :, :]
#         x_tensor[:, :, :, 1] = self.x[self.matrix[batch_start:btch_end, 1], :, :]
#         x_tensor[:batch_half, :, :, 2] = self.x[self.matrix[batch_start:batch_mid, 2], :, :]
#         x_tensor[batch_half:, :, :, 2] = self.x[self.matrix[batch_mid:batch_end, 3], :, :]
        
#         # Collect labels into the tensors
#         y_tensor = np.zeros((batch_size, 1))
#         y_tensor[batch_half:] = 1
        
#         rand_perm = np.random.permutation(batch_size)
        
#         return x_tensor[rand_perm], y_tensor[rand_perm]

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
    
    
class PairSequence(keras.utils.Sequence):
    '''
    The skeleton code of the sequence is based on code from: https://stackoverflow.com/questions/70230687/how-keras-utils-sequence-works
    '''
    def __init__(self, x_in, y_in, samples_per_label=1, batch_size=5, can_shuffle=True, output="x"):
        '''
        samples_per_label: the total number of triplet is samples per label (positive) * (the number of labels-1) * samples per label
                           if the number of samples per label is 2 and the number of labels is 33 is we are assembling 2 * 32 * 2 triplets.
        '''
        # Initialization
        self.batch_size  = batch_size
        self.can_shuffle = can_shuffle
        self.x = x_in
        self.y = y_in
        self.samples_per_label = samples_per_label
        self.label_list = np.unique(self.y)
        assert output in ['x', 'y', 'xy'], f'The output must be x, y, or xy: recevied {output}.'
        self.output = output
    
        self.datalen = len(y_in)
        self.indexes = np.arange(self.datalen)
        if self.can_shuffle:
            np.random.shuffle(self.indexes)
            
        self.index_table = {}
        
        self.make_pstv_and_ngtv_indexes()
        self.make_pair_matrix()
    
    def make_pstv_and_ngtv_indexes(self):
        '''
        Make two lists: (1) a list of label indexes and () a list of all other labels
        '''
        for label in self.label_list:
            assert label in self.y, f'Label {label} is not a valid class.'
            pstv_array = np.where(self.y == label)[0]
            ngtv_array = np.where(self.y != label)[0]

            self.index_table[label] = (pstv_array, ngtv_array)
    
    def make_triplet_indexes(self, label):
        '''
        Return three index arrays per a label: (1) the anchor indexes, (2) the positive indexes, snf (3) the negative indexes.
        '''
        assert label in self.label_list, f'Label {label} is not a valid class.'
        
        pstv_array, ngtv_array = self.index_table[label]
        
        a_array = np.copy(pstv_array)
        np.random.shuffle(a_array)
        
        p_array = np.copy(pstv_array)
        np.random.shuffle(p_array)
        
        n_array = np.copy(ngtv_array)
        np.random.shuffle(n_array)
        n_array = n_array[0:len(pstv_array)]
        
        assert len(a_array) == len(p_array), f'The anchor and the positive arrays must have the same length.'
        assert len(p_array) == len(n_array), f'The negative and the positive arrays must have the same length.'
        
        assert self.y[a_array[0]] == label, 'The anchor must have the desired label.'
        assert self.y[p_array[0]] == label, 'The positive must have the desired label.'
        assert self.y[n_array[0]] != label, 'The negative must not have the desired label.'
        
        return a_array, p_array, n_array
    
    def make_pair_matrix(self):
        '''
        Make a matrix where its first column is the anchor indexes, the second column is the positive  indexes,
        or the negative indexes.
        Make the corresponding label vector: 1 means a similar pair and 0 means a dissimilar pair. 
        This matrix is shuffled.
        '''
        self.matrix      = np.ones((2 * self.datalen, 2), dtype=int) * -1
        self.pair_labels = np.ones((2 * self.datalen, 3), dtype=int) * -1
        
        next_start = 0
        for a_label in self.label_list:
            a_array, p_array, n_array = self.make_triplet_indexes(a_label)
    
            next_end = next_start + len(a_array)
            
            self.matrix[next_start:next_end, 0] = a_array
            self.matrix[next_start:next_end, 1] = p_array
            self.pair_labels[next_start:next_end, 0] = np.squeeze(self.y[a_array])
            self.pair_labels[next_start:next_end, 1] = np.squeeze(self.y[p_array])
            self.pair_labels[next_start:next_end, 2] = 1
            assert np.array_equal(self.pair_labels[next_start:next_end, 0], self.pair_labels[next_start:next_end, 1])
            
            next_start = next_end
            next_end = next_start + len(a_array)
            
            self.matrix[next_start:next_end, 0] = a_array
            self.matrix[next_start:next_end, 1] = n_array
            self.pair_labels[next_start:next_end, 0] = np.squeeze(self.y[a_array])
            self.pair_labels[next_start:next_end, 1] = np.squeeze(self.y[n_array])
            self.pair_labels[next_start:next_end, 2] = 0
            assert not np.array_equal(self.pair_labels[next_start:next_end, 0], self.pair_labels[next_start:next_end, 1])
            
            next_start = next_end
            
        rand_perm = np.random.permutation(len(self.matrix))  
        self.matrix = self.matrix[rand_perm, ...]
        self.pair_labels = self.pair_labels[rand_perm]
                
        assert len(np.where(self.matrix == -1)[0]) == 0, 'Something wrong with the pair matrix.'
        assert len(np.where(self.pair_labels == -1)[0]) == 0, 'Something wrong with the pair labels.'   
            
    def __getitem__(self, index):    
        # Determine batch start and end
        batch_start = index*self.batch_size
        batch_end   = (index+1)*self.batch_size
        if batch_end > len(self.matrix):
            batch_end = len(self.matrix)
        batch_size = batch_end - batch_start
            
        # Allocate tensors to hold x and y for the batch
        _, num_row, num_col = self.x.shape
        x_tensor = np.zeros((batch_size, num_row, num_col, 2))
        
        # Collect images into the tensors    
        x_tensor[:, :, :, 0] = self.x[self.matrix[batch_start:batch_end, 0], :, :]
        x_tensor[:, :, :, 1] = self.x[self.matrix[batch_start:batch_end, 1], :, :]
            
        if self.output == 'x': 
            return x_tensor, x_tensor
        elif self.output == 'y':
            # Collect labels into the tensors
            y_tensor = np.zeros((batch_size, 2))
            y_tensor = self.pair_labels[batch_start:batch_end]

            return x_tensor, y_tensor[:,2]
        elif self.output == 'xy':
            # Collect labels into the tensors
            y_tensor = np.zeros((batch_size, 2))
            y_tensor = self.pair_labels[batch_start:batch_end]

            return x_tensor, {'recon': x_tensor, 'mean-var': np.zeros(len(x_tensor)), 'distance': y_tensor[:,2]}
            
        else:
            raise RuntimeError('Unexpected output format.')

    def __len__(self):
        # Denotes the number of batches per epoch
        return len(self.matrix) // self.batch_size

    def on_epoch_end(self):
        # Make new triplet indexes at the end of each epoch
        if self.can_shuffle:
            self.make_pair_matrix()
            
            
class SingleSequence(keras.utils.Sequence):
    '''
    The skeleton code of the sequence is based on code from: https://stackoverflow.com/questions/70230687/how-keras-utils-sequence-works
    A dataset (or a sequence) for single elements (not pairs not triplets).
    '''
    def __init__(self, x_in, y_in, batch_size=32, can_shuffle=True, is_generator=True):
        assert len(x_in) == len(y_in), f'The length of x does not match that of y: {len(x_in)} {len(y_in)}'
        # Initialization
        self.x = x_in
        self.y = y_in
        self.batch_size  = batch_size
        self.can_shuffle = can_shuffle
        self.is_generator = is_generator
                
        #self.label_list = np.unique(self.y)
        self.datalen = len(y_in)
        self.indexes = np.arange(self.datalen)
        if self.can_shuffle:
            np.random.shuffle(self.indexes)
            
        # self.index_table = {}
        # self.make_pstv_and_ngtv_indexes()
        # self.make_triplet_matrix()
    
#     def make_pstv_and_ngtv_indexes(self):
#         '''
#         Make two lists: (1) a list of label indexes and () a list of all other labels
#         '''
#         for label in self.label_list:
#             assert label in self.y, f'Label {label} is not a valid class.'
#             pstv_array = np.where(self.y == label)[0]
#             ngtv_array = np.where(self.y != label)[0]

#             self.index_table[label] = (pstv_array, ngtv_array)
    
#     def make_triplet_indexes(self, label):
#         '''
#         Return three index arrays per a label: (1) the anchor indexes, (2) the positive indexes, snf (3) the negative indexes.
#         '''
#         assert label in self.label_list, f'Label {label} is not a valid class.'
        
#         pstv_array, ngtv_array = self.index_table[label]
        
#         a_array = np.copy(pstv_array)
#         np.random.shuffle(a_array)
        
#         p_array = np.copy(pstv_array)
#         np.random.shuffle(p_array)
        
#         n_array = np.copy(ngtv_array)
#         np.random.shuffle(n_array)
#         n_array = n_array[0:len(pstv_array)]
        
#         assert len(a_array) == len(p_array), f'The anchor and the positive arrays must have the same length.'
#         assert len(p_array) == len(n_array), f'The negative and the positive arrays must have the same length: {len(p_array)} {len(n_array)}.'
        
#         assert self.y[a_array[0]] == label, 'The anchor must have the desired label.'
#         assert self.y[p_array[0]] == label, 'The positive must have the desired label.'
#         assert self.y[n_array[0]] != label, 'The negative must not have the desired label.'
        
#         return a_array, p_array, n_array
    
#     def make_triplet_matrix(self):
#         '''
#         Make a matrix where its first column is the anchor indexes, the second column is the positive  indexes, and
#         the third column is the negative indexes. 
#         This matrix is shuffled.
#         '''
#         self.matrix = np.ones((self.datalen, 3), dtype=int) * -1
        
#         next_start = 0
#         for a_label in self.label_list:
#             a_array, p_array, n_array = self.make_triplet_indexes(a_label)
#             next_end = next_start + len(a_array)
#             self.matrix[next_start:next_end, 0] = a_array
#             self.matrix[next_start:next_end, 1] = p_array
#             self.matrix[next_start:next_end, 2] = n_array
#             next_start = next_end
            
#         np.random.shuffle(self.matrix)
        
#         assert len(np.where(self.matrix == -1)[0]) == 0, 'Something wrong with the triplet matrix.'
     
    def __getitem__(self, index):    
        # Determine batch start and end
        batch_start = index*self.batch_size
        batch_end   = (index+1)*self.batch_size
        if batch_end > self.datalen:
            batch_end = self.datalen
        batch_size = batch_end - batch_start
            
        # Allocate tensors to hold x and y for the batch
        _, num_row, num_col = self.x.shape
        x_tensor = np.zeros((batch_size, num_row, num_col, 1))
        
        # Collect images into the tensors    
        x_tensor[:, :, :, 0] = self.x[self.indexes[batch_start:batch_end], :, :]
            
        if self.is_generator: 
            return x_tensor, x_tensor
        else:
            # Collect labels into the tensors
            y_tensor = np.zeros((batch_size, 1))
            y_tensor[:, 0] = np.squeeze(self.y[self.indexes[batch_start:batch_end]])
            return x_tensor, y_tensor
            
    def __len__(self):
        # Denotes the number of batches per epoch
        return self.datalen // self.batch_size

    def on_epoch_end(self):
        # Make new triplet indexes at the end of each epoch
        if self.can_shuffle:
            np.random.shuffle(self.indexes)

#
# To do: Test a batch and the outputs
#
class TripletClassifierSequence(keras.utils.Sequence):
    '''
    The skeleton code of the sequence is based on code from: https://stackoverflow.com/questions/70230687/how-keras-utils-sequence-works
    Make similar (anchor, positive, another positive) and dissmilar triplets (anchor, positive, negative)
    '''
    def __init__(self, x_in, y_in, batch_size=1024):
        '''
        '''
        # Initialization
        self.batch_size  = batch_size
        self.x = x_in
        self.y = y_in

        self.label_list = np.unique(self.y)

        self.datalen = len(y_in)
        self.indexes = np.arange(self.datalen)
        np.random.shuffle(self.indexes)
            
        self.index_table = {}
        self.make_pstv_and_ngtv_indexes()
        self.make_triplet_matrix()
    
    def make_pstv_and_ngtv_indexes(self):
        '''
        Make two lists: (1) a list of label indexes and () a list of all other labels
        '''
        for label in self.label_list:
            assert label in self.y, f'Label {label} is not a valid class.'
            pstv_array = np.where(self.y == label)[0]
            ngtv_array = np.where(self.y != label)[0]

            self.index_table[label] = (pstv_array, ngtv_array)
    
    def make_triplet_indexes(self, label):
        '''
        Return four index arrays per a label: (1) the anchor indexes, (2) the positive indexes, (3) the negative indexes, and (4) another array of positive indexes.
        '''
        assert label in self.label_list, f'Label {label} is not a valid class.'
        
        pstv_array, ngtv_array = self.index_table[label]
        
        a_array = np.copy(pstv_array)
        np.random.shuffle(a_array)
        
        p_array = np.copy(pstv_array)
        np.random.shuffle(p_array)
        
        n_array = np.copy(ngtv_array)
        np.random.shuffle(n_array)
        n_array = n_array[0:len(pstv_array)]
        
        p_2_array = np.copy(pstv_array)
        np.random.shuffle(p_2_array)
        
        assert len(a_array) == len(p_array), f'The anchor and the positive arrays must have the same length: {len(a_array)} {len(p_array)}. '
        assert len(p_array) == len(n_array), f'The negative and the positive arrays must have the same length: {len(p_array)} {len(n_array)}.'
        assert len(n_array) == len(p_2_array), f'The negative and the second positive arrays must have the same length: {len(n_array)} {len(p_2_array)}.'
        
        assert self.y[a_array[0]] == label,   'The anchor must have the desired label.'
        assert self.y[p_array[0]] == label,   'The positive must have the desired label.'
        assert self.y[n_array[0]] != label,   'The negative must not have the desired label.'
        assert self.y[p_2_array[0]] == label, 'The positive must have the desired label.'
        
        return a_array, p_array, n_array, p_2_array
    
    def make_triplet_matrix(self):
        '''
        Make a matrix where its first column is the anchor indexes, the second column is the positive  indexes, and
        the third column is the negative indexes. 
        This matrix is shuffled.
        '''
        self.matrix = np.ones((self.datalen, 4), dtype=int) * -1
    
        next_start = 0
        for a_label in self.label_list:
            a_array, p_array, n_array, p_2_array = self.make_triplet_indexes(a_label)
            next_end = next_start + len(a_array)
            self.matrix[next_start:next_end, 0] = a_array
            self.matrix[next_start:next_end, 1] = p_array
            self.matrix[next_start:next_end, 2] = n_array
            self.matrix[next_start:next_end, 3] = p_2_array
            next_start = next_end
            
        np.random.shuffle(self.matrix)

        assert len(np.where(self.matrix == -1)[0]) == 0, 'Something wrong with the triplet matrix.'
            
            
    def __getitem__(self, index):    
        # Determine batch start and end
        batch_start = index*self.batch_size
        batch_end   = (index+1)*self.batch_size
        if batch_end > self.datalen:
            batch_end = self.datalen
        batch_size = batch_end - batch_start
        batch_half = batch_size // 2
        batch_mid  = batch_start + batch_half
            
        # Allocate tensors to hold x and y for the batch
        _, num_row, num_col = self.x.shape
        x_tensor = np.zeros((batch_size, num_row, num_col, 3))
        
        # Collect images into the tensors    
        x_tensor[:, :, :, 0] = self.x[self.matrix[batch_start:batch_end, 0], :, :]
        x_tensor[:, :, :, 1] = self.x[self.matrix[batch_start:batch_end, 1], :, :]
        x_tensor[:batch_half, :, :, 2] = self.x[self.matrix[batch_start:batch_mid, 2], :, :]
        x_tensor[batch_half:, :, :, 2] = self.x[self.matrix[batch_mid:batch_end, 3], :, :]
        
        # Collect labels into the tensors
        y_tensor = np.zeros((batch_size, 1))
        y_tensor[batch_half:] = 1
        
        rand_perm = np.random.permutation(batch_size)
        
        return x_tensor[rand_perm], y_tensor[rand_perm]
            
    def __len__(self):
        # Denotes the number of batches per epoch
        return self.datalen // self.batch_size

    def on_epoch_end(self):
        # Make new triplet indexes at the end of each epoch
        self.make_triplet_matrix()
        
