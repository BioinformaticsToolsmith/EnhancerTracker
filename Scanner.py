import numpy as np
from Bio import SeqIO
import sys
from Enhancer import Enhancer
import tensorflow as tf
import os

def has_overlap(region_1, region_2):
    '''
    Check if two regions overlap
    region_1: the first region
    region_2: the second region
    '''

    return min(region_1.get_end(), region_2.get_end()) - max(region_1.get_start(), region_2.get_start()) > 0

def find_best_enhancer(l):
    '''
    Find the enhancer with the highest confidence score
    l: the list of enhancers to search
    '''

    best_enhancer = None
    for enhancer in l:
        if best_enhancer is None or enhancer.get_confidence() > best_enhancer.get_confidence():
            best_enhancer = enhancer

    return best_enhancer

class Scanner:
    '''
    Scans a chromosome for enhancers
    '''

    def __init__(self, ensemble, enhancer_path, chrom_list, model_count = 29, is_merit = True, max_enhancer_size = 600, batch_size = 512):
        '''
        ensemble: the ensemble of neural networks to use; must be loaded
        enhancer_path: the path to the enhancer file to use
        chrom_list: the list of chromosomes to scan
        model_count: the number of models to use
        is_merit: whether to use the average weighted voting
        max_enhancer_size: the maximum size of the enhancer
        batch_size: the batch size to use
        '''

        self.nucleotide_table = {'A':'A', 'C':'C', 'G':'G', 'T':'T', 
						'N':'N', 'R':'G', 'Y':'C', 'M':'A', 
						'K':'T', 'S':'G', 'W':'T', 'H':'C', 
						'B':'T', 'V':'A', 'D':'T'}
        self.base_num_table = {'A':0, 'C':1, 'G':2, 'T':3}


        self.ensemble = ensemble
        self.max_enhancer_size = max_enhancer_size
        self.enhancer_path = enhancer_path
        self.chrom_list = chrom_list
        self.model_count = model_count
        self.is_merit = is_merit
        self.batch_size = batch_size

        # The maximum size of a segment; based on the batch_size to not overload the memory
        self.max_segment_size = self.batch_size * 200

        self.region_size = -1
        self.step_size = -1

        try:
            ensemble.model_list[0].model
        except:
            raise Exception("The given ensemble is not loaded!")

        assert len(self.chrom_list) > 0, "No chromosomes to scan"
        assert self.max_enhancer_size > 0, "The maximum enhancer size must be greater than 0"
        assert self.batch_size > 0, "The batch size must be greater than 0"
        assert self.model_count > 0, "The model count must be greater than 0"
        assert self.max_segment_size > 0, "The maximum segment size must be greater than 0"
        assert os.path.exists(self.enhancer_path), f"The given enhancer file does not exist: {self.enhancer_path}"
        assert os.path.isfile(self.enhancer_path), f"The given enhancer file is not a file: {self.enhancer_path}"

        for chrom in self.chrom_list:
            assert os.path.exists(chrom), f"The given chromosome file does not exist: {chrom}"
            assert os.path.isfile(chrom), f"The given chromosome file is not a file: {chrom}"
        
        # Load the enhancer pair that will be used for comparison
        self.enhancer_one, self.enhancer_two = self.load_enhancer_pair()

    

    def load_enhancer_pair(self):
        '''
        Load the enhancer pair from the given file and convert into one-hot encoding
        '''

        pair_list = []
        for record in SeqIO.parse(self.enhancer_path, 'fasta'):
            pair_list.append(str(record.seq))

        assert len(pair_list) >= 1, f"The given enhancer file does not contain at least one sequence!"
        if len(pair_list) > 2:
            print("Warning: the given enhancer file contains more than two sequences. Only the first two will be used.")
        
        return self.convert_to_one_hot(pair_list[0], self.max_enhancer_size), self.convert_to_one_hot(pair_list[1], self.max_enhancer_size)

    def convert_to_one_hot(self, sequence, max_size):
        '''
        Convert a sequence to one-hot encoding

        sequence: the sequence to convert
        max_size: the maximum size of the sequence

        returns the one-hot encoding of the sequence
        '''


        sequence = sequence.upper()
        sequence = sequence.translate(str.maketrans(self.nucleotide_table))

        one_hot = np.zeros((4, max_size), dtype=np.ubyte)
        for i in range(min(max_size, len(sequence))):
            one_hot[self.base_num_table[sequence[i]], i] = 1

        return one_hot
    
    def predict_region(self, triplet_matrix):
        '''
        Send to neural network for predicting
        triplet_matrix: the matrix of triplets to send to the neural network

        returns the prediction and confidence
        '''

        # prediction and confidence
        p, c = self.ensemble.predict(triplet_matrix, use_count = self.model_count, verbose = 0, is_loaded = True, is_merit = self.is_merit) 

        return p, c

    def scan(self, region_size, step_size, output_path = None, header_name = None):
        '''
        Scans a chromosome for enhancers
        Sends to output file if provided
        Otherwise, sends to terminal

        region_size: the size of the region to scan
        step_size: the step size between regions
        output_path: the path to the output directory
        header_name: the name of the header to use for the output files
        '''


        self.region_size = region_size
        self.step_size = step_size
        for chrom in self.chrom_list:
            print("Processing", chrom, "...")

            # Read the chromosome
            rec = SeqIO.read(chrom, 'fasta')
            sequence = str(rec.seq) # the sequence of the chromosome
            id = rec.id # the ID of the chromosome
            self.chrom_size = len(sequence)
            assert self.region_size < self.chrom_size, f"The chromosome at {chrom} is too small! The region size is {self.region_size} but the chromosome size is {self.chrom_size}"

            # Creating segments
            segment_list = self.create_segments(sequence)

            # Scanning the chromosome
            region_list = self.scan_segments(segment_list, sequence, id)

            # Post-processing
            enhancer_list = self.post_process(region_list)

            # Write to output
            if output_path is not None:
                self.write_to_output(output_path, header_name, id, segment_list, region_list, enhancer_list)
            else:
                for enhancer in enhancer_list:
                    print(str(enhancer))



    def create_segments(self, sequence):
        '''
        Create segments from the given sequence that are separated by blocks of Ns or are split up from too-large regions.

        sequence: the sequence to create segments from

        returns a list of segments
        '''

        # will contain segments from the chromosome that are separated by blocks of Ns
        segment_list = []
        segment_start = 0
        segment_end = 0

        # If first group of nucleotides are N's, then skip
        while segment_start < self.chrom_size and sequence[segment_start] == 'N':
            segment_start += 1

        i = segment_start

        # Creating segments
        while i < self.chrom_size:
            # If the current base is an N, then we have reached the end of a segment
            if sequence[i] == 'N':
                segment_end = i          

                segment_size = segment_end - segment_start
                if segment_size > self.max_segment_size:
                    subsegment_start = segment_start

                    while subsegment_start < segment_end:
                        # End the subsegment at the start plus max_segment_size
                        # But not beyond the original segment_end
                        subsegment_end = min(subsegment_start + self.max_segment_size, segment_end)

                        # If we are not at the end, we need to subtract the overlap from the subsegment end
                        if subsegment_end != segment_end:
                            subsegment_end -= self.region_size - self.step_size  # Overlap size

                        # Add the subsegment to the segment list
                        segment_list.append((subsegment_start, subsegment_end))

                        # Start the next subsegment where the last one ended, ensuring the overlap
                        subsegment_start = subsegment_end
                else:
                    segment_list.append((segment_start, segment_end))

                # Skip over all Ns
                while i < self.chrom_size and sequence[i] == 'N':
                    i += 1
                
                segment_start = i

            i += 1

        # Add the last segment
        if sequence[-1] != 'N':
            segment_list.append((segment_start, self.chrom_size))

        # Remove segments that are too small
        segment_list = [x for x in segment_list if x[1] - x[0] >= self.region_size]

        return segment_list
    
    def scan_segments(self, segment_list, sequence, id):
        '''
        Scan the segments for enhancers

        segment_list: the list of segments to scan
        sequence: the sequence of the chromosome
        id: the ID of the chromosome

        returns a list of regions that may potentially be enhancers
        '''


        region_list = []

        # For progress bar
        cumulative = 0
        total_length = 0
        for segment in segment_list:
            total_length += segment[1] - segment[0]

        print(f"Number of segments: {len(segment_list)}")
        # for segment in segment_list:
        #     print(segment[0], segment[1])
        for s_i, segment in enumerate(segment_list):

            # Get the one-hot encoding of the segment
            segment_start, segment_end = segment
            segment_one_hot = self.convert_to_one_hot(sequence[segment_start:segment_end], segment_end - segment_start)

            # Number of regions in the segment; depending on stride, may contain overlapping regions
            region_groups = ((segment_end - segment_start - self.region_size) // self.step_size) + 1

            # Create a matrix of triplets
            segment_matrix = np.zeros((region_groups, 4, self.max_enhancer_size, 3), dtype=np.ubyte)
            for i in range(region_groups):

                region_one_hot = np.zeros((4, self.max_enhancer_size), dtype=np.ubyte)
                region_one_hot[:, :self.region_size] = segment_one_hot[:, i * self.step_size:i * self.step_size + self.region_size]

                segment_matrix[i, :, :, :] = np.dstack((self.enhancer_one, self.enhancer_two, region_one_hot))

            # Send batches of batch size to the neural network
            for i in range(0, region_groups, self.batch_size):
                percentage = (cumulative / total_length) * 100
                sys.stdout.write("\rCompleted: {:.2f}%, {}/{}".format(percentage, s_i, len(segment_list)))
                sys.stdout.flush()
                s = tf.convert_to_tensor(segment_matrix[i:i + self.batch_size])
                cumulative += len(s) * self.step_size

                prediction, confidence = self.predict_region(s)

                # Append to the list the regions that have a 1 in the prediction, i.e., the regions that are predicted to be enhancers
                for j in range(len(prediction)):
                    if prediction[j] == 1:
                        region_start = segment_start + (i * self.step_size) + (j * self.step_size)
                        region_end = region_start + self.region_size
                        c = float("{:.2f}".format(confidence[j]))
                        region_list.append(Enhancer(id, region_start, region_end, c))
                    

        sys.stdout.write("\rCompleted: 100.00%, " + str(len(segment_list)) + "/" + str(len(segment_list)) + "\n")
        sys.stdout.flush()

        return region_list
        
    def post_process(self, region_list):
        '''
        Post-process the regions to find the best enhancers from overlapping regions

        region_list: the list of regions to post-process

        returns a list of enhancers
        '''


        # Find the overlapping regions and store them in a list, do not store them if they are not overlapping. 
        enhancer_list = []
        i = 1
        if len(region_list) > 0:
            chain_list = [region_list[0]]
            for i in range(1, len(region_list)):

                # Write percentage
                percentage = (i / len(region_list))
                sys.stdout.write("\rCompleted: {:.2f}%".format(percentage * 100))
                sys.stdout.flush()

                if has_overlap(chain_list[-1], region_list[i]):
                    chain_list.append(region_list[i])
                else:
                    enhancer_list.append(find_best_enhancer(chain_list))
                    chain_list = [region_list[i]]
        else:
            chain_list = []



        # Add the last enhancer
        if len(chain_list) > 0:
            enhancer_list.append(find_best_enhancer(chain_list))
                
        sys.stdout.write("\rCompleted: 100.00%\n")
        sys.stdout.flush()

        return enhancer_list
    
    def write_to_output(self, output_dir, header_name, id, segment_list, region_list, enhancer_list):
        '''
        Write the results to the output directory

        output_dir: the path to the output directory
        header_name: the name of the header to use for the output files
        id: the ID of the chromosome
        segment_list: the list of segments
        region_list: the list of regions
        enhancer_list: the list of enhancers
        '''

        print("Writing segments to file...")
        with open(f"{output_dir}/{header_name}_segments.bed", 'w') as f:
            for segment in segment_list:
                f.write(f"{id}\t{segment[0]}\t{segment[1]}\n")

        print("Writing overlapping raw confidence scores to file...")
        with open(f"{output_dir}/{header_name}_overlapping.bed", 'w') as f:
            for enhancer in region_list:
                f.write(enhancer.to_bed(with_confidence = True) + "\n")

        print("Writing enhancers to file...")
        with open(f"{output_dir}/{header_name}_enhancers.bed", 'w') as f:
            for enhancer in enhancer_list:
                f.write(enhancer.to_bed(with_confidence = True) + "\n")


