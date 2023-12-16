import argparse
import subprocess
import os.path
import os


class ConvertSeq:
    def __init__(self, bed_path_dir, dataset, output_dir):
        '''
        This script works on permutations of records.
        Converts a bed file of records to the required format and intersects with the FANTOM5 dataset for the validity of enhancers. 

        bed_path_dir: The directory of bed files that need to be converted.
        dataset: The FANTOM5 dataset to be used for the intersection.
        output_dir: The output directory to be written to.
        '''
        self.bed_path_dir = bed_path_dir
        self.output_dir = output_dir
        self.dataset = dataset

        assert os.path.exists(self.bed_path_dir), f'Bed path directory {self.bed_path_dir} does not exist.'
        assert os.path.isdir(self.bed_path_dir), f'Bed path directory {self.bed_path_dir} is not a directory!'
        assert os.path.exists(self.dataset), f'Dataset {self.dataset} does not exist.'

    def convert(self):
        '''
        Converts the bed file to the required format for intersecting using the FANTOM5 dataset. 
        '''
        region_list = []
        
        # for x in [90]:
        #     for y in [0]:
        #         for z in [200]:
        for x in [60, 70, 80, 90]: 
            for y in [0, 1, 2, 3, 4, 5]:
                # for z in [600]:
                for z in [100, 200, 300, 400, 500, 600]:

                    if not os.path.exists(self.output_dir):
                        os.makedirs(self.output_dir)

                    if not os.path.exists(f'{self.output_dir}/{x}_{y}'):
                        os.makedirs(f'{self.output_dir}/{x}_{y}')
                    
                    assert os.path.exists(f'{self.output_dir}'), f'Output directory {self.output_dir} does not exist.'

                    bed_path = f'{self.bed_path_dir}/{x}_{y}/{z}_enhancers.bed'
                    with open(bed_path, 'r') as bed_file:
                        for line in bed_file:
                            columns = line.strip().split('\t')
                            chr = line.split(':')[0]
                            absolute_start = int(line.strip().split('-')[0].split(':')[1])
            
                            start = absolute_start + int(columns[1])
                            end = absolute_start + int(columns[2])
                            confidence = columns[3]
                            region_list.append(f'{chr}\t{start}\t{end}\t{confidence}')
                    
                    enhancer_region = f'{self.output_dir}/{x}_{y}/{z}_enhancer_region.bed'

                    with open(enhancer_region, 'w') as bed_file:
                        for region in region_list:
                            bed_file.write(f'{region}\n')

                    self.intersect(enhancer_region, x, y, z)
                    region_list.clear()

    def intersect(self, enhancer_region, x, y, z):
        '''
        Intersects the bed file with the FANTOM5 dataset
        '''
        command = [
            'bedtools', 'subtract', '-a', enhancer_region, '-b', self.dataset, '-A'
        ]    

        #result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #assert result.returncode == 0, f'Subprocess failed with return code {result.returncode}.'

        result = subprocess.run(command, capture_output=True, text=True, check=True)
        assert result.returncode == 0, f'Subprocess failed with return code {result.returncode}.'
        # print("\n"+" ".join(command)+"\n")
        # print(result.stdout)
        # print(type(result))
        # exit()
        with open(f'{self.output_dir}/{x}_{y}/{z}_intersected_enhancers.bed', 'w') as bed_file:
            bed_file.write(f'{result.stdout}')

if __name__ == '__main__':
    ####

    # Example run: 
    # python3 generate_overlap_seq_permutations.py Out_12_12_2023 ../Data/FANTOM/F5.hg38.enhancers.bed Permutated_intersected_enhancers

    #### 
    parser = argparse.ArgumentParser(description='Converts a bed file of records to the required format and intersects with the FANTOM5 dataset for the validity of enhancers.')
    parser.add_argument('-b', '--bed_path_dir', help='The directory containing the permutations of records to be converted.')
    parser.add_argument('-d', '--dataset', help='The FANTOM5 dataset to be used for the intersection.')
    parser.add_argument('-o', '--output_dir', help='The output file to be written to.')
    args = parser.parse_args()

    convert = ConvertSeq(args.bed_path_dir, args.dataset, args.output_dir)
    convert.convert()