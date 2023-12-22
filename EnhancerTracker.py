import os
import warnings
import argparse
import tensorflow as tf
from TripletClassifierEnsemble import TripletClassifierEnsemble
from Scanner import Scanner

def parse_arguments():
    parser = argparse.ArgumentParser(description="DNA Sequence Analysis Script")
    parser.add_argument("enhancer_file", help="Path to the FASTA file containing enhancers.")
    parser.add_argument("region_file", help="Path to the FASTA file containing the region to be scanned.")
    parser.add_argument("output_file", help="Path to the output file.")
    return parser.parse_args()

if __name__ == "__main__":
    #
    # Example run: python3 EnhancerTracker.py ../Data/90_triplet_0.fa ../Data/seq_0.fa ../Data/Results
    #
    print("Loading the neural networks...")
    model_directory = "./Models_29"
    model_count = 29
    d1, d2, d3 = 4, 600, 3

    args = parse_arguments()

    enhancer_file_path = args.enhancer_file
    region_file_path = args.region_file
    output_file_path = args.output_file

    chrom_list = [f"{region_file_path}/" for x in os.listdir(region_file_path)] if os.path.isdir(region_file_path) else [region_file_path]

    assert os.path.exists(enhancer_file_path), "Enhancer file does not exist."
    assert os.path.exists(region_file_path), "Region file does not exist."

    assert enhancer_file_path.endswith(".fa"), "Enhancer file must be a FASTA file."
    assert region_file_path.endswith(".fa"), "Region file must be a FASTA file."

    ensemble = TripletClassifierEnsemble(model_directory, (d1, d2, d3))
    ensemble.load_model_info()
    ensemble.load_models(model_count)

    scanner = Scanner(ensemble, enhancer_file_path, chrom_list, model_count, is_merit=True, max_enhancer_size=600, batch_size=126) #512

    for x, step_size in zip([600], [100]):
        region_size = x
        header = x

        scanner.scan(region_size, step_size, output_path=output_file_path, header_name=header)
