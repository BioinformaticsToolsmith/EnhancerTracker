# Will run EnhancerTracker to work on NON-PERMUTATIONS.
# Grabs the first two enhancers from a triplet and uses the centered enhancer (third part of triplet) for scanning.

import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from TripletClassifierEnsemble import TripletClassifierEnsemble
from Scanner import Scanner

out_dir = './Out_12_12_2023_100000'

print("Loading the neural networks...")
model_directory = "./Models"
#print("@@@@@ Model count is not 29")
model_count = 29
#model_count = 3
d1, d2, d3 = 4, 600, 3
ensemble = TripletClassifierEnsemble(model_directory, (d1, d2, d3))
ensemble.load_model_info()
ensemble.load_models(model_count)


for x in [90, 80, 70, 60]:
# for x in [90]:
    gnm_path = f"../Data/GNM_12_12_2023_100000/{x}/seq_0.fa"
    chrom_list = [f"{gnm_path}/{x}" for x in os.listdir(gnm_path)] if os.path.isdir(gnm_path) else [gnm_path]
    triplet_path = f"../Data/Triplets_12_12_2023_100000/{x}_triplet_0.fa"
    out_path = f"{out_dir}/{x}_0/"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    scanner = Scanner(ensemble, triplet_path, chrom_list, model_count, is_merit=True, max_enhancer_size=600, batch_size=126) #512

    for z, step_size in zip([100, 200, 300, 400, 500, 600], [50, 100, 100, 100, 100, 100]):
    # for z, step_size in zip([200], [100]):
        region_size = z
        header = z

        scanner.scan(region_size, step_size, output_path=out_path, header_name=header)
