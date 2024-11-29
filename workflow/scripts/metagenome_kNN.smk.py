from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from snakemake_stub import *


import gzip, json, collections
from typing import Sequence, Mapping, Collection
from Bio import SeqIO
import scipy.sparse as sp
import numpy as np
import pandas as pd
import sys
import pickle
sys.path.append("scripts")
sys.path.append("/home/miaocj/docker_dir/kNN-overlap-finder/.snakemake/conda/4262b1bf4bf1ffb403c0eb7a42ad5906_/lib/python3.12/site-packages")
sys.path.append("/home/miaocj/docker_dir/kNN-overlap-finder/.snakemake/conda/4506eccf78279d93d0e8a34c035e91c5_/lib/python3.12/site-packages")
sys.path.append("/home/miaocj/docker_dir/kNN-overlap-finder/.snakemake/conda/6bda807e3967eae797c7b1b9eeaee8db_/lib/python3.12/site-packages")
sys.path.append("/home/miaocj/docker_dir/kNN-overlap-finder/.snakemake/conda/c2a47d89d1d34e789fdf782557bb7194_/lib/python3.12/site-packages")
sys.path.append("/home/miaocj/docker_dir/kNN-overlap-finder/.snakemake/conda/c6c5514ada15b890fb27d1e36371554c_/lib/python3.12/site-packages")
sys.path.append("/home/miaocj/docker_dir/kNN-overlap-finder/.snakemake/conda/d964a294c2d0fef56a434c021026281e_/lib/python3.12/site-packages")
sys.path.append("/home/miaocj/docker_dir/kNN-overlap-finder/.snakemake/conda/e1c932db5cd4271709e54d8028824bc9_/lib/python3.12/site-packages")
import mmh3
import sharedmem
from sklearn.neighbors import NearestNeighbors  
import numpy as np  
from meta_nearest_neighbors import ExactNearestNeighbors,HNSW,NearestNeighborsConfig,compute_nearest_neighbors
from metagenome_function import evaluate_meta

output_ref_npz_file = snakemake.input['ref_feature_matrix']
output_que_npz_file = snakemake.input['que_feature_matrix']
ref_read_tax_file = snakemake.input['ref_tax']
que_read_tax_file = snakemake.input['que_tax']
nbr_path = snakemake.output['nbr_indice']
stat_path = snakemake.output['stat']
method = snakemake.wildcards['method']

ref_feature_matrix= sp.load_npz(output_ref_npz_file)
que_feature_matrix = sp.load_npz(output_que_npz_file)

with open('workflow/notebooks/meta_config_dict.pkl', 'rb') as file:  
    config_dict = pickle.load(file)

config = config_dict[method]

neighbor_indices, elapsed_time, peak_memory = compute_nearest_neighbors(
    ref=ref_feature_matrix,
    que=que_feature_matrix,
    config=config,
    n_neighbors=1)

print(neighbor_indices)
np.savez(nbr_path, neighbor_indices)

with gzip.open(ref_read_tax_file, "rt") as f:
    ref_read_tax = json.load(f)
with gzip.open(que_read_tax_file, "rt") as f:
    que_read_tax = json.load(f)

df_rows = []
precision_sep,sensitivity_sep,precision,sensitivity = evaluate_meta(neighbor_indices,ref_read_tax,que_read_tax)
evaluate_dict = dict(
    precision_sep=precision_sep,
    sensitivity_sep=sensitivity_sep,
    precision=precision,
    sensitivity=sensitivity)
df_rows.append(evaluate_dict)
df = pd.DataFrame(df_rows)
df.to_csv(stat_path,sep='\t')
