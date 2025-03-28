
import pickle, os, gzip, json, sys, itertools
from pathlib import Path
from importlib import reload
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import time
import argparse,math

sys.path.append("scripts")
sys.path.append("../../scripts")
from str2config import parse_string_to_config
from nearest_neighbors import (
    ExactNearestNeighbors,
    PAFNearestNeighbors
)
from evaluate import NearestNeighborsConfig, compute_nearest_neighbors

MAX_SAMPLE_SIZE = int(1e9)
COVERAGE_DEPTH = 20

parser = argparse.ArgumentParser(description="Get neighbors using different method and parameters")
parser.add_argument("--input",  nargs='+', required=True, 
                   help="muitiple input files")
parser.add_argument("--output", nargs='+', required=True, 
                   help="muitiple output files")
parser.add_argument("--method", type=str, required=True,
                   help="method for fing neighbors")
parser.add_argument("--ann-parameter", type=str, required=False)
parser.add_argument("--dim-parameter", type=str, required=False)
parser.add_argument("--n-neighbors", type=int, default=20, required=False)

args = parser.parse_args()

npz_path = args.input[0]
json_path = args.input[1]
tsv_path = args.input[2]
paf_path = args.input[3]

nbr_path = args.output[0]
time_path = args.output[1]

method = args.method

if args.ann_parameter:
    ANN_parameter_file = args.ann_parameter
    with open(ANN_parameter_file, 'r') as f:
        ANN_parameter = json.load(f)
    print(f'ANN Params: {ANN_parameter}')
else:
    ANN_parameter = {}

if args.dim_parameter:
    dim_parameter_file = args.dim_parameter
    with open(dim_parameter_file, 'r') as f:
        dim_parameter = json.load(f)
    print(f'Dimension reductiom params: {dim_parameter}')
else:
    dim_parameter = {}
    
meta_df = pd.read_table(tsv_path).iloc[:MAX_SAMPLE_SIZE, :].reset_index()
read_indices = {read_name: read_id for read_id, read_name in meta_df['read_name'].items()}
feature_matrix = sp.sparse.load_npz(npz_path)[meta_df.index, :]

with gzip.open(json_path, "rt") as f:
    read_features = json.load(f)
    read_features = {i: read_features[i] for i in meta_df.index}

feature_weights = {i: 1 for i in range(feature_matrix.shape[1])}

kw = dict(data=feature_matrix)
max_bucket_size = COVERAGE_DEPTH * 1.5
max_n_neighbors = args.n_neighbors
if 'density_base_auto' in dim_parameter:
    real_dim_parameter = {'density':(1/math.sqrt(feature_matrix.shape[1]))*int(dim_parameter['density_base_auto'])}
else:
    real_dim_parameter = dim_parameter

print(method)
if method == 'Minimap2':
    elapsed_time = {}
    start_time = time.time()
    neighbor_indices = PAFNearestNeighbors().get_neighbors(
            data=feature_matrix, n_neighbors=max_n_neighbors, paf_path=paf_path, read_indices=read_indices
        )
    elapsed_time['nearest_neighbors'] = time.time() - start_time
elif 'Exact' in method and 'chr1_248M' in tsv_path:
    print('For saving time, extract 1w reads as query reads.')
    config = parse_string_to_config(method,{'sample_query_number':10000})
    neighbor_indices, elapsed_time, peak_memory = compute_nearest_neighbors(
        data=feature_matrix,
        config=config,
        n_neighbors=max_n_neighbors,
        read_features=read_features,
    )
else:
    config = parse_string_to_config(method,ANN_parameter,real_dim_parameter)
    neighbor_indices, elapsed_time, peak_memory = compute_nearest_neighbors(
        data=feature_matrix,
        config=config,
        n_neighbors=max_n_neighbors,
        read_features=read_features,
    )

np.savez(nbr_path, neighbor_indices)
with open(time_path, 'w', encoding='utf-8') as f:
    json.dump(elapsed_time, f, ensure_ascii=False)