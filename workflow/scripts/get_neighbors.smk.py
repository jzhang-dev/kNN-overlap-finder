
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
    idPAFNearestNeighbors,
    PAFNearestNeighbors,
    MHAPNearestNeighbors,
    MECAT2NearestNeighbors,
    wtdbg2NearestNeighbors
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
parser.add_argument("--threads", type=int, default=64, required=False)

args = parser.parse_args()

if len(args.input) == 3:
    npz_path = args.input[0]
    tsv_path = args.input[1]
    paf_path = args.input[2]
else:
    npz_path = args.input[0]
    tsv_path = args.input[1]
nbr_path = args.output[0]
time_path = args.output[1]

method = args.method

## read ANN method parameters file
ANN_threads_param = {'n_jobs':args.threads}
if args.ann_parameter:
    ANN_parameter_file = args.ann_parameter
    with open(ANN_parameter_file, 'r') as f:
        ANN_parameter = json.load(f)
        ANN_parameter.update(ANN_threads_param)
else:
    ANN_parameter = ANN_threads_param
print(f'ANN Params: {ANN_parameter}')

## read dimension reduction method parameters file
if args.dim_parameter:
    dim_parameter_file = args.dim_parameter
    with open(dim_parameter_file, 'r') as f:
        dim_parameter = json.load(f)
    print(f'Dimension reductiom params: {dim_parameter}')
else:
    dim_parameter = {}

## process SRP multi-threads and batch process
if 'mpSRP' in method:
    dim_parameter.update({'temp_dir':os.path.dirname(npz_path),
                          'batch_size':100_000,
                          'n_jobs':3})

print('start loading feature matrix...')
feature_matrix = sp.sparse.load_npz(npz_path)
print('feature matrix loading done!')

kw = dict(data=feature_matrix)
max_bucket_size = COVERAGE_DEPTH * 1.5
max_n_neighbors = args.n_neighbors
if 'density_base_auto' in dim_parameter:
    real_dim_parameter = {'density':(1/math.sqrt(feature_matrix.shape[1]))*int(dim_parameter['density_base_auto'])}
else:
    real_dim_parameter = dim_parameter

print(method)

if method in ['minimap2','xRead','BLEND','MHAP','MECAT2','wtdbg2']:
    method_class_dict = {'MHAP':MHAPNearestNeighbors,
                     'MECAT2':MECAT2NearestNeighbors,
                     'wtdbg2':wtdbg2NearestNeighbors}
    elapsed_time = {}
    start_time = time.time()
    meta_df = pd.read_table(tsv_path).iloc[:MAX_SAMPLE_SIZE, :].reset_index()
    read_indices = {read_name: read_id for read_id, read_name in meta_df['read_name'].items()}
    if method in ['minimap2','BLEND']:
        neighbor_indices = PAFNearestNeighbors().get_neighbors(
                data=feature_matrix, n_neighbors=max_n_neighbors, paf_path=paf_path, read_indices=read_indices
            )
    elif method == 'xRead':
        neighbor_indices = idPAFNearestNeighbors().get_neighbors(
                data=feature_matrix, n_neighbors=max_n_neighbors, paf_path=paf_path, read_indices=read_indices
            )
    else:
        neighbor_indices = method_class_dict[method]().get_neighbors(
                data=feature_matrix, n_neighbors=max_n_neighbors, paf_path=paf_path, read_indices=read_indices
            )
    elapsed_time['nearest_neighbors'] = time.time() - start_time
elif 'Exact' in method and 'chr1_248M' in tsv_path:
    print('For saving time, extract 1w reads as query reads.')
    config = parse_string_to_config(method,{'sample_query_number':10000},{})
    neighbor_indices, elapsed_time, peak_memory = compute_nearest_neighbors(
        data=feature_matrix,
        config=config,
        n_neighbors=max_n_neighbors,
    )
else:
    config = parse_string_to_config(method,ANN_parameter,real_dim_parameter)
    neighbor_indices, elapsed_time = compute_nearest_neighbors(
        data=feature_matrix,
        config=config,
        n_neighbors=max_n_neighbors,
    )

np.savez(nbr_path, neighbor_indices)
with open(time_path, 'w', encoding='utf-8') as f:
    json.dump(elapsed_time, f, ensure_ascii=False)