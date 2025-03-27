
import pickle, os, gzip, json, sys, itertools
from pathlib import Path
from importlib import reload
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import time

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

time_path = sys.argv[1]
paf_path = sys.argv[2]
npz_path = sys.argv[3]
tsv_path = sys.argv[4]
json_path = sys.argv[5]
nbr_path = sys.argv[6]
method = sys.argv[7]
n_neighbors = sys.argv[8]

meta_df = pd.read_table(tsv_path).iloc[:MAX_SAMPLE_SIZE, :].reset_index()
read_indices = {read_name: read_id for read_id, read_name in meta_df['read_name'].items()}
feature_matrix = sp.sparse.load_npz(npz_path)[meta_df.index, :]

with gzip.open(json_path, "rt") as f:
    read_features = json.load(f)
    read_features = {i: read_features[i] for i in meta_df.index}

feature_weights = {i: 1 for i in range(feature_matrix.shape[1])}

fig, ax = plt.subplots(figsize=(8, 2.5))
ax.hist([len(x) for x in read_features.values()], bins=100)
ax.set_xlim(left=0)
ax.set_xlabel("Number of features per read")
ax.set_ylabel("Number of reads")
ax.grid(color='k', alpha=0.1)

kw = dict(data=feature_matrix)
max_bucket_size = COVERAGE_DEPTH * 1.5
max_n_neighbors = int(n_neighbors)

print(method)
if method == 'Minimap2':
    elapsed_time = {}
    start_time = time.time()
    neighbor_indices = PAFNearestNeighbors().get_neighbors(
            data=feature_matrix, n_neighbors=max_n_neighbors, paf_path=paf_path, read_indices=read_indices
        )
    elapsed_time['nearest_neighbors'] = time.time() - start_time    
else:
    config = parse_string_to_config(method,{})
    neighbor_indices, elapsed_time, peak_memory = compute_nearest_neighbors(
        data=feature_matrix,
        config=config,
        n_neighbors=max_n_neighbors,
        read_features=read_features,
    )
print(neighbor_indices)

np.savez(nbr_path, neighbor_indices)
with open(time_path, 'w', encoding='utf-8') as f:
    json.dump(elapsed_time, f, ensure_ascii=False)