import pickle, os, gzip, json, sys, itertools
from pathlib import Path
from importlib import reload
from dataclasses import dataclass, field
import collections
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pysam
import scipy as sp
import seaborn
import sharedmem


plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["figure.dpi"] = 300


sys.path.append("scripts")
sys.path.append("../../scripts")

from data_io import is_fwd_id, get_fwd_id, get_sibling_id
from dim_reduction import SpectralEmbedding, scBiMapEmbedding
from nearest_neighbors import (
    ExactNearestNeighbors,
    NNDescent,
    WeightedLowHash,
    PAFNearestNeighbors,
    LowHash,
    HNSW,
    ProductQuantization,
    _NearestNeighbors,
    IVFProductQuantization,
    SimHash,
)
from graph import OverlapGraph, GenomicInterval, get_overlap_statistics, remove_false_edges
from truth import get_overlaps
from evaluate import NearestNeighborsConfig, compute_nearest_neighbors

MAX_SAMPLE_SIZE = int(1e9)
COVERAGE_DEPTH = 20

sample = snakemake.wildcards['sample']
dataset = snakemake.wildcards['platform']
region = snakemake.wildcards['region']
method = snakemake.wildcards['method']

if method =="Minimap2":
    paf_path = snakemake.input['paf_minimap2']
elif method == "Blend":
    paf_path = snakemake.input['paf_blend']
else:
    paf_path = ''

npz_path = snakemake.input['feature_matrix']
tsv_path = snakemake.input['metadata']
json_path = snakemake.input['read_features']

nbr_path = snakemake.output['nbr_indice']
threads  = snakemake.threads

print(sample, dataset, region)

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
max_n_neighbors = COVERAGE_DEPTH

with open('workflow/notebooks/config_dict.pkl', 'rb') as file:  
    config_dict = pickle.load(file)

config = config_dict[method]
kw = dict(data=feature_matrix)

neighbor_indices, elapsed_time, peak_memory = compute_nearest_neighbors(
    data=feature_matrix,
    config=config,
    n_neighbors=max_n_neighbors,
    read_features=read_features,
)
print(neighbor_indices)
np.savez(nbr_path, neighbor_indices)