import sys
sys.path.append('/opt/conda/lib/python3.12/site-packages')
sys.path.append("/home/miaocj/docker_dir/kNN-overlap-finder/.snakemake/conda/4262b1bf4bf1ffb403c0eb7a42ad5906_/lib/python3.12/site-packages")
sys.path.append("/home/miaocj/docker_dir/kNN-overlap-finder/.snakemake/conda/4506eccf78279d93d0e8a34c035e91c5_/lib/python3.12/site-packages")
sys.path.append("/home/miaocj/docker_dir/kNN-overlap-finder/.snakemake/conda/6bda807e3967eae797c7b1b9eeaee8db_/lib/python3.12/site-packages")
sys.path.append("/home/miaocj/docker_dir/kNN-overlap-finder/.snakemake/conda/c2a47d89d1d34e789fdf782557bb7194_/lib/python3.12/site-packages")
sys.path.append("/home/miaocj/docker_dir/kNN-overlap-finder/.snakemake/conda/c6c5514ada15b890fb27d1e36371554c_/lib/python3.12/site-packages")
sys.path.append("/home/miaocj/docker_dir/kNN-overlap-finder/.snakemake/conda/d964a294c2d0fef56a434c021026281e_/lib/python3.12/site-packages")
sys.path.append("/home/miaocj/docker_dir/kNN-overlap-finder/.snakemake/conda/e1c932db5cd4271709e54d8028824bc9_/lib/python3.12/site-packages")
import resource
resource.setrlimit(resource.RLIMIT_AS, (100000000000, 100000000000)) 
import gzip, json
import time
from Bio import SeqIO
import scipy as sp
from collections import Counter
import numpy as np
import pandas as pd
import pickle, os, gzip, json, sys, itertools
from pathlib import Path
from importlib import reload
from dataclasses import dataclass, field
import collections
import matplotlib.pyplot as plt
import networkx as nx
import  sklearn


sys.path.append("scripts")
sys.path.append("../scripts")
from dim_reduction import SpectralEmbedding, scBiMapEmbedding
from graph import OverlapGraph, GenomicInterval, get_overlap_statistics, remove_false_edges
from truth import get_overlaps
from evaluate import NearestNeighborsConfig, mp_compute_nearest_neighbors
from plots import plot_read_graph, mp_plot_read_graphs, get_graphviz_layout, get_umap_layout
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
)
from dim_get_df import from_nbr_to_evaluation,from_matrix_to_evaluation
MAX_SAMPLE_SIZE = int(1e9)
COVERAGE_DEPTH = 20
max_n_neighbors = 20
npz_path = "/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/human/chr22/ONT_R9/kmer_k16/feature_matrix.npz"
tsv_path = "/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/human/chr22/ONT_R9/kmer_k16/metadata.tsv.gz"
json_path = "/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/human/chr22/ONT_R9/kmer_k16/read_features.json.gz"

meta_df = pd.read_table(tsv_path).iloc[:MAX_SAMPLE_SIZE, :].reset_index()
read_indices = {read_name: read_id for read_id, read_name in meta_df['read_name'].items()}
feature_matrix = sp.sparse.load_npz(npz_path)[meta_df.index, :]

with gzip.open(json_path, "rt") as f:
    read_features = json.load(f)
    read_features = {i: read_features[i] for i in meta_df.index}

start_time = time.time()
mydim = scBiMapEmbedding()
dim_redu = mydim.transform(feature_matrix,n_dimensions=30)
elapsed_time = time.time() - start_time
print("scBiMapEmbedding dimension reduction done")
print("elapsed time :%f"%(elapsed_time))

sp.save_npz('/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/human/chr22/ONT_R9/kmer_k16/scBimap500.npz', dim_redu)
