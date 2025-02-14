import sys
sys.path.append("/home/miaocj/docker_dir/kNN-overlap-finder/.snakemake/conda/4262b1bf4bf1ffb403c0eb7a42ad5906_/lib/python3.12/site-packages")
sys.path.append("/home/miaocj/docker_dir/kNN-overlap-finder/.snakemake/conda/4506eccf78279d93d0e8a34c035e91c5_/lib/python3.12/site-packages")
sys.path.append("/home/miaocj/docker_dir/kNN-overlap-finder/.snakemake/conda/6bda807e3967eae797c7b1b9eeaee8db_/lib/python3.12/site-packages")
sys.path.append("/home/miaocj/docker_dir/kNN-overlap-finder/.snakemake/conda/c2a47d89d1d34e789fdf782557bb7194_/lib/python3.12/site-packages")
sys.path.append("/home/miaocj/docker_dir/kNN-overlap-finder/.snakemake/conda/c6c5514ada15b890fb27d1e36371554c_/lib/python3.12/site-packages")
sys.path.append("/home/miaocj/docker_dir/kNN-overlap-finder/.snakemake/conda/d964a294c2d0fef56a434c021026281e_/lib/python3.12/site-packages")
sys.path.append("/home/miaocj/docker_dir/kNN-overlap-finder/.snakemake/conda/e1c932db5cd4271709e54d8028824bc9_/lib/python3.12/site-packages")
import pickle, os, gzip, json, sys, itertools
from pathlib import Path
from importlib import reload
from dataclasses import dataclass, field
import collections
import networkx as nx
import numpy as np
import pandas as pd
import pysam
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import sharedmem

sys.path.append("scripts")
sys.path.append("../../scripts")

from data_io import is_fwd_id, get_fwd_id, get_sibling_id
from graph import OverlapGraph, GenomicInterval, get_overlap_statistics, remove_false_edges,get_neighbor_overlap_bases,get_precision
from truth import get_overlaps
from evaluate import NearestNeighborsConfig, mp_compute_nearest_neighbors

MAX_SAMPLE_SIZE = int(1e9)
COVERAGE_DEPTH = 20

sample = snakemake.wildcards['sample']
dataset = snakemake.wildcards['platform']
region = snakemake.wildcards['region']
method = snakemake.wildcards['method']

npz_path = snakemake.input['feature_matrix']
tsv_path = snakemake.input['metadata']
json_path = snakemake.input['read_features']
nbr_path = snakemake.input['nbr_indice']

stat_path = snakemake.output['overlap']
overlap_sizes_file= snakemake.output['neighbor_overlap_sizes']
threads  = snakemake.threads

print(sample, dataset, region)

meta_df = pd.read_table(tsv_path).iloc[:MAX_SAMPLE_SIZE, :].reset_index()
read_indices = {read_name: read_id for read_id, read_name in meta_df['read_name'].items()}
feature_matrix = sp.sparse.load_npz(npz_path)[meta_df.index, :]

with gzip.open(json_path, "rt") as f:
    read_features = json.load(f)
    read_features = {i: read_features[i] for i in meta_df.index}
    
data = np.load(nbr_path) 
nbr_indices = data['arr_0']

def get_read_intervals(meta_df):
    read_intervals = {
        i: [GenomicInterval(strand, start, end)]
        for i, strand, start, end in zip(
            meta_df.index,
            meta_df["reference_strand"],
            meta_df["reference_start"],
            meta_df["reference_end"],
        )
    }
    return read_intervals

read_intervals = get_read_intervals(meta_df)

reference_graph = OverlapGraph.from_intervals(read_intervals)
nr_edges = set((node_1, node_2) for node_1, node_2, data in reference_graph.edges(data=True) if not data['redundant'])
connected_component_count = len(list(nx.connected_components(reference_graph)))
len(reference_graph.nodes), len(reference_graph.edges), len(nr_edges), connected_component_count

max_n_neighbors=20
df_rows = []
read_ids = np.array(list(read_features))
k_values = np.arange(2, max_n_neighbors + 1)

for k in k_values:
    graph = OverlapGraph.from_neighbor_indices(
    neighbor_indices=nbr_indices,
    n_neighbors=k,
    read_ids=read_ids,
    require_mutual_neighbors=False,)
    graph_stats = get_overlap_statistics(query_graph=graph, reference_graph=reference_graph)
    graph_per_rank =  OverlapGraph.from_neighbor_indices(
    neighbor_indices=nbr_indices[:,k-1],
    n_neighbors=1,
    read_ids=read_ids,
    require_mutual_neighbors=False,)
    graph_stats_per_rank = get_precision(query_graph=graph_per_rank, reference_graph=reference_graph)
    stats = {"description": method, "n_neighbors": k,
                **graph_stats_per_rank,
                **graph_stats}
    df_rows.append(stats)

neighbor_overlap_sizes = get_neighbor_overlap_bases(graph,reference_graph,k_values[max_n_neighbors-2])
with open(overlap_sizes_file, 'wb') as f:
    pickle.dump(neighbor_overlap_sizes, f)

# data = []
# for i, sublist in enumerate(neighbor_overlap_sizes):
#     for value in sublist:
#         data.append({'Group': f'{i+1}', 'Value': value})
# df = pd.DataFrame(data)
# plt.figure(figsize=(8, 6),dpi=300)  # 设置图像大小
# sns.boxplot(x='Group', y='Value', data=df)
# plt.xlabel('Neighbor Rank')
# plt.ylabel('Overlap Size')
# plt.show()

df = pd.DataFrame(df_rows)
print(df)
df['connected_fraction'] = 1 - df['singleton_fraction']
df.to_csv(stat_path,sep='\t')