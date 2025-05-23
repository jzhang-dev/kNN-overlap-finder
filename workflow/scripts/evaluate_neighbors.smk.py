import sys
import pickle, os, gzip, json, sys, itertools,re
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
ref_graph_path = snakemake.input['ref_graph']
tsv_path = snakemake.input['metadata']
nbr_path = snakemake.input['nbr_indice']
df_file= snakemake.output['integral_stat']

filename = os.path.abspath(nbr_path)
pattern  = r'data/([^/]+)/([^/]+)/([^/]+)/([^/]+)/([^/]+)/(.+)_nbr_matrix'
thread = re.search(pattern,filename).group(1)
sample = re.search(pattern,filename).group(2)
region = re.search(pattern,filename).group(3)
platform = re.search(pattern,filename).group(4)
encode = re.search(pattern,filename).group(5)
method = re.search(pattern,filename).group(6)
print(thread,sample,region,platform,encode,method)

meta_df = pd.read_table(tsv_path).iloc[:MAX_SAMPLE_SIZE, :].reset_index()
data = np.load(nbr_path) 
nbr_indices = data['arr_0']

with open(ref_graph_path,'rb') as f:
    reference_graph = pickle.load(f)
print("reference graph loading done")
ref_edges_num = reference_graph.number_of_edges()
filter_ref_graph_num =  sum(1 for u, v, data in reference_graph.edges(data=True) if data.get('overlap_size', 0) > 500)

max_n_neighbors=20
df_rows = []
read_ids = np.array(list(meta_df.index))
k_values = np.arange(1, max_n_neighbors)

neighbor_edges_nums = []
precisions = []
mean_overlap_sizes = []
recall = []
singloten = []
connected_component = []

for k in k_values:
    graph = OverlapGraph.from_neighbor_indices(
    neighbor_indices=nbr_indices,
    n_neighbors=k,
    read_ids=read_ids,
    require_mutual_neighbors=False)
    print(f"neighbor{k} graph construct done!")

    neighbor_edges_nums.append(len(graph.edges())/nbr_indices.shape[0])
    overlap_sizes = get_neighbor_overlap_bases(query_graph=graph, reference_graph=reference_graph)
    mean_overlap_sizes.append(sum(overlap_sizes)/len(overlap_sizes))
    precisions.append(1-(overlap_sizes.count(0)/len(overlap_sizes)))

    tf = sum(1 for x in overlap_sizes if x >= 500)
    recall.append(tf/filter_ref_graph_num)
    
    remove_false_edges(graph, reference_graph)
    singloten.append(len([node for node in graph if len(graph[node]) <= 1])) 
    connected_component.append(nx.number_connected_components(graph))


di = {
    'thread':thread,
    'sample':sample,
    'region':region,
    'platform':platform,
    'encode':encode,
    'method':method,
    'n_neighbors':range(1,max_n_neighbors),
    'edges_num':neighbor_edges_nums,
    'precision':precisions,
    'overlap_size':mean_overlap_sizes,
    'recall':recall,
    'singloten':singloten,
    'connected_component':connected_component
    }
df = pd.DataFrame(di)
df.to_csv(df_file,sep='\t',index=False)
