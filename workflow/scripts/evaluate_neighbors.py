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

from graph import OverlapGraph, GenomicInterval, get_overlap_statistics, remove_false_edges,get_neighbor_overlap_bases,get_precision

MAX_SAMPLE_SIZE = int(1e9)
COVERAGE_DEPTH = 20
nbr_path = sys.argv[1]
filename = os.path.abspath(nbr_path)
if 'kmer_k' in filename:
    p1 =r'(.+)evaluation[^/]*/(.+)(kmer_k\d+\/).+'
elif 'minimizer' in filename:
    p1 =r'(.+)evaluation[^/]*/(.+)(minimizer_k\d+_w\d+\/).+'
str1 = re.search(p1,filename).group(1)
str2 = re.search(p1,filename).group(2)
str3 = re.search(p1,filename).group(3)
tsv_path = str1+'regional_reads/'+str2+'metadata.tsv.gz'
ref_graph_path = str1+'regional_reads/'+str2+'reference_graph.gpickle'
df_file= nbr_path[:-14]+'full_stat.tsv'

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
reference_edges = set(
    tuple(sorted((node_1, node_2)))
    for node_1, node_2 in reference_graph.edges
)
filter_reference_edges = set(
    tuple(sorted((node_1, node_2)))
    for node_1, node_2, data in reference_graph.edges(data=True)
    if data['overlap_size'] >= 500
)

max_n_neighbors=20
df_rows = []
read_ids = np.array(list(meta_df.index))
k_values = np.arange(1, max_n_neighbors)

neighbor_edges_nums = []
precisions = []
mean_overlap_sizes = []
filter_recall = []
recall = []
singleton = []
singleton_percentage = []
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
    print(precisions)
    query_edges = set(
        tuple(sorted((read_1, read_2))) for read_1, read_2 in graph.edges
    )
    true_positive_edges = query_edges & reference_edges
    filter_true_positive_edges = query_edges & filter_reference_edges
    recall.append(len(true_positive_edges)/len(reference_edges))
    filter_recall.append(len(filter_true_positive_edges)/len(filter_reference_edges))

    remove_false_edges(graph, reference_graph)
    singleton.append(len([node for node in graph if len(graph[node]) <= 1]))
    singleton_percentage.append(len([node for node in graph if len(graph[node]) <= 1])/reference_graph.number_of_nodes()) 
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
    'filter_recall':filter_recall,
    'singleton':singleton,
    'singleton_percentage':singleton_percentage,
    'connected_component':connected_component
    }
df = pd.DataFrame(di)
df.to_csv(df_file,sep='\t',index=False)
