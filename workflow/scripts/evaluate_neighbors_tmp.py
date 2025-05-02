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
sys.path.append("/home/miaocj/docker_dir/kNN-overlap-finder/scripts")

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
tsv_path = str1+'feature_matrix/'+str2+str3+'metadata.tsv.gz'
ref_graph_path = str1+'regional_reads/'+str2+'reference_graph.gpickle'
df_file= nbr_path[:-14]+'integral_stat_filter_10k.tsv'

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

max_n_neighbors=20
df_rows = []
read_ids = np.array(list(meta_df.index))
k_values = np.arange(1, max_n_neighbors)

neighbor_edges_nums = []
precisions = []
mean_overlap_sizes = []

valid_reads = set(meta_df['read_name'])

# 预构建 read_name 到索引的映射（避免重复过滤）
read_to_indices = meta_df.groupby('read_name').indices

ind_10k = list(meta_df[meta_df['read_length'] > 8000].read_id)

print('done')

def filter_get_neighbor_overlap_bases(query_graph: nx.Graph, reference_graph: nx.Graph, ind_5k):
    overlap_size_bases = []
    ind_5k_set = set(ind_5k)  # 转换为集合加速查找
    ref_edges = reference_graph.adj  # 直接访问邻接表（比 edges() 更快）

    for node_0, node_1 in query_graph.edges():
        if node_0 in ind_5k_set or node_1 in ind_5k_set:
            overlap_size = ref_edges.get(node_0, {}).get(node_1, {}).get("overlap_size", 0)
            overlap_size_bases.append(overlap_size)
    return overlap_size_bases

for k in k_values:
    graph = OverlapGraph.from_neighbor_indices(
    neighbor_indices=nbr_indices,
    n_neighbors=k,
    read_ids=read_ids,
    require_mutual_neighbors=False)
    print(f"neighbor{k} graph construct done!")
    neighbor_edges_nums.append(len(graph.edges())/nbr_indices.shape[0])
    overlap_sizes = filter_get_neighbor_overlap_bases(query_graph=graph, reference_graph=reference_graph,ind_5k=ind_10k)
    mean_overlap_sizes.append(sum(overlap_sizes)/len(overlap_sizes))
    precisions.append(1-(overlap_sizes.count(0)/len(overlap_sizes)))
    print(1-(overlap_sizes.count(0)/len(overlap_sizes)))

di = {
    'thread':thread,
    'sample':sample,
    'region':region,
    'platform':platform,
    'encode':encode,
    'method':method,
    'n_neighbors':range(1,max_n_neighbors),
    'edges_num':neighbor_edges_nums,
    'integral_precision':precisions,
    'integral_mean':mean_overlap_sizes
    }
df = pd.DataFrame(di)
# df.to_csv(df_file,sep='\t',index=False)
