import sys
import pickle, os, sys, re
import networkx as nx
import numpy as np
import pandas as pd
import collections
sys.path.append("scripts")
sys.path.append("../../scripts")

from graph import OverlapGraph,remove_false_edges,get_neighbor_overlap_bases

MAX_SAMPLE_SIZE = int(1e9)
COVERAGE_DEPTH = 20

ref_graph_path = snakemake.input['ref_graph']
nbr_path = snakemake.input['nbr_indice']
df_file= snakemake.output['integral_stat']
tp_counter_path = snakemake.output['tp_counter']

filename = os.path.abspath(nbr_path)
pattern  = r'data/([^/]+)/([^/]+)/([^/]+)/([^/]+)/([^/]+)/(.+)_nbr_matrix'
thread = re.search(pattern,filename).group(1)
sample = re.search(pattern,filename).group(2)
region = re.search(pattern,filename).group(3)
platform = re.search(pattern,filename).group(4)
encode = re.search(pattern,filename).group(5)
method = re.search(pattern,filename).group(6)
print(thread,sample,region,platform,encode,method)

data = np.load(nbr_path) 
nbr_indices = data['arr_0']

with open(ref_graph_path,'rb') as f:
    reference_graph = pickle.load(f)
print("reference graph loading done")

def calculate_precision(neighbor_matrix, G):
    precisions = []
    for n in [6,12,18]:
        TP, FP = 0, 0
        tested_edges = set()  # 用于记录已测试的边（无向图使用有序对）
        for i, neighbors in enumerate(neighbor_matrix):
            for j in neighbors[:n]:
                edge = (min(i,j), max(i,j))
                if edge not in tested_edges:
                    tested_edges.add(edge)
                    if G.has_edge(i, j):
                        TP += 1
                    else:
                        FP += 1
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        precisions.append(precision)
        print(f"Precision with {n} neighbors: {precision}")
    return precisions

def statistic_fp_numbers_of_reads(neighbor_matrix, G):
    one_read_tp_neighbors_numbers = []
    for i, neighbors in enumerate(neighbor_matrix):
        one_read_tp_neighbors = 0
        for j in neighbors[:20]:
            if G.has_edge(i, j):
                one_read_tp_neighbors += 1
        one_read_tp_neighbors_numbers.append(one_read_tp_neighbors)
    one_read_tp_neighbors_counter = collections.Counter(one_read_tp_neighbors_numbers)
    return one_read_tp_neighbors_counter

precisions = calculate_precision(nbr_indices, reference_graph)
one_read_tp_neighbors_counter = statistic_fp_numbers_of_reads(nbr_indices, reference_graph)

di = {
    'thread':thread,
    'sample':sample,
    'region':region,
    'platform':platform,
    'encode':encode,
    'method':method,
    'n_neighbors':[6,12,18],
    'precision':precisions,
    }
df = pd.DataFrame(di)
df.to_csv(df_file,sep='\t',index=False)
with open(tp_counter_path,'wb') as f:
    pickle.dump(one_read_tp_neighbors_counter, f)
