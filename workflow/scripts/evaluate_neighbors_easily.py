import sys
import pickle, os, sys, re
import networkx as nx
import numpy as np
import pandas as pd
import collections
sys.path.append("scripts")
sys.path.append("../../scripts")

from graph import OverlapGraph,remove_false_edges,get_neighbor_overlap_bases

nbr_path = sys.argv[1]
filename = os.path.abspath(nbr_path)
p1 =r'(.+)evaluation[^/]*/(.+)(hash_k\d+\/).+'

str1 = re.search(p1,filename).group(1)
str2 = re.search(p1,filename).group(2)
str3 = re.search(p1,filename).group(3)
tsv_path = str1+'regional_reads/'+str2+'metadata.tsv.gz'
ref_graph_path = str1+'regional_reads/'+str2+'reference_graph.gpickle'
df_file= nbr_path[:-14]+'with_tp_stat.tsv'

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

# def calculate_precision(neighbor_matrix, G):
#     precisions = []
#     for n in [1,2,3,4,6,12,18]:
#         TP, FP = 0, 0
#         tested_edges = set()  # 用于记录已测试的边（无向图使用有序对）
#         for i, neighbors in enumerate(neighbor_matrix):
#             for j in neighbors[1:n+1]:
#                 edge = (min(i,j), max(i,j))
#                 if edge not in tested_edges:
#                     tested_edges.add(edge)
#                     if G.has_edge(i, j):
#                         TP += 1
#                     else:
#                         FP += 1
#         precision = TP / (TP + FP)
#         precisions.append(precision)
#         print(f"Precision with {n} neighbors: {precision}")
#     return precisions

def statistic_fp_numbers_of_reads(neighbor_matrix, G):
    one_read_tp_neighbors_numbers = []
    for i, neighbors in enumerate(neighbor_matrix):
        one_read_tp_neighbors = 0
        for j in neighbors[:20]:
            if G.has_edge(i, j):
                one_read_tp_neighbors += 1
        one_read_tp_neighbors_numbers.append(one_read_tp_neighbors)
    one_read_tp_neighbors_counter = collections.Counter(one_read_tp_neighbors_numbers)
    tp_neighbors_fraction_counter = {k: v/neighbor_matrix.shape[0] for k, v in one_read_tp_neighbors_counter.items()}
    print("TP neighbors fraction counter:", tp_neighbors_fraction_counter)
    return tp_neighbors_fraction_counter
tp_neighbors_fraction_counter = statistic_fp_numbers_of_reads(nbr_indices, reference_graph)

di = {
    'thread':thread,
    'sample':sample,
    'region':region,
    'platform':platform,
    'encode':encode,
    'method':method,
    }
sorted_counter = dict(sorted(tp_neighbors_fraction_counter.items(), key=lambda x: x[0]))
di.update(sorted_counter)
df = pd.DataFrame(di,index=[0])
df.to_csv(df_file,sep='\t',index=False)
