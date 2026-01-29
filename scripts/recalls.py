import sys
import pickle, os, sys, re
import networkx as nx
import numpy as np
import pandas as pd
import collections
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append("/home/miaocj/docker_dir/kNN-overlap-finder/scripts")

from graph import OverlapGraph,remove_false_edges,get_neighbor_overlap_bases

MAX_SAMPLE_SIZE = int(1e9)
COVERAGE_DEPTH = 20


data = np.load('/home/miaocj/docker_dir/kNN-overlap-finder/data/evaluation64/CHM13/all/filter3_real_new_wy/v53_k11_s15_d1000_n600_neighbors100/FEDRANN_nbr_matrix.npz') 
nbr_indices = data['arr_0']

with open('/home/miaocj/docker_dir/kNN-overlap-finder/data/regional_reads/CHM13/all/filter3_real_new_wy/reference_graph.gpickle','rb') as f:
    reference_graph = pickle.load(f)
print("reference graph loading done")

ref_nbr = {}
for i in range(nbr_indices.shape[0]):
    ref_nbr[i] = {}
    for u, v, attrs in reference_graph.edges(i, data=True):
        ref_nbr[i][v] = attrs['overlap_size']
for k,v in ref_nbr.items():
    my_dict = ref_nbr[k]
    sorted_keys = sorted(my_dict , key=my_dict.get, reverse=True)
    ref_nbr[k] = sorted_keys
    
recalls = []
for k in range(5,80,5):
    all_candidates = set()
    tested_edges = set ()
    finded_pair_num = 0 
    for i, neighbors in ref_nbr.items():
        cutoff = min(len(neighbors),k)
        for j in neighbors[:cutoff]:
            edge = (min(i,j), max(i,j))
            if edge not in all_candidates:
                all_candidates.add(edge)

    for i, neighbors in enumerate(nbr_indices):
        for j in neighbors[:k]:
            if j != -1 and i != j: 
                edge = (min(i,j), max(i,j))
                if edge not in tested_edges:
                    if edge in all_candidates:
                        finded_pair_num += 1
                    tested_edges.add(edge)
    recall = finded_pair_num/len(all_candidates)
    recalls.append(recall)