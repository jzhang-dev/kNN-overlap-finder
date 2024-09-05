import sys
sys.path.append('/opt/conda/lib/python3.12/site-packages')
sys.path.append("/home/miaocj/docker_dir/kNN-overlap-finder/.snakemake/conda/4262b1bf4bf1ffb403c0eb7a42ad5906_/lib/python3.12/site-packages")
sys.path.append("/home/miaocj/docker_dir/kNN-overlap-finder/.snakemake/conda/4506eccf78279d93d0e8a34c035e91c5_/lib/python3.12/site-packages")
sys.path.append("/home/miaocj/docker_dir/kNN-overlap-finder/.snakemake/conda/6bda807e3967eae797c7b1b9eeaee8db_/lib/python3.12/site-packages")
sys.path.append("/home/miaocj/docker_dir/kNN-overlap-finder/.snakemake/conda/c2a47d89d1d34e789fdf782557bb7194_/lib/python3.12/site-packages")
sys.path.append("/home/miaocj/docker_dir/kNN-overlap-finder/.snakemake/conda/c6c5514ada15b890fb27d1e36371554c_/lib/python3.12/site-packages")
sys.path.append("/home/miaocj/docker_dir/kNN-overlap-finder/.snakemake/conda/d964a294c2d0fef56a434c021026281e_/lib/python3.12/site-packages")
sys.path.append("/home/miaocj/docker_dir/kNN-overlap-finder/.snakemake/conda/e1c932db5cd4271709e54d8028824bc9_/lib/python3.12/site-packages")

import gzip, json
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
from nearest_neighbors import (
    ExactNearestNeighbors,
)
def from_matrix_to_evaluation(feature_matrix,read_ids):
    config = NearestNeighborsConfig(
            nearest_neighbors_method=ExactNearestNeighbors,
            description='ExactNearestNeighbors,Euclidean,SpectralEmbedding,500,TF',
            tfidf='None',
            dimension_reduction_method=SpectralEmbedding,
            dimension_reduction_kw=dict(n_dimensions=500),
            nearest_neighbors_kw=dict(metric='euclidean')) 

    configs = [config]
    nbr_dict, time_dict, memory_dict = mp_compute_nearest_neighbors(
        data=feature_matrix,
        configs=configs,
        n_neighbors=max_n_neighbors,
        processes=processes,
    )
    nbr_indices = nbr_dict[0]
    df_rows = []
    for k in k_values:
        graph = OverlapGraph.from_neighbor_indices(
            neighbor_indices=nbr_indices,
            n_neighbors=k,
            read_ids=read_ids,
            require_mutual_neighbors=False,
        )
        graph_stats = get_overlap_statistics(query_graph=graph, reference_graph=reference_graph)
        stats = { "n_neighbors": k}
        stats = {"description":'Exact_Euclidean', "n_neighbors": k,
                    **graph_stats}
        df_rows.append(stats)
    df = pd.DataFrame(df_rows)
    return df, nbr_indices

processes = 8
max_n_neighbors=20
graphs = collections.defaultdict(dict)
k_values = np.arange(2, max_n_neighbors + 1)
def from_nbr_to_evaluation(nbr_indices,read_ids,reference_graph,*,des):
    df_rows = []
    for k in k_values:
        graph = OverlapGraph.from_neighbor_indices(
            neighbor_indices=nbr_indices,
            n_neighbors=k,
            read_ids=read_ids,
            require_mutual_neighbors=False,
        )
        graph_stats = get_overlap_statistics(query_graph=graph, reference_graph=reference_graph)
        stats = { "n_neighbors": k}
        stats = {"description":des, "n_neighbors": k,
                    **graph_stats}
        df_rows.append(stats)
    df = pd.DataFrame(df_rows)
    return df
