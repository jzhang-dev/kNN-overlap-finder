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
from dim_get_df import from_nbr_to_evaluation,from_matrix_to_evaluation

MAX_SAMPLE_SIZE = int(1e9)
COVERAGE_DEPTH = 20
max_n_neighbors = 20
npz_path = "/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/human/IGH/ONT_R9/kmer_k16/feature_matrix.npz"
tsv_path = "/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/human/IGH/ONT_R9//kmer_k16/metadata.tsv.gz"
json_path = "/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/human/IGH/ONT_R9/kmer_k16/read_features.json.gz"

meta_df = pd.read_table(tsv_path).iloc[:MAX_SAMPLE_SIZE, :].reset_index()
read_indices = {read_name: read_id for read_id, read_name in meta_df['read_name'].items()}
feature_matrix = sp.sparse.load_npz(npz_path)[meta_df.index, :]

with gzip.open(json_path, "rt") as f:
    read_features = json.load(f)
    read_features = {i: read_features[i] for i in meta_df.index}

read_ids = np.array(list(read_features))

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
print("complete reference_graph")
nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=20,metric='euclidean')

# ##dimension reduction TruncatedSVD
# start_time = time.time()
# from sklearn.decomposition import TruncatedSVD
# tsvd = TruncatedSVD(n_components=500)
# Trun_dim_mat = tsvd.fit(feature_matrix).transform(feature_matrix) 
# elapsed_time = time.time() - start_time
# print("TruncatedSVD dimension reduction done")
# print("elapsed time :%f"%(elapsed_time))


# nbrs.fit(Trun_dim_mat)
# _, nbr_indices = nbrs.kneighbors(Trun_dim_mat)
# df = from_nbr_to_evaluation(nbr_indices,read_ids,reference_graph,des='TruncatedSVD')
# df.to_csv('/home/miaocj/docker_dir/kNN-overlap-finder/data/evaluation/human/IGH/ONT_R9/kmer_k16/truncatedsvd_dimredu.tsv',sep='\t')

# ## spectrual
# start_time = time.time()
# mydim = SpectralEmbedding()
# dim_redu = mydim.transform(feature_matrix,n_dimensions=500)
# elapsed_time = time.time() - start_time
# print("spectrual dimension reduction done")
# print("elapsed time :%f"%(elapsed_time))

# nbrs.fit(dim_redu)
# _, nbr_indices = nbrs.kneighbors(dim_redu)
# df = from_nbr_to_evaluation(nbr_indices,read_ids,reference_graph,des='spectrual')
# df.to_csv('/home/miaocj/docker_dir/kNN-overlap-finder/data/evaluation/human/IGH/ONT_R9/kmer_k16/spectrual_dimredu.tsv',sep='\t')

# ##scibimap
# start_time = time.time()
# mydim = scBiMapEmbedding()
# dim_redu = mydim.transform(feature_matrix,n_dimensions=500)
# elapsed_time = time.time() - start_time
# print("scBiMapEmbedding dimension reduction done")
# print("elapsed time :%f"%(elapsed_time))

# nbrs.fit(dim_redu)
# _, nbr_indices = nbrs.kneighbors(dim_redu)
# df = from_nbr_to_evaluation(nbr_indices,read_ids,reference_graph,des='scBimap')
# df.to_csv('/home/miaocj/docker_dir/kNN-overlap-finder/data/evaluation/human/IGH/ONT_R9/kmer_k16/scBiMapEmbedding_dimredu.tsv',sep='\t')

##pca
from sklearn.decomposition import PCA
start_time = time.time()
pca = PCA(n_components=500) 
reduced_data = pca.fit_transform(feature_matrix)  
elapsed_time = time.time() - start_time
print("pca dimension reduction done")
print("elapsed time :%f"%(elapsed_time))

nbrs.fit(reduced_data)
_, nbr_indices = nbrs.kneighbors(reduced_data)
df = from_nbr_to_evaluation(nbr_indices,read_ids,reference_graph,des='PCA')
df.to_csv('/home/miaocj/docker_dir/kNN-overlap-finder/data/evaluation/human/IGH/ONT_R9/kmer_k16/pca_dimredu.tsv',sep='\t')

##isomap
from sklearn.manifold import Isomap  
import matplotlib.pyplot as plt  
start_time = time.time()

n_components = 500
isomap = Isomap(n_components=n_components)  
data_projected = isomap.fit_transform(feature_matrix) 
nbrs.fit(data_projected)

elapsed_time = time.time() - start_time
print("isomap dimension reduction done")
print("elapsed time :%f"%(elapsed_time))

_, nbr_indices = nbrs.kneighbors(data_projected)
df = from_nbr_to_evaluation(nbr_indices,read_ids,reference_graph,des='isomap')
df.to_csv('/home/miaocj/docker_dir/kNN-overlap-finder/data/evaluation/human/IGH/ONT_R9/kmer_k16/isomap_dimredu.tsv',sep='\t')

