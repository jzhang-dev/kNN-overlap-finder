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
from sklearn.decomposition import TruncatedSVD
from memory_profiler import profile  
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap  
import matplotlib.pyplot as plt  

MAX_SAMPLE_SIZE = int(1e9)
COVERAGE_DEPTH = 20
max_n_neighbors = 20
region = 'HLA'
npz_path = "/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/human/%s/ONT_R9/kmer_k16/feature_matrix.npz"%(region)
tsv_path = "/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/human/%s/ONT_R9//kmer_k16/metadata.tsv.gz"%(region)
json_path = "/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/human/%s/ONT_R9/kmer_k16/read_features.json.gz"%(region)

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
print("complete reference_graph")
nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=20,metric='euclidean')

class dim_redu:
    def transform(self, data, n_dimensions: int):
        raise NotImplementedError
    
class scbimap(dim_redu):
    def transform(self,data,n_dimensions):
        mydim = scBiMapEmbedding()
        dim_redu = mydim.transform(data,n_dimensions)
        return dim_redu
    
class spectrual(dim_redu):
    def transform(self,data,n_dimensions):
        mydim = SpectralEmbedding()
        dim_redu = mydim.transform(data,n_dimensions)
        return dim_redu

class TruncatedSVD(dim_redu):
    def transform(self,data,n_dimensions):
        tsvd = TruncatedSVD(n_components=n_dimensions)
        dim_redu = tsvd.fit_transform(data) 
        return dim_redu

class pca(dim_redu):
    def transform(self,data,n_dimensions):
        pca = PCA(n_components=n_dimensions) 
        dim_redu = pca.fit_transform(data)  
        return dim_redu
    
class isomap(dim_redu):
    def transform(self,data,n_dimensions):
        isomap = Isomap(n_components=n_dimensions)  
        dim_redu = isomap.fit_transform(data) 
        return dim_redu
    
@profile
def dimension_reduction(
        data,
        n_dimensions,
        dim_redu_method
):
    start_time = time.time()
    dim_reduced_matrix = dim_redu_method().transform(data,n_dimensions)
    elapsed_time = time.time() - start_time
    method = dim_redu_method.__name__
    print("%s dimension reduction done"%(method))
    print("elapsed time :%f"%(elapsed_time))
    
    nbrs.fit(dim_reduced_matrix)
    _, nbr_indices = nbrs.kneighbors(dim_reduced_matrix)
    df = from_nbr_to_evaluation(nbr_indices,read_ids,reference_graph,des=method)
    file = '/home/miaocj/docker_dir/kNN-overlap-finder/data/evaluation/human/%s/ONT_R9/kmer_k16/%s_dimredu.tsv'%(region,method)
    df.to_csv(file,sep='\t')
    return nbr_indices

# nbr_indices = dimension_reduction(feature_matrix,500,scbimap)
# nbr_indices = dimension_reduction(feature_matrix,500,spectrual)
# nbr_indices = dimension_reduction(feature_matrix,500,TruncatedSVD)
nbr_indices = dimension_reduction(feature_matrix,500,pca)
nbr_indices = dimension_reduction(feature_matrix,500,isomap)

