import sys
sys.path.append("./")
sys.path.append("/home/miaocj/docker_dir/kNN-overlap-finder/.snakemake/conda/4262b1bf4bf1ffb403c0eb7a42ad5906_/lib/python3.12/site-packages")
sys.path.append("/home/miaocj/docker_dir/kNN-overlap-finder/.snakemake/conda/4506eccf78279d93d0e8a34c035e91c5_/lib/python3.12/site-packages")
sys.path.append("/home/miaocj/docker_dir/kNN-overlap-finder/.snakemake/conda/6bda807e3967eae797c7b1b9eeaee8db_/lib/python3.12/site-packages")
sys.path.append("/home/miaocj/docker_dir/kNN-overlap-finder/.snakemake/conda/c2a47d89d1d34e789fdf782557bb7194_/lib/python3.12/site-packages")
sys.path.append("/home/miaocj/docker_dir/kNN-overlap-finder/.snakemake/conda/c6c5514ada15b890fb27d1e36371554c_/lib/python3.12/site-packages")
sys.path.append("/home/miaocj/docker_dir/kNN-overlap-finder/.snakemake/conda/d964a294c2d0fef56a434c021026281e_/lib/python3.12/site-packages")
sys.path.append("/home/miaocj/docker_dir/kNN-overlap-finder/.snakemake/conda/e1c932db5cd4271709e54d8028824bc9_/lib/python3.12/site-packages")

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from snakemake_stub import *
import gzip, json, collections
from typing import Sequence, Mapping, Collection
from Bio import SeqIO
import scipy.sparse as sp
import numpy as np
import pandas as pd
import mmh3
import sharedmem
from sklearn.neighbors import NearestNeighbors  
import numpy as np
from sim_meta_function import load_reads,build_kmer_index,build_feature_matrix,mp_get_hashtable,get_simhash,evaluate
from numba import njit, prange
from itertools import chain 
from collections import Counter

ref_database = '/home/miaocj/docker_dir/data/metagenome/bacteria/part1.fa'
query_reads = '/home/miaocj/docker_dir/data/metagenome/bacteria/pbsim_ONT_98_30k_10dep_part1_reads.fasta'

ref_reads_tax_list = []
with open(ref_database) as file:
    for lines in file:
        if lines[0] == '>':
            line = lines.strip().split(' ')
            ref_reads_tax_list.append(line[1]+line[2])
ref_read_tax = {i:tax for i,tax in enumerate(ref_reads_tax_list)}

flag = 0
que_read_tax = {}
with open(query_reads) as file:
    for lines in file:
        if lines[0] == '>':
            start = lines.index('S')
            end = lines.index('_')
            ref_num = lines[start+1:end]
            que_read_tax[flag] = ref_read_tax[int(ref_num)-1]
            flag +=1



read_names, read_orientations, read_sequences = load_reads(ref_database)

sample_fraction=0.1
min_multiplicity=2
seed=562104830
kmer_indices = build_kmer_index(        
        read_sequences=read_sequences,
        k=16,
        sample_fraction=sample_fraction,
        min_multiplicity=min_multiplicity,
        seed=seed)

qread_names, qread_orientations, qread_sequences = load_reads(query_reads)
ref_feature_matrix,ref_read_features = build_feature_matrix(read_sequences=read_sequences,kmer_indices=kmer_indices,k=16)
que_feature_matrix,que_read_features = build_feature_matrix(read_sequences=qread_sequences,kmer_indices=kmer_indices,k=16)
kmer_num = que_feature_matrix.shape[1]
hash_table = mp_get_hashtable(ref_feature_matrix,repeat =100, seed = 4829,processes=10)
ref_reads_simhash_array = get_simhash(ref_read_features,hash_table)
print("ref done")
que_reads_simhash_array = get_simhash(que_read_features,hash_table)


def hamming_distance(x, y):  
    return np.count_nonzero(x != y)
nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto', metric=hamming_distance)
nbrs.fit(ref_reads_simhash_array)  
indices = nbrs.kneighbors(que_reads_simhash_array,return_distance=False)

##evaluate
precision,sensitivity,precision_sep,sensitivity_sep = evaluate(indices,ref_read_tax,que_read_tax)
print(precision,sensitivity,precision_sep,sensitivity_sep )