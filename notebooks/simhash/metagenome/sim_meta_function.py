
import sys
sys.path.append("scripts")
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
from numba import njit, prange
from itertools import chain 
from collections import Counter
from sklearn.metrics import precision_score, recall_score  

def init_reverse_complement():
    TRANSLATION_TABLE = str.maketrans("ACTGactg", "TGACtgac")

    def reverse_complement(sequence: str) -> str:
        """
        >>> reverse_complement("AATC")
        'GATT'
        >>> reverse_complement("CCANT")
        'ANTGG'
        """
        sequence = str(sequence)
        return sequence.translate(TRANSLATION_TABLE)[::-1]

    return reverse_complement
reverse_complement = init_reverse_complement()


def load_reads(fasta_path: str):
    read_sequences = []
    read_names = []
    read_orientations = []

    with open(fasta_path, "rt") as handle:  # Open gzipped file in text mode
        for record in SeqIO.parse(handle, "fasta"):
            seq = str(record.seq)
            read_sequences.append(seq)
            read_names.append(record.id)
            read_orientations.append("+")

            # Include reverse complement
            read_sequences.append(reverse_complement(seq))
            read_names.append(record.id)
            read_orientations.append("-")

    return read_names, read_orientations, read_sequences

def build_kmer_index(
    read_sequences: Sequence[str],
    k: int,
    *,
    sample_fraction: float,
    min_multiplicity: int,
    seed: int,
) -> Mapping[str, int]:
    kmer_counter = collections.Counter()
    for seq in read_sequences:
        for p in range(len(seq) - k + 1):
            kmer = seq[p : p + k]
            kmer_counter[kmer] += 1

    kmer_spectrum = collections.Counter(x for x in kmer_counter.values() if x <= 10)
    print(kmer_spectrum)

    rng = np.random.default_rng(seed=seed)
    vocabulary = set(
        x
        for x, count in kmer_counter.items()
        if count >= min_multiplicity and rng.random() <= sample_fraction
    )
    vocabulary |= set(reverse_complement(x) for x in vocabulary)
    kmer_indices = {kmer: i for i, kmer in enumerate(vocabulary)}
    return kmer_indices

def build_feature_matrix(
    read_sequences: Sequence[str],
    kmer_indices: Mapping[str, int],
    k: int,
) -> tuple[sp.csr_matrix, Sequence[Sequence[int]]]:
    row_ind, col_ind, data = [], [], []
    read_features = []
    for i, seq in enumerate(read_sequences):
        features_i = []
        for p in range(len(seq) - k + 1):
            kmer = seq[p : p + k]
            j = kmer_indices.get(kmer)
            if j is None:
                continue
            features_i.append(j)

        read_features.append(features_i)

        kmer_counts = collections.Counter(features_i)
        for j, count in kmer_counts.items():
            row_ind.append(i)
            col_ind.append(j)
            data.append(count)

    feature_matrix = sp.csr_matrix(
        (data, (row_ind, col_ind)), shape=(len(read_sequences), len(kmer_indices))
    )
    return feature_matrix, read_features


def _hash(kmer_index: int, seed: int) -> np.ndarray:  
    hash_value = mmh3.hash(str(kmer_index), seed=seed)
    binary_string = "{0:032b}".format(hash_value & 0xFFFFFFFF)  
    hash_array = np.array([int(x) for x in binary_string], dtype=np.int32) 
    hash_array = np.where(hash_array == 0, -1, 1)  
    return hash_array  

def mp_get_hashtable(  
    feature_matrix: list,   
    repeat: int,   
    seed: int,
    processes:int,) -> Mapping[int,list]:  
      
    rng = np.random.default_rng(seed)  
    hash_seeds = rng.integers(low=0, high=2**32 - 1, size=repeat, dtype=np.uint64)  
    kmer_num = feature_matrix.shape[1]
    hash_table = np.empty((kmer_num,repeat),dtype=object) 

    with sharedmem.MapReduce(np=processes) as pool:

        def work(i):
            seed = hash_seeds[i]
            result = np.empty(kmer_num, dtype=object) 
            for kmer_index in range(kmer_num):
                result[kmer_index]=_hash(kmer_index, seed=seed)
            return i,result

        def reduce(i, result):
            hash_table[:,i] = result
        pool.map(work, range(repeat), reduce=reduce)
    return hash_table


def get_table(
    kmer_num: int,
    *,
    seed: int,
    repeat=100) -> Mapping[int,list]:  
    
    rng = np.random.default_rng(seed)  
    hash_seeds = rng.integers(low=0, high=2**32 - 1, size=repeat, dtype=np.uint64)  

    hash_table = np.empty((kmer_num,repeat,32),dtype=np.int8)  
    for flag,seed in enumerate(hash_seeds):
        for kmer_index in range(kmer_num):
            hash_table[kmer_index,flag,:]=_hash(kmer_index, seed=seed)
            new_hash_table=np.reshape(hash_table,(kmer_num,repeat*32))
    return new_hash_table

def get_simhash(  
    read_features: list,
    hash_table) -> Mapping[int,list]:  
    all_read_simhash = []
    for read_kmer in read_features:
        one_read_hash = np.sum(hash_table[read_kmer,:],axis=0)
        simhash = np.where(one_read_hash > 0, 1, 0)
        all_read_simhash.append(simhash)
    reads_simhash_array = np.array(all_read_simhash)
    return reads_simhash_array 

def evaluate(indices,ref_read_tax,que_read_tax):
    actual = []
    prediction = []
    for query_read_num,x in enumerate(indices):
        neighbor = x[0]
        neighbor = (neighbor-1)/2  if neighbor %2 !=0 else neighbor/2
        query_read_num = (query_read_num-1)/2  if query_read_num %2 !=0 else query_read_num/2
        prediction.append(ref_read_tax[neighbor])
        actual.append(que_read_tax[query_read_num])

    precision = precision_score(actual, prediction,average='macro')
    sensitivity = recall_score(actual, prediction,average='macro')
    ##计算每个类别的
    precision_sep = precision_score(actual, prediction, average=None)  
    sensitivity_sep = recall_score(actual, prediction, average=None)
    return precision,sensitivity,precision_sep,sensitivity_sep
