import sys
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from snakemake_stub import *
from Bio import SeqIO
import scipy.sparse as sp
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors  
import numpy as np
from metagenome_function import load_reads,build_kmer_index,build_feature_matrix,get_table,get_simhash,evaluate
import gzip
import json,pickle


output_kmer_indices_file = "/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/metagenome/GTDB/ref_kmer_indices.pkl"
origin_ref_datebase = '/home/miaocj/docker_dir/kNN-overlap-finder/data/metagenome_reference/GCR.fa.split/GCR.part_001.fa' 
##used for generate simulated reads, include 19 species
query_reads = '/home/miaocj/docker_dir/kNN-overlap-finder/data/metagenome_reads/part001/pbsim_ONT_95_20k/reads.fa'
que_read_tax_file = '/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/metagenome/que_read_tax.json'
ref_read_tax_file = '/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/metagenome/ref_read_tax.json'
output_que_npz_file= '/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/metagenome/que_feature_matrix.npz'
output_que_json_file = '/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/metagenome/que_read_features.json.gz'

##Part1: generate a read-species dict, used for final examination

ref_reads_tax_list = []
with open(origin_ref_datebase, "rt") as handle:
    for record in SeqIO.parse(handle, "fasta"):
        tax=record.id[:6]
        ref_reads_tax_list.append(tax)  ##ID example: QSQV01000067.1, 'QSQV01' present species code
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


with gzip.open(que_read_tax_file, "wt") as f:
    json.dump(que_read_tax, f)

print("loading reads")
qread_names, qread_orientations, qread_sequences = load_reads(query_reads)
print("done\nbuilding feature matrix")
with open(output_kmer_indices_file, 'rb') as f:  # 'rb' 表示以二进制读模式打开
    kmer_indices = pickle.load(f)
que_feature_matrix,que_read_features = build_feature_matrix(read_sequences=qread_sequences,kmer_indices=kmer_indices,k=16)
sp.save_npz(output_que_npz_file, que_feature_matrix)
with gzip.open(output_que_json_file, "wt") as f:
    json.dump(que_read_features, f)


