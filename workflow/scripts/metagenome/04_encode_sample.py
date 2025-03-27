from Bio import SeqIO
import collections
import numpy as np
import scipy.sparse as sp
import sys 
import pickle
import gzip,json
from encode_function import load_reads,finding_kmer


with open("/home/miaocj/docker_dir/kNN-overlap-finder/data/metagenome_reference/GTDB/kmer_dict.pkl", "rb") as file:
    kmer_dict = pickle.load(file)
print('kmer_dict load finished')
db_path = "/home/miaocj/docker_dir/kNN-overlap-finder/data/metagenome_reads/cami_sample0/anonymous_reads.fa"
read_names, read_orientations, read_sequences = load_reads(db_path=db_path)
print("reads loading done")
# Build matrix
feature_matrix, read_features = finding_kmer(kmer_dict, read_sequences, 16)

output_ref_npz_file = '/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/metagenome/GTDB/sample1_feature_matrix.npz'
output_json_file = '/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/metagenome/GTDB/sample1_read_features.json.gz'
sp.save_npz(output_ref_npz_file, feature_matrix)
with gzip.open(output_json_file, "wt") as f:
    json.dump(read_features, f)
