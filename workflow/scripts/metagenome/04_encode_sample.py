from Bio import SeqIO
import collections
import numpy as np
import scipy.sparse as sp
import sys 

def finding_kmer(kmer_dict, gtdb_part_file,k,batch_size):
    row_ind, col_ind, data = [], [], []
    read_features = []
    for i,record in enumerate(SeqIO.parse(gtdb_part_file, "fasta")):
        features_i = []
        seq = record.seq
        for p in range(len(seq) - k + 1):
            kmer = seq[p : p + k]
            j = kmer_dict.get(kmer)
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
        (data, (row_ind, col_ind)), shape=(batch_size, len(kmer_dict))
    )
    return feature_matrix,read_features


fasta_file = "/home/miaocj/docker_dir/kNN-overlap-finder/data/metagenome_reference/GTDB/mer_sampled.fa"
kmer_dict = collections.defaultdict()
for i,record in enumerate(SeqIO.parse(fasta_file, "fasta")):
    kmer_dict[record.seq] = i
gtdb_part_file = sys.argv[1]
k=16
feature_matrix,read_features = finding_kmer(kmer_dict, gtdb_part_file,16,1666307)
output_ref_npz_file = f'/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/metagenome/GTDB/sample1.npz'
sp.save_npz(output_ref_npz_file, feature_matrix)

