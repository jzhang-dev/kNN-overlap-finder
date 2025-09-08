import sys
sys.path.append("../../../scripts")
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
from Bio import SeqIO
import scipy.sparse as sp
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors  
import numpy as np
from metagenome_function import load_reads,build_kmer_index,build_feature_matrix
import json,pickle,gzip

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

all_ref_database = '/home/miaocj/docker_dir/kNN-overlap-finder/data/metagenome_reference/GTDB/GTDB_database.fna.gz' 
##used for generate simulated reads, include 19 species

# ##Part1: generate a read-species dict, used for final examination
# all_ref_reads_tax_list = []
# with open(all_ref_database, "rt") as handle:
#     for record in SeqIO.parse(handle, "fasta"):
#         tax=record.id[:6]
#         all_ref_reads_tax_list.append(tax)  ##ID example: QSQV01000067.1, 'QSQV01' present species code
# all_ref_read_tax = {i:tax for i,tax in enumerate(all_ref_reads_tax_list)}

# ref_read_tax_file = '/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/metagenome/ref_read_tax.json'

# with gzip.open(ref_read_tax_file, "wt") as f:
#     json.dump(all_ref_read_tax, f)

print("loading reads")
fread_names, fread_orientations, fread_sequences = load_reads(all_ref_database)
print("loading done\nbuilding kmer index")
sample_fraction=0.005
min_multiplicity=2
seed=562104830
kmer_indices = build_kmer_index(        
        read_sequences=fread_sequences,
        k=16,
        sample_fraction=sample_fraction,
        min_multiplicity=min_multiplicity,
        seed=seed)
print("done\nloading query reads")
print("done\nbuilding feature matrix") 

ref_feature_matrix,ref_read_features = build_feature_matrix(read_sequences=fread_sequences,kmer_indices=kmer_indices,k=16)

output_kmer_indices_file = '/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/metagenome/GTDB/ref_kmer_indices.pkl.gz'
output_ref_npz_file = '/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/metagenome/GTDB/ref_feature_matrix.npz'
output_ref_json_file= '/home/miaocj/docker_dir/kNN-overlap-finder/data/feature_matrix/metagenome/GTDB/ref_read_features.json'

with gzip.open(output_kmer_indices_file, 'wb') as f: 
    pickle.dump(kmer_indices, f)

sp.save_npz(output_ref_npz_file, ref_feature_matrix) 
with open(output_ref_json_file, "w") as f:
    json.dump(ref_read_features, f)



##new work flow for metagenome


# split_ref = []
# ref_reads_tax_list = []
# with open(ref_database, "rt") as handle:
#     for record in SeqIO.parse(handle, "fasta"):
#         tax=record.description.split(' ')[1]+record.description.split(' ')[2]
#         k = int(len(record.seq)/2)
#         for p in range(0,len(record.seq) - k + 1,int(len(record.seq)/4)):
#             kmer = record.seq[p : p + k]
#             split_ref.append(kmer)
#             ref_reads_tax_list.append(tax)
#             split_ref.append(reverse_complement(kmer))
# split_ref_tax_dict = {i:tax for i,tax in enumerate(ref_reads_tax_list)}

# ref_reads_tax_list = []
# with open(ref_database) as file:
#     for lines in file:
#         if lines[0] == '>':
#             line = lines.strip().split(' ')
#             ref_reads_tax_list.append(line[1]+line[2])
# ref_read_tax = {i:tax for i,tax in enumerate(ref_reads_tax_list)}
