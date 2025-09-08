from Bio import SeqIO
import collections
import numpy as np
import scipy.sparse as sp
import sys,re,gzip
from pathlib import Path
from typing import Mapping
import pandas as pd
sys.path.append('/home/miaocj/docker_dir/kNN-overlap-finder/scripts')
from accelerate import open_gzipped,parse_fasta

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

def load_reads(
    db_path: str):
    read_names = []
    read_sequences = []
    read_orientations = []

 # Open gzipped file in text mode
    for record in parse_fasta(db_path):
        seq = record[1]
        read_sequences.append(seq)
        read_names.append(record[0])
        read_orientations.append("+")

        # Include reverse complement
        read_sequences.append(reverse_complement(seq))
        read_names.append(record[0])
        read_orientations.append("-")
    return read_names, read_orientations, read_sequences

def finding_kmer(kmer_dict, read_sequences, k):
    row_ind, col_ind, data = [], [], []
    for i,seq in enumerate(read_sequences):
        features_i = []
        for p in range(len(seq) - k + 1):
            kmer = seq[p : p + k]
            j = kmer_dict.get(kmer)
            if j is None:
                continue
            features_i.append(j)
        kmer_counts = collections.Counter(features_i)
        for j, count in kmer_counts.items():
            row_ind.append(i)
            col_ind.append(j)
            data.append(count)
    feature_matrix = sp.csr_matrix(
        (data, (row_ind, col_ind)), shape=(len(read_sequences), len(kmer_dict))
    )
    return feature_matrix


def encode_reads(
    db_path: str,
    kmer_dict:Mapping[int,str], #  {'ATGG':index}
    id_dict: Mapping[str,str], # {read_name:gca_id}
    gtdb_taxonomy: pd.DataFrame, # [gca_id,species]
    k
):
    """
    Encodes sequencing reads from a gzipped FASTA file as a sparse matrix.
    """
    # Load reads
    read_names, read_orientations, read_sequences = load_reads(db_path=db_path)
    print("reads loading done")
    # Build matrix
    feature_matrix = finding_kmer(kmer_dict, read_sequences, k)
    print("find kmer done")
    # Build metadata
    rows = []
    for i in range(len(read_sequences)):
        read_name = read_names[i]
        read_orientation = read_orientations[i]
        gca_id = id_dict[read_name]
        rows.append(
            dict(
                read_name=read_name,
                read_orientation=read_orientation,
                gca_id=gca_id,
                species=gtdb_taxonomy.at[gca_id, "species"],
            )
        )
    metadata = pd.DataFrame(rows)

    return feature_matrix, metadata