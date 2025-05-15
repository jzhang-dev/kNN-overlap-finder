from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from snakemake_stub import *


import gzip, json, collections
from typing import Sequence, Mapping, Collection
from Bio import SeqIO
import scipy.sparse as sp
import numpy as np
import pandas as pd
import xxhash
from multiprocessing import Pool, Manager
from functools import partial

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

    with gzip.open(fasta_path, "rt") as handle:  # Open gzipped file in text mode
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

def process_sequence(args, k, seed, all_kmer_number, max_hash):
    row_idx, seq = args
    seq_counts = collections.defaultdict(int)
    
    # Count kmers in this sequence
    for p in range(len(seq) - k + 1):
        kmer = seq[p:p+k]
        hashed = xxhash.xxh64(kmer, seed=seed).intdigest()
        if hashed <= max_hash:
            seq_counts[hashed] += 1 
    
    # Return non-zero entries for this sequence
    return [(row_idx, hashed, count) for hashed, count in seq_counts.items()]

def build_sparse_matrix_multiprocess(read_sequences, k, seed, sample_fraction, min_multiplicity, n_processes):
    all_kmer_number = 2**64
    max_hash = all_kmer_number * sample_fraction
    
    # Parallel processing
    with Pool(n_processes) as pool:
        func = partial(process_sequence, k=k, seed=seed, 
                      all_kmer_number=all_kmer_number, max_hash=max_hash)
        results = pool.map(func, enumerate(read_sequences))
    
    # Flatten results and build CSR matrix
    row_ind = [r[0] for result in results for r in result]
    col_ind = [r[1] for result in results for r in result]
    data = [r[2] for result in results for r in result]
    print('all feature matrix building done')
    unique_cols, re_col_ind = np.unique(col_ind, return_inverse=True)
    # Determine matrix shape
    n_rows = len(read_sequences)
    n_cols = len(unique_cols)
    _feature_matrix = sp.csr_matrix((data, (row_ind, re_col_ind)),
        shape=(n_rows, n_cols),
        dtype=np.int32
    )
    print(_feature_matrix.shape)
    col_sums = _feature_matrix.sum(axis=0).A1
    mask = col_sums >= min_multiplicity 
    feature_matrix = _feature_matrix[:, mask] 
    print(feature_matrix.shape)
    return feature_matrix

def encode_reads(
    fasta_path: str,
    k,
    *,
    sample_fraction: float,
    min_multiplicity: int,
    seed: int,
    n_processes: int
):
    """
    Encodes sequencing reads from a gzipped FASTA file as a sparse matrix.
    """
    # Load reads
    read_names, read_orientations, read_sequences = load_reads(fasta_path=fasta_path)

    feature_matrix = build_sparse_matrix_multiprocess(
        read_sequences=read_sequences,
        k=k,
        seed=seed,
        sample_fraction=sample_fraction,
        min_multiplicity=min_multiplicity,
        n_processes=n_processes
    )
    return feature_matrix

def main(snakemake: "SnakemakeContext"):

    input_fasta_file = snakemake.input["fasta"]

    output_npz_file = snakemake.output["npz"]
    k = int(snakemake.wildcards["k"])
    sample_fraction = snakemake.params["sample_fraction"]
    min_multiplicity = snakemake.params["min_multiplicity"]
    seed = snakemake.params["seed"]
    threads = snakemake.threads

    feature_matrix = encode_reads(
        fasta_path=input_fasta_file,
        k=k,
        sample_fraction=sample_fraction,
        min_multiplicity=min_multiplicity,
        seed=seed,
        n_processes=threads
    )
    sp.save_npz(output_npz_file, feature_matrix)

if __name__ == "__main__":
    main(snakemake)