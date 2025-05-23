from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from snakemake_stub import *

import sys
sys.path.append("scripts")
sys.path.append("../../scripts")
import gzip, json, collections
from typing import Sequence, Mapping, Collection
from Bio import SeqIO
import scipy.sparse as sp
import numpy as np
import pandas as pd
import xxhash
from multiprocessing import Pool, Manager
from functools import partial
from accelerate import open_gzipped
from fasta_load import FastaLoader
import gc,time

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
    loader = FastaLoader(file_path=fasta_path)
    for record in loader:  # 迭代获取每条序列
        seq = str(record.sequence)
        read_sequences.append(seq)
        read_names.append(record.name)
        read_orientations.append("+")

        # Include reverse complement
        read_sequences.append(reverse_complement(seq))
        read_names.append(record.name)
        read_orientations.append("-")

    return read_names, read_orientations, read_sequences

def process_sequence(args, k, seed, all_kmer_number, max_hash):
    row_idx, seq = args
    seq_counts = collections.defaultdict(int)
    kmers = (seq[p:p+k] for p in range(len(seq) - k + 1))
    # Count kmers in this sequence
    for kmer in kmers:
        hashed = xxhash.xxh3_64(kmer, seed=seed).intdigest()
        if hashed <= max_hash:
            seq_counts[hashed] += 1
    if row_idx % 200_000 == 0:
        print(row_idx)

    count = len(seq_counts)
    if count == 0:
        return np.empty((0, 3), dtype=np.uint64)
    result = np.empty((count, 3), dtype=np.uint64)
    for i, (hashed, cnt) in enumerate(seq_counts.items()):
        result[i] = [row_idx, hashed, cnt]
    return result

def build_sparse_matrix_multiprocess(read_sequences, k, seed, sample_fraction, min_multiplicity, n_processes):
    all_kmer_number = 2**64
    max_hash = all_kmer_number * sample_fraction
    # Parallel processing with imap
    time1 = time.time()
    with Pool(n_processes,maxtasksperchild=100) as pool:
        func = partial(process_sequence, 
                      k=k, seed=seed, 
                      all_kmer_number=all_kmer_number, 
                      max_hash=max_hash)
        
        # Process results incrementally
        row_ind, col_ind, data = [], [], []
        for result in pool.imap(func, enumerate(read_sequences), chunksize=1000):
            if result.size > 0:
                row_ind.append(result[:, 0])
                col_ind.append(result[:, 1])
                data.append(result[:, 2])
        pool.close()
        pool.join()
        gc.collect()

    time2 = time.time()
    print(f'stage1(process reads) elapsed time: {time2-time1}')
        # Concatenate all results
    row_ind = np.concatenate(row_ind)
    col_ind = np.concatenate(col_ind)
    data = np.concatenate(data)
    time3 = time.time()
    print(f'stage2(concat all ind) elapsed time: {time3-time2}')

    # Build sparse matrix
    re_col_ind = pd.factorize(col_ind)[0].tolist()
    n_rows = len(read_sequences)
    n_cols = max(re_col_ind) + 1
    time4 = time.time()
    print(f'stage3(factorize) elapsed time: {time4-time3}')

    _feature_matrix = sp.csr_matrix(
        (data, (row_ind, re_col_ind)),
        shape=(n_rows, n_cols),
        dtype=np.int32
    )
    time5 = time.time()
    print(f'stage4(building feature matrix) elapsed time: {time5-time4}')

    # Filter by multiplicity
    col_sums = _feature_matrix.sum(axis=0).A1
    mask = col_sums >= min_multiplicity
    feature_matrix = _feature_matrix[:, mask]
    print(feature_matrix.shape)
    time6 = time.time()
    print(f'stage5(filter feature matrix) elapsed time: {time6-time5}')
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