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


def hash_kmer_encoding(
    read_sequences: Sequence[str],
    k: int,
    *,
    sample_fraction: float,
    min_multiplicity: int,  
    seed: int
) -> Mapping[str, int]:
    """
    Hash kmer encoding for a list of sequences.
    """


    max_hash = (2**64 - 1) * sample_fraction  # 64-bit hash value
    hash_to_column = {}  # Maps hash value to column index
    current_col_idx = 0
    row_ind, col_ind, data = [], [], []

    for row_idx, seq in enumerate(read_sequences):
        # Use defaultdict for faster per-sequence counting
        seq_counts = collections.defaultdict(int)
        
        # Count kmers in this sequence
        for p in range(len(seq) - k + 1):
            kmer = seq[p:p+k]
            hashed = xxhash.xxh64(kmer, seed=seed).intdigest()
            if hashed <= max_hash:
                seq_counts[hashed] += 1
                # Assign column index if new hash
                if hashed not in hash_to_column:
                    hash_to_column[hashed] = current_col_idx
                    current_col_idx += 1
        
        # Add non-zero entries for this sequence
        for hashed, count in seq_counts.items():
            row_ind.append(row_idx)
            col_ind.append(hash_to_column[hashed])
            data.append(count)
    _feature_matrix = sp.csr_matrix((data, (row_ind, col_ind)),
        shape=(len(read_sequences), current_col_idx),
        dtype=np.int32
    )
    col_sums = _feature_matrix.sum(axis=0).A1  # 例如：[0.8, 1.2, 0.5, ...]
    mask = col_sums >= min_multiplicity  # 布尔掩码，如 [False, True, False, ...]
    feature_matrix = _feature_matrix[:, mask]  # 只保留满足条件的列

    return feature_matrix
        



def encode_reads(
    fasta_path: str,
    k,
    *,
    sample_fraction: float,
    min_multiplicity: int,
    seed: int,
):
    """
    Encodes sequencing reads from a gzipped FASTA file as a sparse matrix.
    """
    # Load reads
    read_names, read_orientations, read_sequences = load_reads(fasta_path=fasta_path)

    feature_matrix = hash_kmer_encoding(
        read_sequences=read_sequences,
        k=k,
        sample_fraction=sample_fraction,
        min_multiplicity=min_multiplicity,
        seed=seed,
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
    )
    sp.save_npz(output_npz_file, feature_matrix)

if __name__ == "__main__":
    main(snakemake)