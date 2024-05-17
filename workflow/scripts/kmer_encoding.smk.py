from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from snakemake_stub import *


import gzip
from Bio import SeqIO
import scipy.sparse as sp
from collections import Counter
import numpy as np


def encode_reads_from_fasta_gz(fasta_gz_file, k, sample_fraction: float, seed: int):
    """
    Encodes sequencing reads from a gzipped FASTA file as a sparse matrix.

    Args:
        fasta_gz_file: The path to the gzipped FASTA file.
        k: The length of k-mers to consider.

    Returns:
        A scipy.sparse CSR matrix representing the k-mer counts for each read.
    """

    vocab = set()
    reads = []

    with gzip.open(fasta_gz_file, "rt") as handle:  # Open gzipped file in text mode
        for record in SeqIO.parse(handle, "fasta"):
            read = str(record.seq)
            reads.append(read)
            vocab.update(read[i : i + k] for i in range(len(read) - k + 1))

    rng = np.random.default_rng(seed=seed)
    vocab_to_index = {
        kmer: i for i, kmer in enumerate(vocab) if rng.random() <= sample_fraction
    }

    row_ind, col_ind, data = [], [], []
    for i, read in enumerate(reads):
        kmer_counts = Counter(read[j : j + k] for j in range(len(read) - k + 1))
        for kmer, count in kmer_counts.items():
            j = vocab_to_index.get(kmer)
            if j is None:
                continue
            row_ind.append(i)
            col_ind.append(vocab_to_index[kmer])
            data.append(count)

    return sp.csr_matrix((data, (row_ind, col_ind)), shape=(len(reads), len(vocab)))


def main(snakemake: "SnakemakeContext"):
    input_file = snakemake.input["fasta"]
    output_file = snakemake.output["npz"]
    k = snakemake.params["k"]
    sample_fraction = snakemake.params["sample_fraction"]
    seed = snakemake.params["seed"]

    matrix = encode_reads_from_fasta_gz(
        fasta_gz_file=input_file, k=k, sample_fraction=sample_fraction, seed=seed
    )
    sp.save_npz(output_file, matrix)


if __name__ == "__main__":
    main(snakemake)
