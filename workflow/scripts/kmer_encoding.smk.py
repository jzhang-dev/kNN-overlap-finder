from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from snakemake_stub import *


import gzip, json
from Bio import SeqIO
import scipy.sparse as sp
from collections import Counter
import numpy as np
import pandas as pd


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


def encode_reads_from_fasta_gz(
    fasta_gz_file, k, sample_fraction: float, seed: int, include_reverse_complement=True
):
    """
    Encodes sequencing reads from a gzipped FASTA file as a sparse matrix.

    Args:
        fasta_gz_file: The path to the gzipped FASTA file.
        k: The length of k-mers to consider.

    Returns:
        A scipy.sparse CSR matrix representing the k-mer counts for each read.
    """

    

    # Load reads
    read_sequences = []
    read_names = []
    strands = []
    read_lengths = []
    start_positions = []
    end_positions = []
    
    with gzip.open(fasta_gz_file, "rt") as handle:  # Open gzipped file in text mode
        for record in SeqIO.parse(handle, "fasta"):
            seq = str(record.seq)
            read_sequences.append(seq)
            read_names.append(record.id)
            strands.append("+")
            read_lengths.append(len(seq))
            start = int(record.id.split("_")[0])
            end = start + len(seq)
            start_positions.append(start)
            end_positions.append(end)

            if include_reverse_complement:
                read_sequences.append(reverse_complement(seq))
                read_names.append(record.id)
                strands.append("-")
                read_lengths.append(len(seq))
                start_positions.append(start)
                end_positions.append(end)

    # Build vocabulary
    vocab = set()
    for seq in read_sequences:
        vocab |= set(seq[p : p + k] for p in range(len(seq) - k + 1))
    rng = np.random.default_rng(seed=seed)
    vocab = set(x for x in vocab if rng.random() <= sample_fraction)
    vocab |= set(reverse_complement(x) for x in vocab)
    kmer_indices = {kmer: i for i, kmer in enumerate(vocab)}

    # Build matrix

    row_ind, col_ind, data = [], [], []
    features = []
    for i, seq in enumerate(read_sequences):
        features_i = []
        for p in range(len(seq) - k + 1):
            kmer = seq[p : p + k]
            j = kmer_indices.get(kmer)
            if j is None:
                continue
            features_i.append(j)
        features.append(features_i)

        kmer_counts = Counter(features_i)
        for j, count in kmer_counts.items():
            row_ind.append(i)
            col_ind.append(j)
            data.append(count)

    feature_matrix = sp.csr_matrix(
        (data, (row_ind, col_ind)), shape=(len(read_sequences), len(vocab))
    )
    metadata = pd.DataFrame(
        dict(
            row_id=list(range(len(read_sequences))),
            read_name=read_names,
            strand=strands,
            read_length=read_lengths,
            start=start_positions,
            end=end_positions,
        )
    )
    return feature_matrix, metadata, features


def main(snakemake: "SnakemakeContext"):
    input_file = snakemake.input["fasta"]
    output_npz_file = snakemake.output["npz"]
    output_tsv_file = snakemake.output["tsv"]
    output_json_file = snakemake.output['json']
    k = snakemake.params["k"]
    sample_fraction = snakemake.params["sample_fraction"]
    seed = snakemake.params["seed"]

    feature_matrix, metadata, read_features = encode_reads_from_fasta_gz(
        fasta_gz_file=input_file, k=k, sample_fraction=sample_fraction, seed=seed
    )
    sp.save_npz(output_npz_file, feature_matrix)
    metadata.to_csv(output_tsv_file, sep="\t", index=False)
    with gzip.open(output_json_file, 'wt') as f:
        json.dump(read_features, f)

if __name__ == "__main__":
    main(snakemake)
