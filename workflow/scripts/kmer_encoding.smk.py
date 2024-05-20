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


def encode_reads(
    fasta_path: str,
    info_path: str,
    k,
    sample_fraction: float,
    seed: int,
    include_reverse_complement=True,
):
    """
    Encodes sequencing reads from a gzipped FASTA file as a sparse matrix.

    Args:
        fasta_gz_file: The path to the gzipped FASTA file.
        k: The length of k-mers to consider.

    Returns:
        A scipy.sparse CSR matrix representing the k-mer counts for each read.
    """
    # Load reads info
    info_df = pd.read_table(info_path).set_index("read_name")

    # Load reads
    read_sequences = []
    read_names = []
    read_orientations = []

    with gzip.open(fasta_path, "rt") as handle:  # Open gzipped file in text mode
        for record in SeqIO.parse(handle, "fasta"):
            seq = str(record.seq)
            read_sequences.append(seq)
            read_names.append(record.id)
            read_orientations.append("+")

            if include_reverse_complement:
                read_sequences.append(reverse_complement(seq))
                read_names.append(record.id)
                read_orientations.append("-")

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
    read_features = []
    for i, seq in enumerate(read_sequences):
        features_i = []
        for p in range(len(seq) - k + 1):
            kmer = seq[p : p + k]
            j = kmer_indices.get(kmer)
            if j is None:
                continue
            features_i.append(j)

        read_features.append(features_i)

        kmer_counts = Counter(features_i)
        for j, count in kmer_counts.items():
            row_ind.append(i)
            col_ind.append(j)
            data.append(count)

    feature_matrix = sp.csr_matrix(
        (data, (row_ind, col_ind)), shape=(len(read_sequences), len(vocab))
    )

    # Build metadata
    rows = []
    for i, seq in enumerate(read_sequences):
        read_name = read_names[i]
        rows.append(
            dict(
                read_id=i,
                read_name=read_name,
                read_orientation=read_orientations[i],
                read_length=info_df.at[read_name, "read_length"],
                reference_strand=info_df.at[read_name, "reference_strand"],
                reference_start=info_df.at[read_name, "reference_start"],
                reference_end=info_df.at[read_name, "reference_end"],
            )
        )
    metadata= pd.DataFrame(rows)

    return feature_matrix, read_features, metadata


def main(snakemake: "SnakemakeContext"):
    input_fasta_file = snakemake.input["fasta"]
    input_tsv_file = snakemake.input["tsv"]
    output_npz_file = snakemake.output["npz"]
    output_json_file = snakemake.output["json"]
    output_tsv_file = snakemake.output["tsv"]
    k = snakemake.params["k"]
    sample_fraction = snakemake.params["sample_fraction"]
    seed = snakemake.params["seed"]

    feature_matrix, read_features,metadata = encode_reads(
        fasta_path=input_fasta_file,
        info_path=input_tsv_file,
        k=k,
        sample_fraction=sample_fraction,
        seed=seed,
    )
    sp.save_npz(output_npz_file, feature_matrix)
    with gzip.open(output_json_file, "wt") as f:
        json.dump(read_features, f)
    metadata.to_csv(output_tsv_file, index=False, sep='\t')


if __name__ == "__main__":
    main(snakemake)
