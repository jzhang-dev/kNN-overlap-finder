from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from snakemake_stub import *


import gzip, json, collections
from typing import Sequence, Mapping, Collection
from Bio import SeqIO
import scipy.sparse as sp
import numpy as np
import pandas as pd
import sys
sys.path.append("scripts")
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
    fasta_path: str,
    paf_path: str):
    read_sequences = []
    read_names = []
    read_orientations = []

    with open_gzipped(paf_path, "rt") as file:
        reads_aligned = []
        for row in file:  
            columns = row.strip().split('\t') 
            reads_aligned.append(columns[0])

 # Open gzipped file in text mode
    for record in parse_fasta(fasta_path):
        if record[0] in reads_aligned:
            seq = record[1]
            read_sequences.append(seq)
            read_names.append(record[0])
            read_orientations.append("+")

            # Include reverse complement
            read_sequences.append(reverse_complement(seq))
            read_names.append(record[0])
            read_orientations.append("-")
    return read_names, read_orientations, read_sequences


def build_kmer_index(
    read_sequences: Sequence[str],
    k: int,
    *,
    sample_fraction: float,
    min_multiplicity: int,
    seed: int,
    processes: int,
) -> Mapping[str, int]:
    kmer_counter = collections.Counter()
    for seq in read_sequences:
        for p in range(len(seq) - k + 1):
            kmer = seq[p : p + k]
            kmer_counter[kmer] += 1

    kmer_spectrum = collections.Counter(x for x in kmer_counter.values() if x <= 10)
    print(kmer_spectrum)

    rng = np.random.default_rng(seed=seed)
    vocabulary = set(
        x
        for x, count in kmer_counter.items()
        if count >= min_multiplicity and rng.random() <= sample_fraction
    )
    vocabulary |= set(reverse_complement(x) for x in vocabulary)
    kmer_indices = {kmer: i for i, kmer in enumerate(vocabulary)}
    return kmer_indices


def build_feature_matrix(
    read_sequences: Sequence[str],
    kmer_indices: Mapping[str, int],
    k: int,
    *,
    processes: int,
) -> tuple[sp.csr_matrix, Sequence[Sequence[int]]]:
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
        kmer_counts = collections.Counter(features_i)
        for j, count in kmer_counts.items():
            row_ind.append(i)
            col_ind.append(j)
            data.append(count)
    feature_matrix = sp.csr_matrix(
        (data, (row_ind, col_ind)), shape=(len(read_sequences), len(kmer_indices))
    )
    return feature_matrix, read_features


def encode_reads(
    fasta_path: str,
    paf_path:str,
    info_path: str,
    k,
    *,
    sample_fraction: float,
    min_multiplicity: int,
    seed: int,
    processes: int,
):
    """
    Encodes sequencing reads from a gzipped FASTA file as a sparse matrix.
    """
    # Load reads info
    info_df = pd.read_table(info_path).set_index("read_name")

    # Load reads
    read_names, read_orientations, read_sequences = load_reads(fasta_path=fasta_path,paf_path=paf_path)

    # Build vocabulary
    kmer_indices = build_kmer_index(
        read_sequences=read_sequences,
        k=k,
        sample_fraction=sample_fraction,
        min_multiplicity=min_multiplicity,
        seed=seed,
        processes=processes,
    )

    # Build matrix
    feature_matrix, read_features = build_feature_matrix(
        read_sequences=read_sequences,
        kmer_indices=kmer_indices,
        k=k,
        processes=processes,
    )

    # Build metadata
    def flip(strand):
        return {"+": "-", "-": "+"}[strand]

    rows = []
    for i in range(len(read_sequences)):
        read_name = read_names[i]
        read_orientation = read_orientations[i]
        reference_strand = info_df.at[read_name, "reference_strand"]
        if read_orientation == "-":
            reference_strand = flip(reference_strand)
        rows.append(
            dict(
                read_id=i,
                read_name=read_name,
                read_orientation=read_orientation,
                read_length=info_df.at[read_name, "read_length"],
                reference_strand=reference_strand,
                reference_start=info_df.at[read_name, "reference_start"],
                reference_end=info_df.at[read_name, "reference_end"],
            )
        )
    metadata = pd.DataFrame(rows)

    return feature_matrix, read_features, metadata


def main(snakemake: "SnakemakeContext"):
    input_fasta_file = snakemake.input["fasta"]
    input_tsv_file = snakemake.input["tsv"]
    paf_file = snakemake.input["paf"]
    output_npz_file = snakemake.output["npz"]
    output_json_file = snakemake.output["json"]
    output_tsv_file = snakemake.output["tsv"]
    k = int(snakemake.wildcards["k"])
    sample_fraction = snakemake.params["sample_fraction"]
    min_multiplicity = snakemake.params["min_multiplicity"]
    seed = snakemake.params["seed"]
    threads = snakemake.threads

    feature_matrix, read_features, metadata = encode_reads(
        fasta_path=input_fasta_file,
        paf_path=paf_file,
        info_path=input_tsv_file,
        k=k,
        sample_fraction=sample_fraction,
        min_multiplicity=min_multiplicity,
        seed=seed,
        processes=threads,
    )
    sp.save_npz(output_npz_file, feature_matrix)
    with open_gzipped(output_json_file, "wt") as f:
        json.dump(read_features, f)
    metadata.to_csv(output_tsv_file, index=False, sep="\t")


if __name__ == "__main__":
    main(snakemake)
