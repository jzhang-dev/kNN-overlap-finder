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

read_names, read_orientations, read_sequences = load_reads(fasta_path=sys.argv[1])
info_df = pd.read_table(sys.argv[2]).set_index("read_name")
def flip(strand):
    return {"+": "-", "-": "+"}[strand]

rows = []
if 'reference_chromosome' in info_df.columns:
    for i in range(len(read_sequences)):
        read_name = read_names[i]
        read_orientation = read_orientations[i]
        reference_strand = info_df.at[read_name, "reference_strand"]
        reference_chromosome = info_df.at[read_name, "reference_chromosome"]
        if read_orientation == "-":
            reference_strand = flip(reference_strand)
        rows.append(
            dict(
                read_id=i,
                read_name=read_name,
                read_orientation=read_orientation,
                read_length=info_df.at[read_name, "read_length"],
                reference_chromosome=reference_chromosome,
                reference_strand=reference_strand,
                reference_start=info_df.at[read_name, "reference_start"],
                reference_end=info_df.at[read_name, "reference_end"],
            )
        )
else:
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
metadata.to_csv(sys.argv[3], index=False, sep="\t")
