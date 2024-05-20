from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from snakemake_stub import *


import gzip, json
from Bio import SeqIO
import scipy.sparse as sp
from collections import Counter
import numpy as np
import pandas as pd


def get_metadata(fasta_gz_file) -> pd.DataFrame:
    read_sequences = []
    read_names = []
    strands = []
    read_lengths = []
    start_positions = []
    end_positions = []

    with gzip.open(fasta_gz_file, "rt") as handle:
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

    metadata = pd.DataFrame(
        dict(
            read_name=read_names,
            read_length=read_lengths,
            reference_strand=strands,
            reference_start=start_positions,
            reference_end=end_positions,
        )
    )
    return metadata


def main(snakemake: "SnakemakeContext"):
    input_file = snakemake.input["fasta"]

    output_tsv_file = snakemake.output["tsv"]

    metadata = get_metadata(input_file)

    metadata.to_csv(output_tsv_file, sep="\t", index=False)


if __name__ == "__main__":
    main(snakemake)
