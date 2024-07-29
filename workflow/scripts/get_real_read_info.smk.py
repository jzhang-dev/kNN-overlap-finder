from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from snakemake_stub import *


import gzip, json
from Bio import SeqIO
import scipy.sparse as sp
from collections import Counter
import numpy as np
import pandas as pd

def get_metadata(fasta_gz_file,paf_gz_file) -> pd.DataFrame:
    read_sequences = []
    read_names = []
    strands = []
    read_lengths = []
    start_positions = []
    end_positions = []

    with gzip.open(paf_gz_file, "rt") as file:
        max_values = {}  
        for row in file:  
            columns = row.strip().split('\t') 
            query_id = columns[0]  
            match_bases = int(columns[9]) 
            max_values[query_id] = columns 
            if query_id in max_values:  
                if match_bases > int(max_values[query_id][9]):  
                    max_values[match_bases] = columns
            else:  
                continue
    with gzip.open(fasta_gz_file, "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            if record.id in max_values.keys():
                columns = max_values[record.id]
                fastq_dict = max_values[record.id]
                read_sequences.append(record.seq)
                read_names.append(record.id)
                strands.append(columns[4])
                read_lengths.append(columns[1])
                start_positions.append(columns[7])
                end_positions.append(columns[8])

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
    fasta_file = snakemake.input["fasta"]
    paf_file = snakemake.input["paf"]
    output_tsv_file = snakemake.output["tsv"]
    metadata = get_metadata(fasta_file,paf_file)
    metadata.to_csv(output_tsv_file, sep="\t", index=False)

if __name__ == "__main__":
    main(snakemake)