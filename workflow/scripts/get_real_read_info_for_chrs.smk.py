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
    chromosomes = []
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
        select_num,allseq_num,aligned_num,select_length,allseq_length,aligned_length = [0,0,0,0,0,0]
        for record in SeqIO.parse(handle, "fasta"):
            allseq_num += 1
            allseq_length += len(record.seq)
            if record.id in max_values.keys():
                aligned_num += 1
                columns = max_values[record.id]
                aligned_length += int(columns[1])
                if int(columns[9])/int(columns[1]) > 0.5:
                    select_num += 1
                    select_length += int(columns[1])
                    fastq_dict = max_values[record.id]
                    read_sequences.append(record.seq)
                    read_names.append(record.id)
                    strands.append(columns[4])
                    read_lengths.append(columns[1])
                    start_positions.append(columns[7])
                    end_positions.append(columns[8])
                    chromosomes.append(columns[5])

    metadata = pd.DataFrame(
        dict(
            read_name=read_names,
            read_length=read_lengths,
            reference_chromosome=chromosomes,
            reference_strand=strands,
            reference_start=start_positions,
            reference_end=end_positions,
        )
    )
    reads_stat = pd.DataFrame(
        dict(
        allseq_num="{:.0f}".format(allseq_num),
        allseq_length="{:.0f}".format(allseq_length),
        select_num="{:.0f}".format(select_num),
        select_length="{:.0f}".format(select_length),
        aligned_num="{:.0f}".format(aligned_num),
        aligned_length="{:.0f}".format(aligned_length),
        percentage_num_of_align="{:.4f}%".format(aligned_num/allseq_num*100),
        percentage_len_of_align="{:.4f}%".format(aligned_length/allseq_length*100),
        percentage_num_of_select="{:.4f}%".format(select_num/allseq_num*100),
        percentage_len_of_select="{:.4f}%".format(select_length/allseq_length*100),
        ),index=[0]
    ).T
    return metadata,reads_stat


def main(snakemake: "SnakemakeContext"):
    fasta_file = snakemake.input["fasta"]
    paf_file = snakemake.input["paf"]
    output_tsv_file = snakemake.output["tsv"]
    output_stat_file = snakemake.output["stat"]
    metadata,read_stat = get_metadata(fasta_file,paf_file)
    metadata.to_csv(output_tsv_file, sep="\t", index=False)
    read_stat.to_csv(output_stat_file,sep="\t")

if __name__ == "__main__":
    main(snakemake)