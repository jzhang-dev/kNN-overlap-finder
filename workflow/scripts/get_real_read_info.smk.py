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
    ## finding the alighnment query reads have max match base
    with open(paf_gz_file, "rt") as file:
        max_values = {}  
        for row in file:  
            columns = row.strip().split('\t') 
            query_id = columns[0]  
            match_bases = int(columns[9]) 
            max_values[query_id] = columns 
            if query_id in max_values:
                if match_bases > int(max_values[query_id][9]):  
                    max_values[query_id] = columns
            else:  
                continue
    ## filter if the match base number < 50% of the reads length
    pass_reads = []
    with gzip.open(fasta_gz_file, "rt") as handle:
        select_num,aligned_num,select_length,aligned_read_length = [0,0,0,0]
        for record in SeqIO.parse(handle, "fasta"):
            assert record.id not in read_names, f"{record.id} is a duplicate read, check fasta file to remove duplicate reads."
            aligned_num += 1
            columns = max_values[record.id]
            aligned_read_length += int(columns[1])
            if int(columns[9])/int(columns[1]) > 0.5:
                pass_reads.append(">%s\n%s\n"%(record.id,record.seq))
                select_num += 1
                select_length += int(columns[1])
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
    reads_stat = pd.DataFrame(
        dict(
        select_num=select_num,
        select_length=select_length,
        aligned_num=aligned_num,
        aligned_length=aligned_read_length,
        percentage_num_of_select=select_num/aligned_num,
        percentage_len_of_select=select_length/aligned_read_length,
        ),index=[0]
    ).T
    return metadata,reads_stat,pass_reads


def main(snakemake: "SnakemakeContext"):
    fasta_file = snakemake.input["fasta_aligned"]
    paf_file = snakemake.input["paf"]

    fasta = snakemake.output["fasta"]
    output_tsv_file = snakemake.output["tsv"]
    output_stat_file = snakemake.output["stat"]
    
    metadata,read_stat,pass_reads = get_metadata(fasta_file,paf_file)
    metadata.to_csv(output_tsv_file, sep="\t", index=False)
    read_stat.to_csv(output_stat_file,sep="\t")
    joined_string = ''.join(pass_reads)  
    with gzip.open(fasta, 'wt') as file:  
        file.write(joined_string)  

if __name__ == "__main__":
    main(snakemake)