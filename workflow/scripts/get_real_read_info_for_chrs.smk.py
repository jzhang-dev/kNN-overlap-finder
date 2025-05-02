from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from snakemake_stub import *


import gzip, json
from Bio import SeqIO
import scipy.sparse as sp
from collections import Counter
import numpy as np
import pandas as pd
from isal import igzip

def get_metadata(fasta_gz_file,paf_gz_file,fasta) -> pd.DataFrame:
    read_names = []
    strands = []
    read_lengths = []
    start_positions = []
    end_positions = [] 
    chromosomes = []
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
    print("Finding every reads' largest alignment, done")
    with igzip.open(fasta_gz_file, "rt") as handle,open(fasta, 'wt') as out_file:
        for record in SeqIO.parse(handle, "fasta"):
            assert record.id not in read_names, f"{record.id} is a duplicate read, check fasta file to remove duplicate reads."
            columns = max_values[record.id]
            if int(columns[9])/int(columns[1]) > 0.5:
                read_names.append(record.id)
                strands.append(columns[4])
                read_lengths.append(columns[1])
                start_positions.append(columns[7])
                end_positions.append(columns[8])
                chromosomes.append(columns[5])
                out_file.write(f">{record.id}\n{record.seq}\n") ##修改为直接写入文件，加快速度
            if len(chromosomes)%10000 == 0:
                print(len(chromosomes))

    print("Filter reads fit the standard, done")
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
    print("generate read info table, done")
    return metadata


def main(snakemake: "SnakemakeContext"):
    fasta_file = snakemake.input["fasta_aligned"]
    paf_file = snakemake.input["paf"]

    output_tsv_file = snakemake.output["tsv"]
    fasta = snakemake.output["fasta"]

    metadata = get_metadata(fasta_file,paf_file,fasta)

    print('Starting write in')
    ## output writing
    metadata.to_csv(output_tsv_file, sep="\t", index=False)
 
if __name__ == "__main__":
    main(snakemake)