from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from snakemake_stub import *
import pandas as pd

def get_metadata(paf_gz_file) -> pd.DataFrame:
    read_names = []
    strands = []
    read_lengths = []
    start_positions = []
    end_positions = [] 
    chromosomes = []
    ## finding the alighnment query reads have max match base
    with open(paf_gz_file, "rt") as file:
        for row in file:  
            columns = row.strip().split('\t') 
            read_names.append(columns[0])
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
    return metadata


def main(snakemake: "SnakemakeContext"):
    paf_file = snakemake.input["paf"]
    output_tsv_file = snakemake.output["tsv"]

    metadata = get_metadata(paf_file)
    ## output writing
    metadata.to_csv(output_tsv_file, sep="\t", index=False)
 
if __name__ == "__main__":
    main(snakemake)