from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from snakemake_stub import *


import gzip
from Bio import AlignIO,SeqIO
import pandas as pd

def fasta_to_dict(fasta_file):  
    with gzip.open(fasta_file, "rt") as handle:  
        sequences = SeqIO.to_dict(SeqIO.parse(handle, "fasta"))  
        sequence_lengths = {key: len(value) for key, value in sequences.items()}  
        return sequence_lengths  
  

def get_metadata(maf_file,sequence_length) -> pd.DataFrame:
    rows = []
    with gzip.open(maf_file, "rt") as f:
        for alignment in AlignIO.parse(f, "maf"):
            if len(alignment) > 2:
                raise ValueError(alignment)
            for record in alignment:
                seq_id = record.id
                new_id = seq_id[:-1] + 'ccs'
                if seq_id == "ref":
                    reference_start = record.annotations.get("start")
                    reference_end = record.annotations.get(
                        "start"
                    ) + record.annotations.get("size")
                elif seq_id.endswith("/0") and new_id in sequence_length.keys():
                    assert record.annotations.get("start") == 0
                    assert reference_start is not None and reference_end is not None                        
                    read_length = sequence_length[new_id]
                    strand = record.annotations.get("strand")
                    strand = {1: "+", -1: "-"}[strand]
                    dict1 = {'read_name':new_id,"read_length":read_length,"reference_strand":strand, "reference_start":reference_start,"reference_end":reference_end}
                    rows.append(dict1) 
                    reference_start, reference_end = None, None
                else:
                    continue
    return pd.DataFrame(rows)


def main(snakemake: "SnakemakeContext"):
    fasta_file = snakemake.input['fasta']
    seq_length = fasta_to_dict(fasta_file)
    input_file = snakemake.input["maf"]
    output_tsv_file = snakemake.output["tsv"]
    metadata = get_metadata(input_file,seq_length)
    metadata.to_csv(output_tsv_file, sep="\t", index=False)

if __name__ == "__main__":
    main(snakemake)
