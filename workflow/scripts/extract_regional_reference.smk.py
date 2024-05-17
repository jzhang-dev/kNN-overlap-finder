import gzip
from Bio import SeqIO

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from snakemake_stub import *



def extract_region(fasta_file, chromosome, start, end) -> str:
    # Determine if the file is gzipped
    is_gzipped = fasta_file.endswith(".gz")

    # Open the FASTA file (handle gzipped and non-gzipped)
    if is_gzipped:
        open_func = gzip.open
        mode = "rt"  # Text mode
    else:
        open_func = open
        mode = "r"

    # Read the FASTA file and extract the region
    with open_func(fasta_file, mode) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            if record.id == chromosome:
                # Extract the region
                sequence = record.seq[start:end]
                sequence = str(sequence).upper()
                break
        else:
            raise ValueError()
    return sequence


def main(snakemake: "SnakemakeContext"):
    fasta_file = snakemake.input["fasta"]
    chromosome, start, end = snakemake.params["region"]

    # Extract the region
    sequence = extract_region(fasta_file, chromosome, start, end)

    with gzip.open(snakemake.output["fasta"], "wt") as f:
        f.write(f">Extracted_region:{chromosome}:{start}-{end}\n")
        f.write(sequence + "\n")  # # type: ignore


if __name__ == "__main__":
    main(snakemake)
