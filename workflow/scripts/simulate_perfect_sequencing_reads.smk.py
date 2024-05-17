import gzip
from Bio import SeqIO
import numpy as np


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from snakemake_stub import *


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


def simulate_sequencing(fasta_file, read_length, depth, output_file, seed):
    # Read the reference genome
    with gzip.open(fasta_file, "rt") as handle:
        genome = SeqIO.read(handle, "fasta")

    genome_length = len(genome.seq)
    reads = {}

    if genome_length < read_length:
        raise ValueError()

    num_reads = round(genome_length * depth / read_length)
    rng = np.random.default_rng(seed=seed)

    genome_sequence = str(genome.seq)
    for _ in range(num_reads):
        start_pos = rng.integers(0, genome_length - read_length)
        read_seq = genome_sequence[start_pos : start_pos + read_length]
        strand = rng.choice(["F", "R"])
        if strand == "R":
            read_seq = reverse_complement(read_seq)
        reads[(start_pos, strand)] = read_seq

    # Write the reads to the output file in FASTA format
    with gzip.open(output_file, "wt") as out_handle:
        for (start, strand), sequence in reads.items():
            out_handle.write(f">{start}_{strand}\n{sequence}\n")  # type: ignore


def main(snakemake: "SnakemakeContext"):
    input_file = snakemake.input["fasta"]
    output_file = snakemake.output["fasta"]
    depth = snakemake.params["depth"]
    read_length = snakemake.params["length_kb"] * 1000
    seed = snakemake.params["seed"]

    simulate_sequencing(
        fasta_file=input_file,
        read_length=read_length,
        depth=depth,
        output_file=output_file,
        seed=seed,
    )


if __name__ == "__main__":
    main(snakemake)
