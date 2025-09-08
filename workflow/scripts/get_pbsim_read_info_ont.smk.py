from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from snakemake_stub import *


import gzip
from Bio import AlignIO
import pandas as pd


def get_metadata(maf_file) -> pd.DataFrame:
    rows = []
    with gzip.open(maf_file, "rt") as f:
        for alignment in AlignIO.parse(f, "maf"):
            if len(alignment) > 2:
                raise ValueError(alignment)
            for record in alignment:
                seq_id = record.id
                if seq_id == "ref":
                    reference_start = record.annotations.get("start")
                    reference_end = record.annotations.get(
                        "start"
                    ) + record.annotations.get("size")
                else:
                    assert record.annotations.get("start") == 0
                    assert reference_start is not None and reference_end is not None
                    read_length = record.annotations.get("srcSize")
                    strand = record.annotations.get("strand")
                    strand = {1: "+", -1: "-"}[strand]

                    rows.append(
                        dict(
                            read_name=seq_id,
                            read_length=read_length,
                            reference_strand=strand,
                            reference_start=reference_start,
                            reference_end=reference_end,
                        )
                    )
                    reference_start, reference_end = None, None
    return pd.DataFrame(rows)


def main(snakemake: "SnakemakeContext"):
    input_file = snakemake.input["maf"]

    output_tsv_file = snakemake.output["tsv"]

    metadata = get_metadata(input_file)

    metadata.to_csv(output_tsv_file, sep="\t", index=False)


if __name__ == "__main__":
    main(snakemake)
