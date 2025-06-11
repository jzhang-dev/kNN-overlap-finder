import sys,pickle
import pandas as pd

sys.path.append("scripts")
sys.path.append("../../scripts")

from graph_for_chrs import OverlapGraph, GenomicInterval

tsv_path = snakemake.input['metadata']
ref_graph_path = snakemake.output['ref_graph']

meta_df = pd.read_table(tsv_path)
print(meta_df)

def get_read_intervals(meta_df):
    read_intervals = {
        i: [GenomicInterval((chromosome,strand), start, end)]
        for i, chromosome, strand, start, end in zip(
            meta_df.index,
            meta_df["reference_chromosome"],
            meta_df["reference_strand"],
            meta_df["reference_start"],
            meta_df["reference_end"],
        )
    }
    return read_intervals

read_intervals = get_read_intervals(meta_df)
print('read_intervals generation done!')
reference_graph = OverlapGraph.from_intervals(read_intervals) 
print('reference_graph generation done')
print('start writing in file')
with open(ref_graph_path, "wb") as f:
    pickle.dump(reference_graph, f)