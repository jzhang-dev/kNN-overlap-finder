import sys,pickle
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("scripts")
sys.path.append("../../scripts")

from graph import OverlapGraph, GenomicInterval, get_overlap_statistics, remove_false_edges,get_neighbor_overlap_bases,get_precision

tsv_path = snakemake.input['metadata']
ref_graph_path = snakemake.output['ref_graph']

meta_df = pd.read_table(tsv_path)


def get_read_intervals(meta_df):
    read_intervals = {
        i: [GenomicInterval(strand, start, end)]
        for i, strand, start, end in zip(
            meta_df.index,
            meta_df["reference_strand"],
            meta_df["reference_start"],
            meta_df["reference_end"],
        )
    }
    return read_intervals

read_intervals = get_read_intervals(meta_df)

reference_graph = OverlapGraph.from_intervals(read_intervals)
with open(ref_graph_path, "wb") as f:
    pickle.dump(reference_graph, f)