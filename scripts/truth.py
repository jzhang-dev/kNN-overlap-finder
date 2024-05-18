#!/usr/bin/python3
# -*- coding: utf-8 -*-

import collections
import pandas as pd
import pysam
from intervaltree import Interval, IntervalTree

from data_io import get_fwd_id, get_sibling_id  # type: ignore


def load_alignment_intervals(bam_path, read_indices, *, region=None):
    # Load read coverage intervals from BAM file
    interval_dict = collections.defaultdict(list)
    for segment in pysam.AlignmentFile(bam_path, "rb").fetch(region=region):
        read_name = segment.query_name
        read_id = read_indices[read_name]
        reference_name = segment.reference_name
        if segment.is_secondary:
            continue
        interval_dict[read_id].append(
            (
                reference_name,
                segment.reference_start,
                segment.reference_end,
            )
        )
    return interval_dict



def get_interval_trees(interval_dict) -> dict[str, IntervalTree]:
    tree_dict = collections.defaultdict(IntervalTree)
    for read, intervals in interval_dict.items():
        for intv in intervals:
            reference_name, start, end = intv
            tree = tree_dict[reference_name]
            tree.addi(start, end, read)
    return tree_dict


def get_overlap_size(interval_1, interval_2):
    start1, end1 = interval_1
    start2, end2 = interval_2
    overlap_size = max(0, min(end1, end2) - max(start1, start2))
    return overlap_size


def get_overlaps(interval_dict):
    tree_dict = get_interval_trees(interval_dict)
    overlap_dict = collections.defaultdict(
        collections.Counter
    )  # read_1 -> {read_2 -> overlap_size}
    i = 0
    for read_0, intervals in interval_dict.items():
        for reference_name, start_0, end_0 in intervals:
            tree = tree_dict[reference_name]
            for intv_1 in tree.overlap(start_0, end_0):
                read_1 = intv_1.data
                if read_1 in overlap_dict[read_0]:
                    continue
                start_1, end_1 = intv_1.begin, intv_1.end
                overlap_size = get_overlap_size((start_0, end_0), (start_1, end_1))
                overlap_dict[read_0][read_1] += overlap_size
                overlap_dict[read_1][read_0] += overlap_size

        if i % 1000 == 0:
            print(i, end=" ")
        i += 1
    return overlap_dict


def get_neighbors(overlap_dict, read):
    nbr_dict = overlap_dict[read]
    neighbors = list(nbr_dict)
    neighbors.sort(key=lambda x: nbr_dict[x], reverse=True)
    return neighbors


def get_read_graph_statistics(read_graph, overlap_dict, *, n_neighbors=6):
    nn_dict = {}

    def get_nn(read) -> set[int]:
        nonlocal nn_dict
        if read in nn_dict:
            nn = nn_dict[read]
        else:
            nn = get_neighbors(overlap_dict, read)[:n_neighbors]
            nn = set(nn)
            nn_dict[read] = nn
        return nn

    # Precision
    ## precision = true overlaps / detected edges
    ## k-precision = true nearest neighbors / detected edges
    ## NR-precision = non-redundant edges / detected edges
    false_positive_edges = set() 
    k_false_positive_edges = set() 
    for read_1, read_2 in read_graph.edges:
        read_1, read_2 = get_fwd_id(read_1), get_fwd_id(read_2)
        if overlap_dict[read_1][read_2] <= 0:
            false_positive_edges.add((read_1, read_2))
        if read_2 not in get_nn(read_1) and read_1 not in get_nn(read_2):
            k_false_positive_edges.add((read_1, read_2))
    precision = 1 - len(false_positive_edges) / len(read_graph.edges)
    k_precision = 1 - len(k_false_positive_edges) / len(read_graph.edges)

    # Recall
    ## recall = detected overlaps / all overlaps
    ## k-recall = detected nearest neighbors / all nearest neighbors
    ## NR-recall = detected non-redundant edges / all non-redundant edges
    false_negative_edges = set() # 
    neighbor_count = 0
    for read_1 in read_graph.nodes:
        neighbors = get_nn(read_1)
        neighbor_count += len(neighbors)
        for read_2 in neighbors:
            if not read_graph.has_edge(read_1, read_2) and not read_graph.has_edge(
                read_1, get_sibling_id(read_2)
            ):
                false_negative_edges.add((read_1, read_2))
    recall = 1 - len(false_negative_edges) / neighbor_count

    result = dict(
        precision=precision,
        k_precision=k_precision,
        false_positive_edges=false_positive_edges,
        recall=recall,
        false_negative_edges=false_negative_edges,
    )
    return result
