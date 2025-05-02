#!/usr/bin/python3
# -*- coding: utf-8 -*-

import collections
from typing import Type, Sequence, Mapping, Any, Collection, MutableMapping
from dataclasses import dataclass
import numpy as np
from numpy import ndarray
import networkx as nx
from intervaltree import Interval, IntervalTree

from data_io import get_sibling_id, get_fwd_id, parse_paf_file
from align import _PairwiseAligner, run_multiprocess_alignment, AlignmentResult


@dataclass
class GenomicInterval:
    chromosome: set
    start: int
    end: int


def get_interval_trees(
    read_intervals: Mapping[int, Collection[GenomicInterval]]
) -> dict[str, IntervalTree]:
    tree_dict = collections.defaultdict(IntervalTree)
    for read, intervals in read_intervals.items():
        for intv in intervals:
            tree = tree_dict[intv.chromosome]
            tree.addi(intv.start, intv.end, read)
    return tree_dict


def get_overlap_candidates(
    neighbor_indices: ndarray,
    n_neighbors: int,
    read_ids: Sequence[int],
):
    # if neighbor_indices.shape[1] < n_neighbors:
    #     raise ValueError("Not enough neighbors in `neighbor_indices`.")
    _read_ids = np.array(read_ids)
    overlap_candidates = []
    for i1, row in enumerate(neighbor_indices):
        k1 = _read_ids[row[0]]
        row = row[(row >= 0) & (row != row[0])]
        for neighbor_order, i2 in enumerate(row[:n_neighbors]):
            overlap_candidates.append((k1, _read_ids[i2], {"neighbor_order": neighbor_order}))
    return overlap_candidates


class OverlapGraph(nx.Graph):

    @classmethod
    def from_overlap_candidates(
        cls,
        candidates: Collection[tuple[int, int, dict]],
        require_mutual_neighbors: bool = False,
    ):
        if require_mutual_neighbors:
            candidates = set((k1, k2, data) for k1, k2, data in candidates)
            removed = set()
            for k1, k2, data in candidates:
                if (k2, k1, data) not in candidates:
                    removed.add((k1, k2, data))
            candidates -= removed

        read_graph = cls()
        read_graph.add_edges_from(candidates)
        return read_graph

    @classmethod
    def from_neighbor_indices(
        cls,
        neighbor_indices: ndarray,
        n_neighbors: int,
        read_ids: Sequence[int],
        *,
        require_mutual_neighbors: bool = False,
    ):
        overlap_candidates = get_overlap_candidates(
            neighbor_indices=neighbor_indices,
            n_neighbors=n_neighbors,
            read_ids=read_ids,
        )
        return cls.from_overlap_candidates(
            candidates=overlap_candidates,
            require_mutual_neighbors=require_mutual_neighbors,
        )

    @classmethod
    def from_pairwise_alignment(
        cls,
        alignment_dict: Mapping[tuple[int, int], AlignmentResult],
        *,
        n_neighbors=6,
        min_alignment_score: int | None = 0,
    ):
        node_neighbors = collections.defaultdict(dict)
        for (k1, k2), result in alignment_dict.items():
            score = result.score
            node_neighbors[k1][k2] = score
            node_neighbors[k2][k1] = score

        graph = cls()
        for node, neighbors in node_neighbors.items():
            graph.add_node(node)
            nbr_scores = list(sorted(neighbors.values(), reverse=True))
            if min_alignment_score is not None:
                nbr_scores = [x for x in nbr_scores if x >= min_alignment_score]
            if not nbr_scores:
                continue
            min_score = nbr_scores[:n_neighbors][-1]
            nearest_neighbors = {
                x: score for x, score in neighbors.items() if score > min_score
            }

            for nbr_node, score in nearest_neighbors.items():
                if nbr_node == node or nbr_node == get_sibling_id(node):
                    # Skip self edges
                    continue
                graph.add_edge(node, nbr_node, alignment_score=score)
        return graph

    @classmethod
    def from_intervals(cls, read_intervals: Mapping[int, Collection[GenomicInterval]]):
        # Find all overlaps
        trees = get_interval_trees(read_intervals=read_intervals)
        graph = cls()
        contained_reads = set()
        for read_0, intervals in read_intervals.items():
            parent_reads = set()
            for intv in intervals:
                tree = trees[intv.chromosome]
                start_0 = intv.start
                end_0 = intv.end
                for intv_1 in tree.overlap(start_0, end_0):
                    read_1 = intv_1.data
                    if read_1 == read_0:
                        continue
                    start_1, end_1 = intv_1.begin, intv_1.end
                    if start_1 < start_0 and end_1 > end_0:
                        parent_reads.add(read_1)
                    if graph.has_edge(read_0, read_1):
                        continue
                    overlap_size = max(0, min(end_0, end_1) - max(start_0, start_1))
                    graph.add_edge(
                        read_0,
                        read_1,
                        overlap_size=overlap_size
                    )
            if len(parent_reads) == 1:
                contained_reads.add(read_0)

        # Label contained reads
        nx.set_node_attributes(graph, "contained", False)
        for read in contained_reads:
            graph.nodes[read]["contained"] = True

        return graph

    def align_edges(
        self,
        read_features,
        feature_weights,
        aligner: Type[_PairwiseAligner],
        traceback: bool = True,
        align_kw={},
        *,
        processes=4,
        _cache: MutableMapping | None = None,
        **kw,
    ):
        candidates = list(self.edges)
        alignment_dict = run_multiprocess_alignment(
            candidates=candidates,
            read_features=read_features,
            feature_weights=feature_weights,
            aligner=aligner,
            align_kw=align_kw,
            traceback=traceback,
            processes=processes,
            _cache=_cache,
            **kw,
        )
        return alignment_dict


def remove_false_edges(graph, reference_graph):
    false_edges = []
    for u, v in graph.edges:
        if not reference_graph.has_edge(u, v):
            false_edges.append((u, v))
    graph.remove_edges_from(false_edges)


