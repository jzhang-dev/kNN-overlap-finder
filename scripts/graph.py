#!/usr/bin/python3
# -*- coding: utf-8 -*-

import collections
from typing import Type, Sequence, Mapping, Any, Collection, MutableMapping
from dataclasses import dataclass
import numpy as np
import networkx as nx
from intervaltree import Interval, IntervalTree


from data_io import get_sibling_id, get_fwd_id, parse_paf_file
from align import _PairwiseAligner, run_multiprocess_alignment, AlignmentResult


@dataclass
class GenomicInterval:
    chromosome: str
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


class OverlapGraph(nx.Graph):

    @classmethod
    def from_overlap_candidates(
        cls,
        candidates: Collection[tuple[int, int]],
        require_mutual_neighbors: bool = False,
    ):
        if require_mutual_neighbors:
            candidates = set(candidates)
            removed = set()
            for k1, k2 in candidates:
                if (k2, k1) not in candidates:
                    removed.add((k1, k2))
            candidates -= removed

        read_graph = cls()
        read_graph.add_edges_from(candidates)
        return read_graph

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
        trees = get_interval_trees(read_intervals=read_intervals)
        graph = cls()
        for read_0, intervals in read_intervals.items():
            for intv in intervals:
                tree = trees[intv.chromosome]
                start_0 = intv.start
                end_0 = intv.end
                for intv_1 in tree.overlap(start_0, end_0):
                    read_1 = intv_1.data
                    if read_1 == read_0:
                        continue
                    if graph.has_edge(read_0, read_1):
                        continue
                    start_1, end_1 = intv_1.begin, intv_1.end
                    overlap_size = max(0, min(end_0, end_1) - max(start_0, start_1))
                    left_overhang_size = abs(start_0 - start_1)
                    left_overhang_node = read_0 if start_0 <= start_1 else read_1
                    right_overhang_size = abs(end_0 - end_1)
                    right_overhang_node = read_0 if end_0 >= end_1 else read_1
                    graph.add_edge(
                        read_0,
                        read_1,
                        overlap_size=overlap_size,
                        left_overhang_size=left_overhang_size,
                        left_overhang_node=left_overhang_node,
                        right_overhang_size=right_overhang_size,
                        right_overhang_node=right_overhang_node,
                        redundant=True,
                    )
        for node_0 in graph.nodes():
            nearest_left_node = None
            min_left_overhang = float("inf")
            nearest_right_node = None
            min_right_overhang = float("inf")
            for node_1, data in graph[node_0].items():
                if (
                    data["left_overhang_node"] == node_1
                    and data["left_overhang_size"] < min_left_overhang
                ):
                    min_left_overhang = data["left_overhang_size"]
                    nearest_left_node = node_1
                if (
                    data["right_overhang_node"] == node_1
                    and data["right_overhang_size"] < min_right_overhang
                ):
                    min_right_overhang = data["right_overhang_size"]
                    nearest_right_node = node_1
            if nearest_left_node is not None:
                graph[node_0][nearest_left_node]["redundant"] = False
            if nearest_right_node is not None:
                graph[node_0][nearest_right_node]["redundant"] = False

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


def get_overlap_statistics(query_graph: OverlapGraph, reference_graph: OverlapGraph):
    reference_edges = set(
        tuple(sorted((node_1, node_2))) for node_1, node_2 in reference_graph.edges
    )
    nr_reference_edges = set(
        tuple(sorted((node_1, node_2)))
        for node_1, node_2, data in reference_graph.edges(data=True)
        if not data["redundant"]
    )
    query_edges = set(
        tuple(sorted((read_1, read_2))) for read_1, read_2 in query_graph.edges
    )
    true_positive_edges = query_edges & reference_edges
    nr_true_positive_edges = query_edges & nr_reference_edges

    recall = len(true_positive_edges) / len(reference_edges)
    nr_recall = len(nr_true_positive_edges) / len(nr_reference_edges)

    precision = len(true_positive_edges) / len(query_edges)
    nr_precision = len(nr_true_positive_edges) / len(query_edges)

    singleton_count = len([node for node in query_graph if len(query_graph[node]) <= 1])
    component_sizes = [len(x) for x in nx.connected_components(query_graph)]
    component_sizes.sort(reverse=True)
    component_sizes = np.array(component_sizes)
    node_count = len(query_graph.nodes)
    N50 = component_sizes[component_sizes.cumsum() >= node_count * 0.5].max()

    result = dict(
        precision=precision,
        nr_precision=nr_precision,
        recall=recall,
        nr_recall=nr_recall,
        singleton_count=singleton_count,
        N50=N50,
    )
    return result
