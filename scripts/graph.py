#!/usr/bin/python3
# -*- coding: utf-8 -*-

import collections
from typing import Type, Sequence, Mapping, Any, Collection, MutableMapping
from functools import cache
from dataclasses import dataclass
import gzip
import networkx as nx
from data_io import get_sibling_id, is_fwd_id, get_fwd_id
from align import _PairwiseAligner

class ReadGraph(nx.Graph):
    @classmethod
    def from_overlap_candidates(cls, candidates: Collection[tuple[int, int]]):
        read_graph = cls()
        read_graph.add_edges_from(candidates)
        return read_graph

    @classmethod
    def from_pairwise_alignment(
        cls,
        candidates: Collection[tuple[int, int]],
        alignment_dict,
        n_neighbors=6,
        *,
        min_alignment_score: int | None = 0,
    ):
        node_neighbors = collections.defaultdict(dict)

        for k1, k2 in candidates:
            score = alignment_dict[(k1, k2)].score
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
                if nbr_node == get_sibling_id(node):
                    # Skip self edges
                    continue
                graph.add_edge(node, nbr_node, alignment_score=score)
        return graph

    def align_edges(
        self,
        read_markers,
        marker_weights,
        aligner: Type[_PairwiseAligner],
        align_kw={},
        *,
        processes=4,
        _cache: MutableMapping | None = None,
        **kw,
    ):
        candidates = list(self.edges)
        alignment_dict = run_multiprocess_alignment(
            candidates=candidates,
            read_markers=read_markers,
            marker_weights=marker_weights,
            aligner=aligner,
            align_kw=align_kw,
            traceback=True,
            processes=processes,
            _cache=_cache,
            **kw,
        )
        return alignment_dict
