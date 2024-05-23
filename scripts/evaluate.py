import os, gzip, pickle
import time
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse._csr import csr_matrix
from typing import Sequence, Type, Mapping
from dataclasses import dataclass, field
import numpy as np

from nearest_neighbors import _NearestNeighbors
from dim_reduction import SpectralMatrixFree
from graph import ReadGraph, get_read_graph_statistics
from align import cWeightedSemiglobalAligner, run_multiprocess_alignment


@dataclass
class NearestNeighborsConfig:
    method: Type[_NearestNeighbors]
    use_tfidf: bool = False
    dim_reduction: int | None = None
    n_neighbors: int = 20
    nearest_neighbor_kw: dict = field(default_factory=dict, repr=False)
    elapsed_time: float | None = None
    require_mutual_neighbors: bool = False
    pre_align_graph: ReadGraph | None = field(default=None, repr=False)
    pre_align_stats: dict | None = None
    post_align_intersect: bool = False
    post_align_graph: ReadGraph | None = field(default=None, repr=False)
    post_align_stats: dict | None = None

    def _get_overlap_candidates(self, data, read_ids):
        start_time = time.time()
        if self.use_tfidf:
            data = TfidfTransformer(use_idf=True, smooth_idf=True).fit_transform(data)
        if self.dim_reduction is not None:
            reducer = SpectralMatrixFree(self.dim_reduction)
            reducer.fit(data)
            _, data = reducer.transform()
        neighbor_indices = self.method(data).get_neighbors(
            n_neighbors=self.n_neighbors, **self.nearest_neighbor_kw
        )
        elapsed_time = time.time() - start_time

        overlap_candidates = []
        for i1, row in enumerate(neighbor_indices):
            k1 = read_ids[i1]
            row = [i2 for i2 in row if i2 != i1 and i2 >= 0]
            overlap_candidates += [(k1, read_ids[i2]) for i2 in row[: self.n_neighbors]]

        return overlap_candidates, elapsed_time

    def compute_pre_alignment_graph(
        self, data, read_ids, reference_graph: ReadGraph, *, intersect: bool = False
    ):
        overlap_candidates, elapsed_time = self._get_overlap_candidates(data, read_ids)
        self.elapsed_time = elapsed_time
        graph = ReadGraph.from_overlap_candidates(
            overlap_candidates, require_mutual_neighbors=intersect
        )
        self.pre_align_graph = graph
        result = get_read_graph_statistics(graph, reference_graph=reference_graph)
        self.pre_align_stats = result

    def compute_post_alignment_graph(
        self,
        read_features,
        feature_weights,
        reference_graph: ReadGraph,
        *,
        n_neighbors=6,
        min_alignment_score=0,
        processes=8,
        batch_size=200,
        _cache: dict = {},
        **kw,
    ):
        if self.pre_align_graph is None:
            raise TypeError()

        alignment_dict = self.pre_align_graph.align_edges(
            read_features=read_features,
            feature_weights=feature_weights,
            aligner=cWeightedSemiglobalAligner,
            align_kw=dict(max_cells=int(1e10)),
            traceback=False,
            processes=processes,
            batch_size=batch_size,
            min_free_memory_gb=48,
            max_total_wait_seconds=600,
            _cache=_cache,
            **kw,
        )

        graph = ReadGraph.from_pairwise_alignment(
            alignment_dict,
            n_neighbors=n_neighbors,
            min_alignment_score=min_alignment_score,
        )
        self.post_align_graph = graph
        result = get_read_graph_statistics(graph, reference_graph=reference_graph)
        self.post_align_stats = result


def mp_evaluate_configs(
    configs: Sequence[NearestNeighborsConfig],
    feature_matrix: csr_matrix,
    read_features: Mapping[int, Sequence[int]],
    feature_weights: Mapping[int, int],
    reference_graph: ReadGraph,
    *,
    post_align_n_neighbors=6,
    processes=8,
    batch_size=200,
    alignment_pickle_path=None,
):
    data = feature_matrix
    pickle_file = alignment_pickle_path

    if pickle_file is not None and os.path.isfile(pickle_file):
        with gzip.open(pickle_file, "rb") as f:
            alignment_dict = pickle.load(f)  # type:ignore
    else:
        alignment_dict = {}
    initial_alignment_count = len(alignment_dict)

    def print_stats(stats):
        if stats is None:
            raise TypeError()
        print(
            f"precision={stats['precision']:.3f}",
            f"nr_precision={stats['nr_precision']:.3f}",
            f"recall={stats['recall']:.3f}",
            f"nr_recall={stats['nr_recall']:.3f}",
            "\n",
        )

    config_list = configs
    for config in config_list:
        print(config)
        config.compute_pre_alignment_graph(
            data=data, reference_graph=reference_graph, read_ids=list(read_features)
        )

        print("Pre-alignment:")
        print_stats(config.pre_align_stats)

        config.compute_post_alignment_graph(
            reference_graph=reference_graph,
            _cache=alignment_dict,
            feature_weights=feature_weights,
            read_features=read_features,
            n_neighbors=post_align_n_neighbors,
            min_alignment_score=0,
            processes=processes,
            batch_size=batch_size,
        )

        print("Post-alignment:")
        print_stats(config.post_align_stats)

    if pickle_file is not None and len(alignment_dict) > initial_alignment_count:
        with gzip.open(pickle_file, "wb") as f:
            pickle.dump(alignment_dict, f)  # type: ignore
