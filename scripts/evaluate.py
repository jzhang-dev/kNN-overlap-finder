import os, gzip, pickle
import time
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse._csr import csr_matrix
from typing import Sequence, Type, Mapping
from dataclasses import dataclass, field
import numpy as np
import sharedmem

from nearest_neighbors import _NearestNeighbors
from dim_reduction import SpectralMatrixFree
from graph import OverlapGraph, get_overlap_statistics
from align import cWeightedSemiglobalAligner, run_multiprocess_alignment


@dataclass
class NearestNeighborsConfig:
    data: csr_matrix = field(repr=False)
    method: Type[_NearestNeighbors]
    use_tfidf: bool = False
    dim_reduction: int | None = None
    nearest_neighbor_kw: dict = field(default_factory=dict, repr=True)
    _neighbor_indices: np.ndarray | None = field(default=None, repr=False)
    _elapsed_time: float | None = None

    def compute_nearest_neighbors(self, n_neighbors: int, *, verbose=True):
        data = self.data
        start_time = time.time()
        if self.use_tfidf:
            if verbose:
                print("TF-IDF transform.")
            data = TfidfTransformer(use_idf=True, smooth_idf=True).fit_transform(data)
        if self.dim_reduction is not None:
            if verbose:
                print("Dimension reduction.")
            reducer = SpectralMatrixFree(self.dim_reduction)
            reducer.fit(data)
            _, data = reducer.transform()
        self._neighbor_indices = self.method(data).get_neighbors(
            n_neighbors=n_neighbors, **self.nearest_neighbor_kw
        )
        elapsed_time = time.time() - start_time
        self._elapsed_time = elapsed_time
        if verbose:
            print(f"Elapsed time: {elapsed_time:.2f} s")

    def _get_overlap_candidates(self, n_neighbors: int, read_ids, *, verbose=True):
        neighbor_indices = self._neighbor_indices
        if neighbor_indices is None:
            raise TypeError()
        if neighbor_indices.shape[1] < n_neighbors:
            raise ValueError("Not enough neighbors computed.")
        overlap_candidates = []

        for i1, row in enumerate(neighbor_indices):
            k1 = read_ids[i1]
            row = [i2 for i2 in row if i2 != i1 and i2 >= 0]
            overlap_candidates += [(k1, read_ids[i2]) for i2 in row[:n_neighbors]]

        return overlap_candidates

    def get_overlap_graph(
        self, n_neighbors: int, read_ids, *, require_mutual_neighbors: bool = False, verbose=True
    ):
        overlap_candidates = self._get_overlap_candidates(
            n_neighbors=n_neighbors, read_ids=read_ids, verbose=verbose
        )
        graph = OverlapGraph.from_overlap_candidates(
            overlap_candidates, require_mutual_neighbors=require_mutual_neighbors
        )
        return graph


def mp_evaluate_configs(
    configs: Sequence[NearestNeighborsConfig],
    feature_matrix: csr_matrix,
    read_features: Mapping[int, Sequence[int]],
    feature_weights: Mapping[int, int],
    reference_graph: OverlapGraph,
    *,
    processes=8,
    verbose=True,
):
    data = feature_matrix
    configs = list(configs)
    if verbose:
        print(f"Evaluating {len(configs)} configs using {processes} processes.")

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

    # Nearest neighbors
    with sharedmem.MapReduce(np=processes) as pool:

        def work(i):
            config = configs[i]
            config.compute_pre_alignment_graph(
                data=data, reference_graph=reference_graph, read_ids=list(read_features)
            )
            return i, config

        def reduce(i, config):
            configs[i] = config
            print(config)
            print("Pre-alignment:")
            print_stats(config.pre_align_stats)

        pool.map(work, range(len(configs)), reduce=reduce)

    # Pairwise alignment
    if not pairwise_alignment:
        return configs

    if pickle_file is not None and os.path.isfile(pickle_file):
        with gzip.open(pickle_file, "rb") as f:
            alignment_dict = pickle.load(f)  # type:ignore
    else:
        alignment_dict = {}
    initial_alignment_count = len(alignment_dict)

    for config in configs:
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
        print(config)
        print("Post-alignment:")
        print_stats(config.post_align_stats)

    if pickle_file is not None and len(alignment_dict) > initial_alignment_count:
        with gzip.open(pickle_file, "wb") as f:
            pickle.dump(alignment_dict, f)  # type: ignore

    return configs
