import os, gzip, pickle
import time
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse._csr import csr_matrix
from typing import Sequence, Type, Mapping
from dataclasses import dataclass, field
import numpy as np
from numpy import ndarray
import sharedmem

from nearest_neighbors import _NearestNeighbors
from dim_reduction import SpectralMatrixFree
from graph import OverlapGraph, get_overlap_statistics
from align import cWeightedSemiglobalAligner, run_multiprocess_alignment


@dataclass
class NearestNeighborsConfig:
    data: csr_matrix = field(repr=False)
    method: Type[_NearestNeighbors]
    description: str = ""
    binarize: bool = False
    tfidf: bool = False
    dim_reduction: int | None = None
    nearest_neighbor_kw: dict = field(default_factory=dict, repr=False)

    # def compute_nearest_neighbors(self, n_neighbors: int, *, verbose=True):
    #     data = self.data.copy()
    #     start_time = time.time()
    #     if self.binarize:
    #         if verbose:
    #             print("Binarization.")
    #         data[data > 0] = 1
    #     if self.tfidf:
    #         if verbose:
    #             print("TF-IDF transform.")
    #         data = TfidfTransformer(use_idf=True, smooth_idf=True).fit_transform(data)
    #     if self.dim_reduction is not None:
    #         if verbose:
    #             print("Dimension reduction.")
    #         reducer = SpectralMatrixFree(self.dim_reduction)
    #         reducer.fit(data)
    #         _, data = reducer.transform()
    #     self._neighbor_indices = self.method().get_neighbors(
    #         data, n_neighbors=n_neighbors, **self.nearest_neighbor_kw
    #     )
    #     elapsed_time = time.time() - start_time
    #     self._elapsed_time = elapsed_time
    #     if verbose:
    #         print(f"Elapsed time: {elapsed_time:.2f} s")

    def get_neighbors(
        self, data: csr_matrix, n_neighbors: int, *, verbose=True
    ) -> ndarray:
        _data: csr_matrix | ndarray = data.copy()
        start_time = time.time()

        if self.binarize:
            if verbose:
                print("Binarization.")
            _data[_data > 0] = 1

        if self.tfidf:
            if verbose:
                print("TF-IDF transform.")
            _data = TfidfTransformer(use_idf=True, smooth_idf=True).fit_transform(_data)

        if self.dim_reduction is not None:
            if verbose:
                print("Dimension reduction.")
            reducer = SpectralMatrixFree(self.dim_reduction)
            reducer.fit(_data)
            _, _data = reducer.transform()

        neighbor_indices = self.method().get_neighbors(
            _data, n_neighbors=n_neighbors, **self.nearest_neighbor_kw
        )
        elapsed_time = time.time() - start_time
        if verbose:
            print(f"Finished {self}. Elapsed time: {elapsed_time:.2f} s")
        return neighbor_indices

    # def _get_overlap_candidates(self, n_neighbors: int, read_ids, *, verbose=True):
    #     neighbor_indices = self._neighbor_indices
    #     if neighbor_indices is None:
    #         raise TypeError()
    #     if neighbor_indices.shape[1] < n_neighbors:
    #         raise ValueError("Not enough neighbors computed.")
    #     overlap_candidates = []

    #     for i1, row in enumerate(neighbor_indices):
    #         k1 = read_ids[i1]
    #         row = [i2 for i2 in row if i2 != i1 and i2 >= 0]
    #         overlap_candidates += [(k1, read_ids[i2]) for i2 in row[:n_neighbors]]

    #     return overlap_candidates

    # def get_overlap_graph(
    #     self,
    #     n_neighbors: int,
    #     read_ids,
    #     *,
    #     require_mutual_neighbors: bool = False,
    #     verbose=True,
    # ):
    #     overlap_candidates = self._get_overlap_candidates(
    #         n_neighbors=n_neighbors, read_ids=read_ids, verbose=verbose
    #     )
    #     graph = OverlapGraph.from_overlap_candidates(
    #         overlap_candidates, require_mutual_neighbors=require_mutual_neighbors
    #     )
    #     return graph


def mp_compute_nearest_neighbors(
    data: csr_matrix,
    configs: Sequence[NearestNeighborsConfig],
    n_neighbors: int,
    *,
    processes=4,
    verbose=True,
) -> Mapping[int, ndarray]:

    if verbose:
        print(f"Evaluating {len(configs)} configs using {processes} processes.")

    nbr_dict = {}
    with sharedmem.MapReduce(np=processes) as pool:

        def work(i):
            if verbose:
                print(i, end=" ")
            config = configs[i]
            neighbor_indices = config.get_neighbors(data=data, n_neighbors=n_neighbors, verbose=verbose)
            return i, neighbor_indices

        def reduce(i, neighbor_indices):
            nbr_dict[i] = neighbor_indices


        pool.map(work, range(len(configs)), reduce=reduce)

    if verbose:
        print("")

    return nbr_dict


# def mp_evaluate_configs(
#     configs: Sequence[NearestNeighborsConfig],
#     feature_matrix: csr_matrix,
#     read_features: Mapping[int, Sequence[int]],
#     feature_weights: Mapping[int, int],
#     reference_graph: OverlapGraph,
#     *,
#     processes=8,
#     verbose=True,
# ):
#     data = feature_matrix
#     configs = list(configs)
#     if verbose:
#         print(f"Evaluating {len(configs)} configs using {processes} processes.")

#     def print_stats(stats):
#         if stats is None:
#             raise TypeError()
#         print(
#             f"precision={stats['precision']:.3f}",
#             f"nr_precision={stats['nr_precision']:.3f}",
#             f"recall={stats['recall']:.3f}",
#             f"nr_recall={stats['nr_recall']:.3f}",
#             "\n",
#         )

#     # Nearest neighbors
#     with sharedmem.MapReduce(np=processes) as pool:

#         def work(i):
#             config = configs[i]
#             config.compute_pre_alignment_graph(
#                 data=data, reference_graph=reference_graph, read_ids=list(read_features)
#             )
#             return i, config

#         def reduce(i, config):
#             configs[i] = config
#             print(config)
#             print_stats(config.pre_align_stats)

#         pool.map(work, range(len(configs)), reduce=reduce)

#     return configs
