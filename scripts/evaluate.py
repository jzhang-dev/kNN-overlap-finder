import os, gzip, pickle
import time
from sklearn.feature_extraction.text import TfidfTransformer
import scipy
from scipy.sparse._csr import csr_matrix
from typing import Sequence, Type, Mapping,Literal
from dataclasses import dataclass, field
import numpy as np
from numpy import ndarray
import sharedmem
import re

from nearest_neighbors import _NearestNeighbors, ExactNearestNeighbors
from dim_reduction import _DimensionReduction, SpectralEmbedding, scBiMapEmbedding
from graph import OverlapGraph, get_overlap_statistics
from align import cWeightedSemiglobalAligner, run_multiprocess_alignment

def manual_tf(data:csr_matrix):
    row_sums = np.sum(data, axis=1)
    tf_data = data / row_sums.reshape(-1, 1)
    tf_data = csr_matrix(tf_data) 
    return tf_data

def manual_idf(data:csr_matrix):
    binary_matrix = csr_matrix((np.ones_like(data.data), data.indices, data.indptr), shape=data.shape)
    col_sums = binary_matrix.sum(axis=0).A1
    N = binary_matrix.shape[0]
    idf = np.log((N + 1) / (col_sums + 1)) + 1 
    idf = idf.astype(binary_matrix.dtype)
    _data = binary_matrix.multiply(idf) 
    return _data

@dataclass
class NearestNeighborsConfig:
    description: str = ""
    tfidf: Literal["TF","IDF","TF-IDF",'None'] = 'None',
    dimension_reduction_method: Type[_DimensionReduction] | None = None
    dimension_reduction_kw: dict = field(default_factory=dict, repr=False)
    nearest_neighbors_method: Type[_NearestNeighbors] = ExactNearestNeighbors
    nearest_neighbors_kw: dict = field(default_factory=dict, repr=False)

    def get_neighbors(
        self, data: csr_matrix, n_neighbors: int, *, sample_query=False, verbose=True
    ) -> tuple[ndarray, Mapping[str, float], Mapping[str, float]]:
        elapsed_time = {}
        peak_memory = {} # TODO

        _data: csr_matrix | ndarray = data.copy()
        start_time = time.time()

        if self.tfidf == 'TF':
            print("TF transform.")
            prepro_data = manual_tf(_data)
        elif self.tfidf == 'binary':
            prepro_data = csr_matrix((np.ones_like(_data.data), _data.indices, _data.indptr), shape=_data.shape)
        elif self.tfidf == 'count':
            prepro_data = _data
        elif self.tfidf == 'IDF':
            print("manually IDF transform.")
            _data = csr_matrix((np.ones_like(_data.data), _data.indices, _data.indptr), shape=_data.shape)
            prepro_data = manual_idf(_data)
        elif self.tfidf == 'TF-IDF':
            print("TF-IDF transform.") 
            _data = manual_tf(_data)
            prepro_data = manual_idf(_data)
        else:
            raise ValueError('Invalid preprocess method.')
        elapsed_time['tfidf'] = time.time() - start_time

        if self.dimension_reduction_method is not None:
            if verbose:
                print("Dimension reduction.")
            start_time = time.time()
            _data = self.dimension_reduction_method().transform(prepro_data, **self.dimension_reduction_kw)
            elapsed_time['dimension_reduction'] = time.time() - start_time
            if scipy.sparse.issparse(_data):
                _data = _data.toarray()
        
        if verbose:
            print("Nearest neighbors.")
        start_time = time.time()

        neighbor_indices = self.nearest_neighbors_method().get_neighbors(
            _data, n_neighbors=n_neighbors, **self.nearest_neighbors_kw
        )
        elapsed_time['nearest_neighbors'] = time.time() - start_time
        if verbose:
            print(f"Finished {self}. Elapsed time: {elapsed_time}. Peak memory: {peak_memory}")
        return neighbor_indices, elapsed_time



def mp_compute_nearest_neighbors(
    data: csr_matrix,
    configs: Sequence[NearestNeighborsConfig],
    n_neighbors: int,
    *,
    processes=4,
    verbose=True,
) -> tuple[Mapping[int, ndarray], Mapping, Mapping]:

    if verbose:
        print(f"Evaluating {len(configs)} configs using {processes} processes.")

    nbr_dict = {}
    time_dict = {}
    memory_dict = {}
    with sharedmem.MapReduce(np=processes) as pool:

        def work(i):
            if verbose:
                print(i, end=" ")
            config = configs[i]
            result = config.get_neighbors(data=data, n_neighbors=n_neighbors, verbose=verbose)
            return i, result

        def reduce(i, result):
            neighbor_indices, elapsed_time, peak_memory = result
            nbr_dict[i] = neighbor_indices
            time_dict[i] = elapsed_time
            memory_dict[i] = peak_memory


        pool.map(work, range(len(configs)), reduce=reduce)

    if verbose:
        print("")

    return nbr_dict, time_dict, memory_dict


def compute_nearest_neighbors(
    data: csr_matrix,
    config: NearestNeighborsConfig,
    n_neighbors: int,
    *,
    verbose=True,
) -> tuple[ndarray,ndarray,ndarray]:
    
    neighbor_indices, elapsed_time = config.get_neighbors(data=data, n_neighbors=n_neighbors, verbose=verbose)

    return neighbor_indices, elapsed_time
