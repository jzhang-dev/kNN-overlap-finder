from dataclasses import dataclass, field
import mmh3
from functools import lru_cache
import collections
from typing import Sequence, Type, Mapping, Iterable, Literal
from scipy import sparse
from scipy.sparse._csr import csr_matrix
import numpy as np
from numpy import matlib, ndarray
from numpy.typing import NDArray
import sklearn.neighbors
import pynndescent
import hnswlib
import faiss
from numba import njit, prange
from itertools import chain 
from collections import Counter
import secrets
import random
import pynear
import os, gzip, pickle
import time
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse._csr import csr_matrix
from typing import Sequence, Type, Mapping,Literal
from dataclasses import dataclass, field
import numpy as np
from numpy import ndarray
import sharedmem

from nearest_neighbors import _NearestNeighbors, ExactNearestNeighbors
from dim_reduction import _DimensionReduction, SpectralEmbedding, scBiMapEmbedding

from data_io import parse_paf_file, get_sibling_id
def hamming_distance(x, y):  
    return np.count_nonzero(x != y)

from nearest_neighbors import _NearestNeighbors, ExactNearestNeighbors
from dim_reduction import _DimensionReduction, SpectralEmbedding, scBiMapEmbedding

@dataclass
class _NearestNeighbors:
    def get_neighbors(
        self, ref: csr_matrix | np.ndarray, que: csr_matrix | np.ndarray, n_neighbors: int
    ) -> np.ndarray:
        raise NotImplementedError()

class ExactNearestNeighbors(_NearestNeighbors):
    def get_neighbors(
        self, ref: csr_matrix | np.ndarray, que: csr_matrix | np.ndarray, metric="cosine", n_neighbors: int = 20
    ):
        
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=1, algorithm='auto',metric=metric).fit(ref)  
        _, nbr_indices = nbrs.kneighbors(que)
        return nbr_indices


class HNSW(_NearestNeighbors):
    def get_neighbors(
        self,
        ref: csr_matrix | np.ndarray,
        que: csr_matrix | np.ndarray,
        n_neighbors: int,
        metric: Literal["euclidean", "cosine"] = "euclidean",
        *,
        threads: int | None = None,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
    ) -> np.ndarray:
        if metric == "euclidean":
            space = "l2"
        else:
            space = metric

        # Initialize the HNSW index
        p = hnswlib.Index(space=space, dim=ref.shape[1])
        if threads is not None:
            p.set_num_threads(threads)
        p.init_index(max_elements=ref.shape[0], ef_construction=ef_construction, M=M)
        ids = np.arange(ref.shape[0])
        p.add_items(ref, ids)
        p.set_ef(ef_search)
        nbr_indices, _ = p.knn_query(que, k=1)
        return nbr_indices

@dataclass
class NearestNeighborsConfig:
    description: str = ""
    tfidf: Literal["TF","IDF","TF-IDF",'None'] = 'None',
    dimension_reduction_method: Type[_DimensionReduction] | None = None
    dimension_reduction_kw: dict = field(default_factory=dict, repr=False)
    nearest_neighbors_method: Type[_NearestNeighbors] = ExactNearestNeighbors
    nearest_neighbors_kw: dict = field(default_factory=dict, repr=False)

    def get_neighbors(
        self, ref: csr_matrix | np.ndarray, que: csr_matrix | np.ndarray, n_neighbors: int, *, verbose=True
    ) -> tuple[ndarray, Mapping[str, float], Mapping[str, float]]:
        elapsed_time = {}
        peak_memory = {} # TODO
        data = sparse.vstack([ref,que])
        _data: csr_matrix | ndarray = data.copy()
        
        if self.tfidf == 'TF':
            if verbose:
                print("TF transform.")
        elif self.tfidf == 'IDF':
            start_time = time.time()
            if verbose:
                print("IDF transform.")
            _data[_data > 0] = 1
            _data = TfidfTransformer(use_idf=True, smooth_idf=True).fit_transform(_data)
            elapsed_time['tfidf'] = time.time() - start_time
        elif self.tfidf == 'TF-IDF':
            start_time = time.time()
            if verbose:
                print("TF-IDF transform.")
            _data = TfidfTransformer(use_idf=True, smooth_idf=True).fit_transform(_data)
            elapsed_time['tfidf'] = time.time() - start_time


        if self.dimension_reduction_method is not None:
            if verbose:
                print("Dimension reduction.")
            start_time = time.time()
            _data = self.dimension_reduction_method().transform(_data, **self.dimension_reduction_kw)
            elapsed_time['dimension_reduction'] = time.time() - start_time
        
        if verbose:
            print("Nearest neighbors.")
        ref_read_num = ref.shape[0]
        _ref = _data[:ref_read_num]
        _que = _data[ref_read_num:]
        start_time = time.time()
        neighbor_indices = self.nearest_neighbors_method().get_neighbors(
            _ref,_que, n_neighbors=n_neighbors, **self.nearest_neighbors_kw
        )
        elapsed_time['nearest_neighbors'] = time.time() - start_time
        if verbose:
            print(f"Finished {self}. Elapsed time: {elapsed_time}. Peak memory: {peak_memory}")
        return neighbor_indices, elapsed_time, peak_memory
    
def compute_nearest_neighbors(
    ref: csr_matrix | np.ndarray, 
    que: csr_matrix | np.ndarray,
    config: NearestNeighborsConfig,
    n_neighbors: int,
    *,
    verbose=True,
) -> tuple[ndarray,ndarray,ndarray]:
    
    neighbor_indices, elapsed_time, peak_memory = config.get_neighbors(ref=ref,que=que, n_neighbors=n_neighbors, verbose=verbose)

    return neighbor_indices, elapsed_time, peak_memory