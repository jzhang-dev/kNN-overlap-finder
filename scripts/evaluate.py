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
import gc
from scipy.sparse import diags
from nearest_neighbors import _NearestNeighbors, ExactNearestNeighbors
from dim_reduction import _DimensionReduction


def tf_transform(feature_matrix: csr_matrix):
    row_sums = feature_matrix.sum(axis=1).A1
    row_sums = row_sums.reshape(-1, 1)
    feature_matrix /= row_sums
    feature_matrix = feature_matrix.tocsr()
    return feature_matrix


def idf_transform(feature_matrix: csr_matrix, idf=None):
    if idf is None:
        # Memory-efficient column sum
        col_sums = np.asarray(feature_matrix.sum(axis=0)).ravel()
        assert feature_matrix.shape is not None
        nrow = feature_matrix.shape[0]
        
        idf = np.log(nrow / (col_sums.astype(np.float32) + 1e-12)).astype(np.float32)
    
    # Sparse matrix multiplication (memory-efficient)
    idf_diag = diags(idf, format='csr')
    feature_matrix = feature_matrix.dot(idf_diag)
    return feature_matrix, idf

def tfidf_transform(feature_matrix: csr_matrix, idf=None):
    if idf is None:
        binary_matrix = feature_matrix.copy()
        binary_matrix[binary_matrix > 0] = 1
        # Memory-efficient column sum
        col_sums = np.asarray(binary_matrix.sum(axis=0)).ravel()
        assert binary_matrix.shape is not None
        nrow = binary_matrix.shape[0]
        
        idf = np.log(nrow / (col_sums.astype(np.float32) + 1e-12)).astype(np.float32)
    
    idf_diag = diags(idf, format='csr')

    row_sums = feature_matrix.sum(axis=1).A1
    row_sums = row_sums.reshape(-1, 1)
    feature_matrix /= row_sums
    feature_matrix = feature_matrix.tocsr()

    feature_matrix = feature_matrix.dot(idf_diag)

    return feature_matrix, idf_diag


@dataclass
class NearestNeighborsConfig:
    description: str = ""
    tfidf: Literal["TF","IDF","TF-IDF",'binary','count'] = 'count',
    dimension_reduction_method: Type[_DimensionReduction] | None = None
    dimension_reduction_kw: dict = field(default_factory=dict, repr=False)
    nearest_neighbors_method: Type[_NearestNeighbors] = ExactNearestNeighbors
    nearest_neighbors_kw: dict = field(default_factory=dict, repr=False)

    def get_neighbors(
        self, data: csr_matrix, n_neighbors: int, *, sample_query=False, verbose=True
    ) -> tuple[ndarray, Mapping[str, float], Mapping[str, float]]:
        elapsed_time = {}

        start_time = time.time()
        if self.tfidf == "IDF":
            data[data > 0] = 1
            data, _ = idf_transform(data)
        elif self.tfidf == "TF-IDF":
            data, _ = tfidf_transform(data)
        elif self.tfidf == "binary":
            data[data > 0] = 1
        elif self.tfidf == "count":
            pass
        elif self.tfidf == "TF":
            data = tf_transform(data)
        else:
            raise ValueError(
                f"Invalid preprocess method. Expected one of TF/IDF/TF-IDF/binary/count."
            )
        elapsed_time['tfidf'] = time.time() - start_time

        if self.dimension_reduction_method is not None:
            if verbose:
                print("Dimension reduction.")
            start_time = time.time()
            low_dimensions_data = self.dimension_reduction_method().transform(data, **self.dimension_reduction_kw)
            elapsed_time['dimension_reduction'] = time.time() - start_time
            if scipy.sparse.issparse(low_dimensions_data):
                low_dimensions_data = low_dimensions_data.toarray()
        else:
            low_dimensions_data = data

        del data
        gc.collect()  

        if verbose:
            print("Nearest neighbors.")
        start_time = time.time()

        neighbor_indices = self.nearest_neighbors_method().get_neighbors(
            low_dimensions_data, n_neighbors=n_neighbors, **self.nearest_neighbors_kw
        )
        elapsed_time['nearest_neighbors'] = time.time() - start_time
        if verbose:
            print(f"Finished {self}. Elapsed time: {elapsed_time}.")
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
