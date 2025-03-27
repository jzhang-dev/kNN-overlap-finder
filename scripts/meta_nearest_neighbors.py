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
from math import ceil
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
    
def generalized_jaccard_similarity(
    x: csr_matrix | np.ndarray, y: csr_matrix | np.ndarray
) -> float:
    if x.shape[0] != 1 or y.shape[0] != 1:
        raise ValueError()
    if x.shape[1] != y.shape[1]:
        raise ValueError()

    s = sparse.vstack([x, y])  # TODO: dense
    jaccard_similarity = s.min(axis=0).sum() / s.max(axis=0).sum()
    return jaccard_similarity


def generalized_jaccard_distance(
    x: csr_matrix | np.ndarray, y: csr_matrix | np.ndarray
) -> float:
    return 1 - generalized_jaccard_similarity(x, y)

class ExactNearestNeighbors(_NearestNeighbors):
    def get_neighbors(
        self,        
        ref: csr_matrix | np.ndarray,
        que: csr_matrix | np.ndarray,
        metric="cosine", n_neighbors: int = 20
    ):

        if metric == "jaccard" and isinstance(data, csr_matrix):
            data = data.toarray()
        if metric == "generalized_jaccard":
            _metric = generalized_jaccard_distance
        else:
            _metric = metric

        nbrs = sklearn.neighbors.NearestNeighbors(
            n_neighbors=n_neighbors, metric=_metric
        )
        nbrs.fit(ref)
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
        
        if sparse.issparse(ref):
            ref = ref.toarray()
        if sparse.issparse(que):
            que = que.toarray()
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
        nbr_indices, _ = p.knn_query(que, k=n_neighbors)
        return nbr_indices

class HNSW_parallel(_NearestNeighbors):
    ## Parallel will use more memory, but short time
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
        
        if sparse.issparse(ref):
            ref = ref.toarray()
        if sparse.issparse(que):
            que = que.toarray()
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

        # Split query matrix into chunks for parallel processing
        def process_query_chunk(query_chunk):
            return p.knn_query(query_chunk, k=n_neighbors)[0]

        # Number of chunks (equal to the number of threads)
        num_chunks = threads if threads is not None else 4
        query_chunks = np.array_split(que, num_chunks)

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=num_chunks) as executor:
            results = list(executor.map(process_query_chunk, query_chunks))

        # Concatenate results
        nbr_indices = np.vstack(results)
        return nbr_indices

class SimHash(_NearestNeighbors):
    @staticmethod
    def _get_hash_table(
        feature_count: int, repeats: int, seed: int
    ) -> NDArray[np.int8]:
        rng = np.random.default_rng(seed)
        hash_table = rng.integers(
            0, 2, size=(feature_count, repeats * 8), dtype=np.int8
        )
        hash_table = hash_table * 2 - 1
        return hash_table
    
    @staticmethod
    def get_simhash(
        data: NDArray | csr_matrix, hash_table: NDArray
    ) -> NDArray[np.uint8]:
        simhash = data @ hash_table
        binary_simhash = np.where(simhash > 0, 1, 0).astype(np.uint8)
        packed_simhash = np.packbits(binary_simhash, axis=-1) 
        return packed_simhash

    def get_neighbors(
        self,
        ref: csr_matrix | np.ndarray,
        que: csr_matrix | np.ndarray,
        n_neighbors: int,
        repeats=400,
        seed=20141025,
    ) -> np.ndarray:
        assert ref.shape != () and que.shape != ()
        data = sparse.vstack([ref,que])
        kmer_num = data.shape[1]
        hash_table = self._get_hash_table(kmer_num, repeats=repeats, seed=seed)
        simhash = self.get_simhash(data, hash_table)
        ref_sim = simhash[:ref.shape[0]]
        que_sim = simhash[ref.shape[0]:]
        vptree = pynear.VPTreeBinaryIndex()
        vptree.set(ref_sim)
        vptree_indices, vptree_distances = vptree.searchKNN(que_sim, n_neighbors + 1)
        nbr_indices = np.array(vptree_indices)[:, :-1][:, ::-1]
        return nbr_indices

class LowHash(_NearestNeighbors):

    @staticmethod
    def _hash(x: int, seed: int) -> int:
        hash_value = mmh3.hash(str(x), seed=int(seed))
        return hash_value

    @staticmethod
    def _get_hash_values(data: Iterable[int], repeats: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        hash_seeds = rng.integers(low=0, high=2**32 - 1, size=repeats, dtype=np.uint64)
        hash_values = []
        for k in range(repeats):
            s = hash_seeds[k]
            for x in data:
                hash_values.append(LowHash._hash(x, seed=s))
        hash_values = np.array(hash_values, dtype=np.int64)
        return hash_values

    @staticmethod
    def _get_lowhash_count(
        hash_count: int,
        lowhash_fraction: float | None = None,
        lowhash_count: int | None = None,
    ) -> int:
        if lowhash_fraction is None and lowhash_count is None:
            raise TypeError(
                "Either `lowhash_fraction` or `lowhash_count` must be specified."
            )
        if lowhash_fraction is not None and lowhash_count is not None:
            raise TypeError(
                f"`lowhash_fraction` and `lowhash_count` cannot be specified at the same time. {lowhash_fraction=} {lowhash_count=}"
            )

        if lowhash_fraction is not None:
            lowhash_count = ceil(hash_count * lowhash_fraction)
            lowhash_count = max(lowhash_count, 1)
        if lowhash_count is None:
            raise ValueError()
        return lowhash_count

    def _lowhash(
        self,
        data: csr_matrix | np.ndarray,
        repeats: int,
        lowhash_fraction: float | None,
        lowhash_count: int | None = None,
        seed: int = 5731343,
        verbose=True,
    ) -> csr_matrix:

        sample_count, feature_count = data.shape
        buckets = sparse.dok_matrix(
            (feature_count * repeats, sample_count), dtype=np.bool_
        )

        # Calculate hash values
        hash_values = self._get_hash_values(
            np.arange(feature_count), repeats=repeats, seed=seed
        )

        # For each sample, find the lowest hash values for its features
        for j in range(sample_count):
            feature_indices = sparse.find(data[j, :] > 0)[1]
            hash_count = feature_indices.shape[0]
            sample_lowhash_count = self._get_lowhash_count(
                hash_count=hash_count,
                lowhash_fraction=lowhash_fraction,
                lowhash_count=lowhash_count,
            )
            for k in range(repeats):
                bucket_indices = feature_indices + (k * feature_count)
                sample_hash_values = hash_values[bucket_indices]
                low_hash_buckets = bucket_indices[
                    np.argsort(sample_hash_values)[:sample_lowhash_count]
                ]
                buckets[low_hash_buckets, j] = 1
            if verbose and j % 1000 == 0:
                print(j, end=" ")
        if verbose:
            print("")
        buckets = sparse.csr_matrix(buckets)
        return buckets

    def _get_adjacency_matrix(
        self,
        ref_reads_num: int,
        buckets: csr_matrix,
        n_neighbors: int,
        min_bucket_size,
        max_bucket_size,
        min_cooccurence_count,
    ) -> np.ndarray:

        # Select neighbor candidates based on cooccurence counts
        row_sums = buckets.sum(axis=1).A1  # type: ignore
        matrix = buckets[
            (row_sums >= min_bucket_size) & (row_sums <= max_bucket_size), :
        ].astype(np.uint8)
        print(matrix.shape)
        ref_matrix = matrix[:,:ref_reads_num]
        que_matrix = matrix[:,ref_reads_num:]
        cooccurrence_matrix = que_matrix.T.dot(ref_matrix) 
        print(cooccurrence_matrix.shape)
        sparse.save_npz('/home/miaocj/docker_dir/kNN-overlap-finder/data/evaluation/metagenome/bacteria/pbsim_ONT_95_20k/kmer_16/cooccurrence_matrix.npz', cooccurrence_matrix)
        neighbor_dict = collections.defaultdict(dict)
        nonzero_indices = list(zip(*cooccurrence_matrix.nonzero()))
        for i, j in nonzero_indices:
            if i >= j:
                continue
            count = cooccurrence_matrix[i, j]
            neighbor_dict[i][j] = count
            neighbor_dict[j][i] = count

        # Construct neighbor matrix
        n_rows = que_matrix.shape[1]
        nbr_matrix = np.empty((n_rows, n_neighbors), dtype=np.int64)
        nbr_matrix[:, :] = -1
        for i in range(n_rows):
            row_nbr_dict = {
                j: count
                for j, count in neighbor_dict[i].items()
                if count >= min_cooccurence_count
            }
            neighbors = list(
                sorted(row_nbr_dict, key=lambda x: row_nbr_dict[x], reverse=True)
            )[:n_neighbors]
            nbr_matrix[i, : len(neighbors)] = (
                neighbors  # len(neighbors) could be smaller than n_neighbors
            )
        return nbr_matrix

    def get_neighbors(
        self,
        ref: csr_matrix | np.ndarray,
        que: csr_matrix | np.ndarray,
        n_neighbors: int,
        lowhash_fraction: float | None = None,
        lowhash_count: int | None = None,
        repeats=100,
        min_bucket_size=2,
        max_bucket_size=float("inf"),
        min_cooccurence_count=1,
        *,
        seed=1,
        verbose=True,
    ) -> np.ndarray:
        data = sparse.vstack([ref,que])
        ref_reads_num = ref.shape[0]
        buckets = self._lowhash(
            data,
            repeats=repeats,
            lowhash_fraction=lowhash_fraction,
            lowhash_count=lowhash_count,
            seed=seed,
            verbose=verbose,
        )
        nbr_matrix = self._get_adjacency_matrix(
            ref_reads_num,
            buckets,
            n_neighbors=n_neighbors,
            min_bucket_size=min_bucket_size,
            max_bucket_size=max_bucket_size,
            min_cooccurence_count=min_cooccurence_count,
        )
        return nbr_matrix

class WeightedLowHash(LowHash):

    def _pcws_low_hash(
        self,
        data: csr_matrix | np.ndarray,
        lowhash_fraction: float | None = None,
        lowhash_count: int | None = None,
        repeats=1,
        *,
        seed=1,
        use_weights=True,
        verbose=True,
    ) -> csr_matrix:
        data = data.T.copy()  # rows for features; columns for instances
        if not use_weights:
            data[data > 0] = 1
        feature_count, sample_count = data.shape
        lowhash_buckets = sparse.dok_matrix(
            (feature_count * repeats, sample_count), dtype=np.bool_
        )

        dimension_count = repeats
        # fingerprints_k = np.zeros((instance_num, dimension_num))

        rng = np.random.default_rng(seed)
        beta = rng.uniform(0, 1, (feature_count, dimension_count))
        x = rng.uniform(0, 1, (feature_count, dimension_count))
        u1 = rng.uniform(0, 1, (feature_count, dimension_count))
        u2 = rng.uniform(0, 1, (feature_count, dimension_count))

        for j_sample in range(0, sample_count):
            feature_indices = sparse.find(data[:, j_sample] > 0)[0]
            gamma = -np.log(np.multiply(u1[feature_indices, :], u2[feature_indices, :]))
            t_matrix = np.floor(
                np.divide(
                    matlib.repmat(
                        np.log(data[feature_indices, j_sample].todense()),
                        1,
                        dimension_count,
                    ),
                    gamma,
                )
                + beta[feature_indices, :]
            )
            y_matrix = np.exp(np.multiply(gamma, t_matrix - beta[feature_indices, :]))
            a_matrix = np.divide(
                -np.log(x[feature_indices, :]),
                np.divide(y_matrix, u1[feature_indices, :]),
            )

            hash_count = feature_indices.shape[0]
            sample_lowhash_count = self._get_lowhash_count(
                hash_count=hash_count,
                lowhash_fraction=lowhash_fraction,
                lowhash_count=lowhash_count,
            )
            lowhash_positions = np.argsort(a_matrix, axis=0)[:sample_lowhash_count]
            lowhash_features = feature_indices[lowhash_positions]

            bucket_indices = []
            for k in range(repeats):
                features = lowhash_features[:, k]
                bucket_indices.append(features + k * feature_count)

            lowhash_buckets[np.concatenate(bucket_indices), j_sample] = 1

            if verbose and j_sample % 1_000 == 0:
                print(j_sample, end=" ")
        if verbose:
            print("")
        lowhash_buckets = sparse.csr_matrix(lowhash_buckets)
        return lowhash_buckets

    def get_neighbors(
        self,
        ref: csr_matrix | np.ndarray,
        que: csr_matrix | np.ndarray,
        n_neighbors: int,
        lowhash_fraction: float | None = None,
        lowhash_count: int | None = None,
        repeats=100,
        min_bucket_size=2,
        max_bucket_size=float("inf"),
        min_cooccurence_count=1,
        *,
        seed=1,
        use_weights=True,
        verbose=True,
    ) -> np.ndarray:
        data = sparse.vstack([ref,que])
        ref_reads_num = ref.shape[0]
        buckets = self._pcws_low_hash(
            data,
            repeats=repeats,
            lowhash_fraction=lowhash_fraction,
            lowhash_count=lowhash_count,
            seed=seed,
            use_weights=use_weights,
            verbose=verbose,
        )
        nbr_matrix = self._get_adjacency_matrix(
            ref_reads_num,
            buckets,
            n_neighbors=n_neighbors,
            min_bucket_size=min_bucket_size,
            max_bucket_size=max_bucket_size,
            min_cooccurence_count=min_cooccurence_count,
        )
        return nbr_matrix
    
@dataclass
class NearestNeighborsConfig:
    description: str = ""
    tfidf: Literal["TF","IDF","TF-IDF",'None'] = 'None',
    dimension_reduction_method: Type[_DimensionReduction] | None = None
    dimension_reduction_kw: dict = field(default_factory=dict, repr=False)
    nearest_neighbors_method: Type[_NearestNeighbors] = ExactNearestNeighbors
    nearest_neighbors_kw: dict = field(default_factory=dict, repr=False)

    def preprocess_dim(
        self, ref: csr_matrix | np.ndarray, que: csr_matrix | np.ndarray, *, verbose=True
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
            transformer = TfidfTransformer(use_idf=True, smooth_idf=True)
            ref_train = transformer.fit_transform(ref)
            que_fit =  transformer.transform(que)
            _data = sparse.vstack([ref_train,que_fit])
            elapsed_time['tfidf'] = time.time() - start_time
        elif self.tfidf == 'TF-IDF':
            start_time = time.time()
            if verbose:
                print("TF-IDF transform.")
            transformer = TfidfTransformer(use_idf=True, smooth_idf=True)
            ref_train = transformer.fit_transform(ref)
            que_fit =  transformer.transform(que)
            elapsed_time['tfidf'] = time.time() - start_time

        if self.dimension_reduction_method is not None:
            if verbose:
                print("Dimension reduction.")
            start_time = time.time()
            _data = self.dimension_reduction_method().transform(_data, **self.dimension_reduction_kw)
            elapsed_time['dimension_reduction'] = time.time() - start_time
        ref_read_num = ref_train.shape[0]
        _ref = _data[:ref_read_num]
        _que = _data[ref_read_num:]
        return _ref, _que, elapsed_time, peak_memory
    
    def get_neighbors(
        self, _ref: csr_matrix | np.ndarray ,_que: csr_matrix | np.ndarray, 
        elapsed_time, peak_memory,n_neighbors: int, *, verbose=True
    ) -> tuple[ndarray, Mapping[str, float], Mapping[str, float]]:
        
        start_time = time.time()
        neighbor_indices = self.nearest_neighbors_method().get_neighbors(
            _ref,_que, n_neighbors=n_neighbors, **self.nearest_neighbors_kw
        )
        elapsed_time['nearest_neighbors'] = time.time() - start_time
        if verbose:
            print(f"Finished {self}. Elapsed time: {elapsed_time}. Peak memory: {peak_memory}")
        return neighbor_indices, elapsed_time, peak_memory
    
def do_proprecess_dim(
    ref: csr_matrix | np.ndarray, 
    que: csr_matrix | np.ndarray,
    config: NearestNeighborsConfig,
    *,
    verbose=True,
) -> tuple[ndarray,ndarray,ndarray]:
    _ref, _que, elapsed_time, peak_memory = config.preprocess_dim(ref,que)
    return _ref, _que, elapsed_time, peak_memory 

def compute_nearest_neighbor(
    _ref: csr_matrix | np.ndarray, 
    _que: csr_matrix | np.ndarray,
    elapsed_time,
    peak_memory,
    config: NearestNeighborsConfig,
    n_neighbors,
    *,
    verbose=True,
) -> tuple[ndarray,ndarray,ndarray]:
    neighbor_indices, elapsed_time, peak_memory = config.get_neighbors(_ref,_que,elapsed_time, peak_memory,n_neighbors)
    return neighbor_indices, elapsed_time, peak_memory