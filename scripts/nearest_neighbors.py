from dataclasses import dataclass, field
import mmh3
from functools import lru_cache
import collections
from typing import Sequence, Type, Mapping, Iterable, Literal
from warnings import warn
from math import ceil
from scipy import sparse
from scipy.sparse._csr import csr_matrix
import numpy as np
from numpy import matlib, ndarray
from numpy.typing import NDArray
import sklearn.neighbors
import hnswlib
from itertools import chain 
from collections import Counter
import secrets
import random
import pynear
import faiss
import pynndescent
from data_io import parse_paf_file, get_sibling_id
def hamming_distance(x, y):  
    return np.count_nonzero(x != y)

@dataclass
class _NearestNeighbors:
    def get_neighbors(
        self, data: csr_matrix | np.ndarray, n_neighbors: int
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
        self, data: csr_matrix | np.ndarray, 
        metric="cosine", 
        n_neighbors: int = 20,
        n_jobs: int | None = 64,
        sample_query_number: int|None = None,
        seed =654556,
    ):

        if metric == "jaccard" and isinstance(data, csr_matrix):
            data = data.toarray()
        if metric == "generalized_jaccard":
            _metric = generalized_jaccard_distance
        else:
            _metric = metric

        nbrs = sklearn.neighbors.NearestNeighbors(
            n_neighbors=n_neighbors, metric=_metric,n_jobs=n_jobs
        )

        nbrs.fit(data)
        if sample_query_number != None:
            random_row_indices = np.random.choice(data.shape[0], size=sample_query_number, replace=False)
            sampled_matrix = data[random_row_indices, :]
            _, nbr_indices = nbrs.kneighbors(sampled_matrix)
        else:
            _, nbr_indices = nbrs.kneighbors(data)
        return nbr_indices


class NNDescent(_NearestNeighbors):
    def get_neighbors(
        self,
        data: csr_matrix | np.ndarray,
        metric="cosine",
        n_neighbors: int = 20,
        *,
        n_trees: int| None = 300,
        leaf_size: int| None = 200,
        low_memory: bool = True,
        n_jobs: int | None = 64,
        seed: int | None = 683985,
        verbose: bool = True,
    ):
        index = pynndescent.NNDescent(
            data,
            metric=metric,
            n_neighbors=n_neighbors,
            n_trees=n_trees,
            leaf_size=leaf_size,
            low_memory=low_memory,
            n_jobs=n_jobs,
            random_state=seed,
            verbose=verbose,
        )
        nbr_indices, _ = index.neighbor_graph  # type: ignore
        return nbr_indices


class HNSW(_NearestNeighbors):
    def get_neighbors(
        self,
        data: csr_matrix | np.ndarray,
        n_neighbors: int,
        metric: Literal["euclidean", "cosine"] = "euclidean",
        *,
        threads: int | None = 64,
        M: int = 512,
        ef_construction: int = 200,
        ef_search: int = 50,
    ) -> np.ndarray:
        """
        See https://www.pinecone.io/learn/series/faiss/vector-indexes/
        M — the number of nearest neighbors that each vertex will connect to.
        efSearch — how many entry points will be explored between layers during the search.
        efConstruction — how many entry points will be explored when building the index.
        """
        if sparse.issparse(data):
            data = data.toarray()
        if metric == "euclidean":
            space = "l2"
        else:
            space = metric

        # Initialize the HNSW index
        p = hnswlib.Index(space=space, dim=data.shape[1])
        if threads is not None:
            p.set_num_threads(threads)
        p.init_index(max_elements=data.shape[0], ef_construction=ef_construction, M=M)
        ids = np.arange(data.shape[0])
        p.add_items(data, ids)
        p.set_ef(ef_search)
        nbr_indices, _ = p.knn_query(data, k=n_neighbors)
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
        data: csr_matrix | np.ndarray,
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
        cooccurrence_matrix = matrix.T.dot(matrix)

        neighbor_dict = collections.defaultdict(dict)
        nonzero_indices = list(zip(*cooccurrence_matrix.nonzero()))
        for i, j in nonzero_indices:
            if i >= j:
                continue

            count = cooccurrence_matrix[i, j]
            neighbor_dict[i][j] = count
            neighbor_dict[j][i] = count

        # Construct neighbor matrix
        n_rows = data.shape[0]
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
        data: csr_matrix | np.ndarray,
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

        buckets = self._lowhash(
            data,
            repeats=repeats,
            lowhash_fraction=lowhash_fraction,
            lowhash_count=lowhash_count,
            seed=seed,
            verbose=verbose,
        )
        nbr_matrix = self._get_adjacency_matrix(
            data,
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
        non_zero_counts = lowhash_buckets.getnnz(axis=1)
        num_non_zero_rows = (non_zero_counts > 0).sum()
        print(f'num_non_zero_rows:{num_non_zero_rows}')
        return lowhash_buckets

    def get_neighbors(
        self,
        data: csr_matrix | np.ndarray,
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
            data,
            buckets,
            n_neighbors=n_neighbors,
            min_bucket_size=min_bucket_size,
            max_bucket_size=max_bucket_size,
            min_cooccurence_count=min_cooccurence_count,
        )
        return nbr_matrix


# @njit(parallel=True)
# def _argpartition(arr, k, axis=0):
#     """
#     This function works like numpy.argpartition,
#     but only returns the first k indices.
#     This function is designed for Numba,
#     as numpy.argpartition is not fully supported in Numba.

#     >>> a = np.array([[1, 2, 3], [3,0,1], [3, 3, 1], [5, 0, 0]])
#     >>> _argpartition(a, 2, axis=0)
#     array([[0, 3, 3],
#            [2, 1, 2]])
#     """
#     if axis == 0:
#         result = np.empty((k, arr.shape[1]), dtype=np.int64)
#         for i in prange(arr.shape[1]):
#             partitioned_indices = np.argpartition(arr[:, i], k)[:k]
#             result[:, i] = partitioned_indices
#     elif axis == 1:
#         result = np.empty((arr.shape[0], k), dtype=np.int64)
#         for i in prange(arr.shape[0]):
#             partitioned_indices = np.argpartition(arr[i, :], k)[:k]
#             result[i, :] = partitioned_indices
#     else:
#         raise ValueError("axis must be 0 or 1")
#     return result


# class JITWeightedLowHash(WeightedLowHash):
#     # This is not faster. Why?

#     @staticmethod
#     def _get_random_numbers(seed: int, feature_count: int, dimension_count: int):
#         rng = np.random.default_rng(seed)
#         beta = rng.uniform(0, 1, (feature_count, dimension_count))
#         x = rng.uniform(0, 1, (feature_count, dimension_count))
#         u1 = rng.uniform(0, 1, (feature_count, dimension_count))
#         u2 = rng.uniform(0, 1, (feature_count, dimension_count))
#         return beta, x, u1, u2


#     @staticmethod
#     @njit
#     def _get_lowhash_positions(
#         weights: ndarray,
#         feature_indices: ndarray,
#         dimension_count: int,
#         lowhash_count: int,
#         beta: ndarray,
#         x: ndarray,
#         u1: ndarray,
#         u2: ndarray,
#     ) -> ndarray:
#         gamma = -np.log(np.multiply(u1[feature_indices, :], u2[feature_indices, :]))
#         t_matrix = np.floor(
#             np.divide(
#                 np.repeat(np.log(weights), dimension_count).reshape(
#                     -1, dimension_count
#                 ),
#                 gamma,
#             )
#             + beta[feature_indices, :]
#         )
#         y_matrix = np.exp(np.multiply(gamma, t_matrix - beta[feature_indices, :]))
#         a_matrix = np.divide(
#             -np.log(x[feature_indices, :]),
#             np.divide(y_matrix, u1[feature_indices, :]),
#         )
#         lowhash_positions = _argpartition(a_matrix, lowhash_count, axis=0)
#         return lowhash_positions

#     def _pcws_low_hash(
#         self,
#         data: csr_matrix | np.ndarray,
#         lowhash_fraction: float | None = None,
#         lowhash_count: int | None = None,
#         repeats=1,
#         *,
#         seed=1,
#         use_weights=True,
#         verbose=True,
#     ) -> csr_matrix:
#         data = data.T.copy()  # rows for features; columns for instances; this will be a sparse CSC matrix
#         if not use_weights:
#             data[data > 0] = 1
#         feature_count, sample_count = data.shape
#         lowhash_buckets = sparse.dok_matrix(
#             (feature_count * repeats, sample_count), dtype=np.bool_
#         )

#         dimension_count = repeats
#         # fingerprints_k = np.zeros((instance_num, dimension_num))

#         beta, x, u1, u2 = self._get_random_numbers(
#             seed=seed, feature_count=feature_count, dimension_count=dimension_count
#         )

#         for j_sample in range(0, sample_count):
#             feature_indices = sparse.find(data[:, j_sample] > 0)[0]
#             hash_count = feature_indices.shape[0]
#             sample_lowhash_count = self._get_lowhash_count(
#                 hash_count=hash_count,
#                 lowhash_fraction=lowhash_fraction,
#                 lowhash_count=lowhash_count,
#             )
#             weights = data[feature_indices, j_sample].todense()
#             lowhash_positions = self._get_lowhash_positions(
#                 weights=weights,
#                 feature_indices=feature_indices,
#                 dimension_count=dimension_count,
#                 lowhash_count=sample_lowhash_count,
#                 beta=beta,
#                 x=x,
#                 u1=u1,
#                 u2=u2,
#             )
#             lowhash_features = feature_indices[lowhash_positions]
#             bucket_indices = []
#             for k in range(repeats):
#                 features = lowhash_features[:, k]
#                 bucket_indices.append(features + k * feature_count)

#             lowhash_buckets[np.concatenate(bucket_indices), j_sample] = 1

#             if verbose and j_sample % 1_000 == 0:
#                 print(j_sample, end=" ")
#         if verbose:
#             print("")
#         lowhash_buckets = sparse.csr_matrix(lowhash_buckets)
#         return lowhash_buckets


class PAFNearestNeighbors(_NearestNeighbors):
    def get_neighbors(
        self,
        data: csr_matrix | np.ndarray,
        n_neighbors: int,
        paf_path: str,
        read_indices: Mapping[str, int],
        *,
        min_alignment_length: int = 0,
    ) -> np.ndarray:
        # Calculate cumulative alignment lengths
        alignment_lengths = collections.defaultdict(collections.Counter)
        for record in parse_paf_file(paf_path):
            i1 = read_indices.get(record.query_name)
            i2 = read_indices.get(record.target_name)
            if i1 is None or i2 is None:
                # Assume query or target is excluded
                continue
            if record.strand == "-":
                i1 = get_sibling_id(i1)
            length = record.alignment_block_length
            alignment_lengths[i1][i2] += length
            alignment_lengths[i2][i1] += length
            i1, i2 = get_sibling_id(i1), get_sibling_id(i2)
            alignment_lengths[i1][i2] += length
            alignment_lengths[i2][i1] += length
        if len(alignment_lengths) == 0:
            warn(f"No overlaps found from {paf_path}")

        # Construct neighbor matrix
        n_rows = data.shape[0]
        nbr_matrix = np.empty((n_rows, n_neighbors), dtype=np.int64)
        nbr_matrix[:, :] = -1
        for i in range(n_rows):
            row_nbr_dict = {
                j: length
                for j, length in alignment_lengths[i].items()
                if length >= min_alignment_length
            }
            neighbors = list(
                sorted(row_nbr_dict, key=lambda x: row_nbr_dict[x], reverse=True)
            )[:n_neighbors]
            nbr_matrix[i, : len(neighbors)] = neighbors
        return nbr_matrix

class RPForest(_NearestNeighbors):
    def get_neighbors(
        self,
        data: csr_matrix | np.ndarray,
        n_neighbors: int,
        metric: Literal["euclidean","cosine"] = "euclidean",
        *,
        leaf_size=50,
        no_trees=10,
    ) -> np.ndarray:

        model = RPForest(leaf_size=leaf_size, no_trees=no_trees)
        model.fit(data)
        nns = model.query(data, n_neighbors)
        return nns


class ProductQuantization(_NearestNeighbors):
    def get_neighbors(
        self,
        data: csr_matrix | np.ndarray,
        n_neighbors: int,
        metric: Literal["euclidean","cosine"] = "euclidean",
        *,
        m=64,
        n_bits=8,
        seed=455390,
        threads: int = 64
    ) -> np.ndarray:

        faiss.omp_set_num_threads(threads)
        if sparse.issparse(data):
            data = data.toarray()
            #raise TypeError("ProductQuantization does not support sparse arrays.")
        feature_count = data.shape[1]
        if feature_count % m != 0:
            new_feature_count = feature_count // m * m
            feature_indices = np.random.default_rng(seed).choice(
                feature_count, new_feature_count, replace=False, shuffle=False
            )
            data = data[:, feature_indices]
        else:
            new_feature_count = feature_count
        assert data.shape[1]

        if metric == "euclidean":
            measure = faiss.METRIC_L2
        else:
            measure = faiss.METRIC_INNER_PRODUCT
            data = np.array(data,order='C').astype('float32')
            faiss.normalize_L2(data)
        
        param = f"PQ{m}x{n_bits}"
        index = faiss.index_factory(new_feature_count,param,measure)
        index.train(data)
        index.add(data)
        _, nbr_indices = index.search(data, n_neighbors)  # type: ignore
        return nbr_indices

class IVFProductQuantization(_NearestNeighbors):
    def get_neighbors(
        self,
        data: csr_matrix | np.ndarray,
        n_neighbors: int,
        metric: Literal["euclidean","cosine"] = "euclidean",
        *,
        M=128,
        n_list=1024,
        n_bits=8, 
        n_probe=100,
        seed=455390,
        threads: int = 64
    ) -> np.ndarray:
        
        faiss.omp_set_num_threads(threads)

        if sparse.issparse(data):
            raise TypeError("ProductQuantization does not support sparse arrays.")
        feature_count = data.shape[1]
        if feature_count % M != 0:
            new_feature_count = feature_count // M * M
            feature_indices = np.random.default_rng(seed).choice(
                feature_count, new_feature_count, replace=False, shuffle=False
            )
            data = data[:, feature_indices]
        else:
            new_feature_count = feature_count
        assert data.shape[1]

        data = np.array(data, order='C').astype('float32')
        faiss.normalize_L2(data)

        if metric == "euclidean":
            measure = faiss.METRIC_L2
        else:
            measure = faiss.METRIC_INNER_PRODUCT
        
        quantizer = faiss.IndexFlatL2(new_feature_count)  # 量化器
        index = faiss.IndexIVFPQ(quantizer, new_feature_count, n_list, M, n_bits)
        index.metric_type = measure
        index.train(data)
        index.add(data)
        index.nprobe = n_probe
        _, nbr_indices = index.search(data, n_neighbors)
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
        data: csr_matrix | np.ndarray,
        n_neighbors: int,
        repeats=500,
        seed=20141025,
    ) -> np.ndarray:
        assert data.shape is not None
        kmer_num = data.shape[1]
        hash_table = self._get_hash_table(kmer_num, repeats=repeats, seed=seed)
        simhash = self.get_simhash(data, hash_table)
        vptree = pynear.VPTreeBinaryIndex()
        vptree.set(simhash)
        vptree_indices, vptree_distances = vptree.searchKNN(simhash, n_neighbors + 1)
        nbr_indices = np.array(vptree_indices)[:, :-1][:, ::-1]
        return nbr_indices
    

class BlockSimHash(SimHash):
    @staticmethod
    def get_simhash(
        data: NDArray | csr_matrix, hash_table: NDArray, *, block_size=2**30
    ) -> NDArray[np.uint8]:
        sample_count, _ = data.shape
        _, hash_size = hash_table.shape
        if hash_size % 8 != 0:
            raise ValueError()
        result = np.empty(shape=(sample_count, hash_size // 8), dtype=np.uint8)

        # 将稠密矩阵 hash_table 分块，逐块进行乘法计算，减少内存峰值
        for start_col in range(0, hash_size, 8 * block_size):

            end_col = min(start_col + 8 * block_size, hash_size)
            block_hash_table = hash_table[:, start_col:end_col]

            # 进行块矩阵乘法
            block_result = data.dot(block_hash_table)
            binary_block_result = np.where(block_result > 0, 1, 0).astype(np.uint8)
            packed_block_result = np.packbits(binary_block_result, axis=-1)

            # 将结果写入预先分配的结果矩阵
            result[:, start_col // 8 : end_col // 8] = packed_block_result

        return result
    

class NewSimHash(SimHash):
    @staticmethod
    def get_simhash(data: NDArray | csr_matrix, hash_table: NDArray):
        simhash = (data @ hash_table).astype(np.uint8)
        return simhash

class NewSimHash2(SimHash):
    @staticmethod
    def get_simhash(data: NDArray | csr_matrix, hash_table: NDArray):
        simhash = (data @ hash_table)
        return simhash
    
    def get_neighbors(
        self,
        data: csr_matrix | np.ndarray,
        n_neighbors: int,
        repeats=400,
        seed=20141025,
    ) -> np.ndarray:
        import sklearn.neighbors

        assert data.shape is not None
        kmer_num = data.shape[1]

        hash_table = self._get_hash_table(kmer_num, repeats=repeats, seed=seed)

        simhash = self.get_simhash(data, hash_table)
        nbrs = sklearn.neighbors.NearestNeighbors(
            n_neighbors=n_neighbors, metric='cosine'
        )

        nbrs.fit(simhash)
        _, nbr_indices = nbrs.kneighbors(simhash)
        return nbr_indices
    
class NewSimHash3(SimHash):
    @staticmethod
    def get_simhash(data: NDArray | csr_matrix, hash_table: NDArray):
        simhash = (data @ hash_table)
        return simhash
    
    def get_neighbors(
        self,
        data: csr_matrix | np.ndarray,
        n_neighbors: int,
        repeats=400,
        seed=20141025,
    ) -> np.ndarray:
        assert data.shape is not None
        kmer_num = data.shape[1]
        hash_table = self._get_hash_table(kmer_num, repeats=repeats, seed=seed)
        simhash = self.get_simhash(data, hash_table)
        myhnsw = HNSW()
        nbr_indices = myhnsw.get_neighbors(simhash,n_neighbors)
        return nbr_indices