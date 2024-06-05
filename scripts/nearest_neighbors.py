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
from numpy import matlib
import sklearn.neighbors
import pynndescent
import hnswlib
import faiss


from data_io import parse_paf_file, get_sibling_id


# def get_marker_matrix(
#     read_markers, marker_weights, *, use_multiplicity=True, verbose=True
# ) -> csr_matrix:
#     read_list = list(read_markers)
#     col_indices = {read: j for j, read in enumerate(read_list)}
#     marker_list = list(marker_weights)
#     row_indices = {marker: i for i, marker in enumerate(marker_list)}

#     values = []
#     rows = []
#     columns = []
#     for read, j in col_indices.items():
#         if use_multiplicity:
#             marker_multiplicity = collections.Counter(read_markers[read][0])
#         else:
#             marker_multiplicity = {x: 1 for x in read_markers[read][0]}
#         for marker, count in marker_multiplicity.items():
#             i = row_indices[marker]
#             values.append(marker_weights[marker] * count)
#             rows.append(i)
#         columns += [j] * len(marker_multiplicity)
#         if verbose and j % 10_000 == 0:
#             print(j, end=" ")

#     marker_matrix = sparse.coo_matrix(
#         (values, (rows, columns)),
#         shape=(len(row_indices), len(col_indices)),
#         dtype=np.uint16,
#     )
#     marker_matrix = marker_matrix.T
#     marker_matrix = csr_matrix(marker_matrix)
#     return marker_matrix


@dataclass
class _NearestNeighbors:
    def get_neighbors(
        self, data: csr_matrix | np.ndarray, n_neighbors: int
    ) -> np.ndarray:
        raise NotImplementedError()


class ExactNearestNeighbors(_NearestNeighbors):
    def get_neighbors(
        self, data: csr_matrix | np.ndarray, metric="cosine", n_neighbors: int = 20
    ):
        nbrs = sklearn.neighbors.NearestNeighbors(
            n_neighbors=n_neighbors, metric=metric
        )
        if metric == "jaccard" and isinstance(data, csr_matrix):
            data = data.toarray()
        nbrs.fit(data)
        _, nbr_indices = nbrs.kneighbors(data)
        return nbr_indices


class NNDescent(_NearestNeighbors):
    def get_neighbors(
        self,
        data: csr_matrix | np.ndarray,
        metric="cosine",
        n_neighbors: int = 20,
        *,
        n_trees: int = 100,
        low_memory: bool = True,
        n_jobs: int | None = None,
        seed: int | None = 683985,
        verbose: bool = True,
    ):
        index = pynndescent.NNDescent(
            data,
            metric=metric,
            n_neighbors=n_neighbors,
            n_trees=n_trees,
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
        threads: int | None = None,
        M: int = 16,
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
            raise TypeError("HNSW cannot be used on sparse arrays.")
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
        hash_value = mmh3.hash(str(x), seed=seed)
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

        dimension_num = repeats
        # fingerprints_k = np.zeros((instance_num, dimension_num))

        rng = np.random.default_rng(seed)
        beta = rng.uniform(0, 1, (feature_count, dimension_num))
        x = rng.uniform(0, 1, (feature_count, dimension_num))
        u1 = rng.uniform(0, 1, (feature_count, dimension_num))
        u2 = rng.uniform(0, 1, (feature_count, dimension_num))

        for j_sample in range(0, sample_count):
            feature_indices = sparse.find(data[:, j_sample] > 0)[0]
            gamma = -np.log(np.multiply(u1[feature_indices, :], u2[feature_indices, :]))
            t_matrix = np.floor(
                np.divide(
                    matlib.repmat(
                        np.log(data[feature_indices, j_sample].todense()),
                        1,
                        dimension_num,
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


class ProductQuantization(_NearestNeighbors):
    def get_neighbors(
        self,
        data: csr_matrix | np.ndarray,
        n_neighbors: int,
        metric: Literal["euclidean"] = "euclidean",
        *,
        m=8,
        nbits=8,
        seed=455390,
    ) -> np.ndarray:
        if metric == "euclidean":
            faiss_metric = faiss.METRIC_L2
        else:
            raise ValueError()

        if sparse.issparse(data):
            raise TypeError("ProductQuantization does not support sparse arrays.")
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

        index_pq = faiss.IndexPQ(new_feature_count, m, nbits, faiss_metric)
        index_pq.train(data)  # type: ignore
        index_pq.add(data)  # type: ignore
        _, nbr_indices = index_pq.search(data, n_neighbors)  # type: ignore
        return nbr_indices
