from dataclasses import dataclass, field
from functools import lru_cache
import collections
from typing import Sequence, Type
from math import ceil
from scipy import sparse
from scipy.sparse._csr import csr_matrix
import numpy as np
from numpy import matlib
import sklearn.neighbors
import pynndescent
from sklearn.feature_extraction.text import TfidfTransformer


def get_marker_matrix(
    read_markers, marker_weights, *, use_multiplicity=True, verbose=True
) -> csr_matrix:
    read_list = list(read_markers)
    col_indices = {read: j for j, read in enumerate(read_list)}
    marker_list = list(marker_weights)
    row_indices = {marker: i for i, marker in enumerate(marker_list)}

    values = []
    rows = []
    columns = []
    for read, j in col_indices.items():
        if use_multiplicity:
            marker_multiplicity = collections.Counter(read_markers[read][0])
        else:
            marker_multiplicity = {x: 1 for x in read_markers[read][0]}
        for marker, count in marker_multiplicity.items():
            i = row_indices[marker]
            values.append(marker_weights[marker] * count)
            rows.append(i)
        columns += [j] * len(marker_multiplicity)
        if verbose and j % 10_000 == 0:
            print(j, end=" ")

    marker_matrix = sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(len(row_indices), len(col_indices)),
        dtype=np.uint16,
    )
    marker_matrix = marker_matrix.T
    marker_matrix = csr_matrix(marker_matrix)
    return marker_matrix


@dataclass
class _NearestNeighbors:
    data: csr_matrix | np.ndarray

    def get_neighbors(self, n_neighbors: int) -> np.ndarray:
        raise NotImplementedError()


class ExactNearestNeighbors(_NearestNeighbors):
    def get_neighbors(self, metric="cosine", n_neighbors: int = 20):
        nbrs = sklearn.neighbors.NearestNeighbors(
            n_neighbors=n_neighbors, metric=metric
        )
        data = self.data
        if metric == "jaccard" and isinstance(data, csr_matrix):
            data = data.toarray()
        nbrs.fit(data)
        _, nbr_indices = nbrs.kneighbors(data)
        return nbr_indices


class NNDescent(_NearestNeighbors):
    def get_neighbors(
        self,
        metric="cosine",
        n_neighbors: int = 20,
        *,
        n_trees: int = 100,
        low_memory: bool = False,
        verbose: bool = True,
    ):
        index = pynndescent.NNDescent(
            self.data,
            metric=metric,
            n_neighbors=n_neighbors,
            n_trees=n_trees,
            low_memory=low_memory,
            verbose=verbose,
        )
        nbr_indices, _ = index.neighbor_graph  # type: ignore
        return nbr_indices


class WeightedLowHash(_NearestNeighbors):

    def _pcws_low_hash(
        self, lowhash_fraction, repeats=1, *, seed=1, verbose=True
    ) -> csr_matrix:
        data = self.data.T  # rows for features; columns for instances
        feature_num, instance_num = data.shape
        lowhash_buckets = sparse.dok_matrix(
            (feature_num * repeats, instance_num), dtype=np.bool_
        )

        dimension_num = repeats
        # fingerprints_k = np.zeros((instance_num, dimension_num))

        rng = np.random.default_rng(seed)
        beta = rng.uniform(0, 1, (feature_num, dimension_num))
        x = rng.uniform(0, 1, (feature_num, dimension_num))
        u1 = rng.uniform(0, 1, (feature_num, dimension_num))
        u2 = rng.uniform(0, 1, (feature_num, dimension_num))

        for j_sample in range(0, instance_num):
            feature_id = sparse.find(data[:, j_sample] > 0)[0]
            gamma = -np.log(np.multiply(u1[feature_id, :], u2[feature_id, :]))
            t_matrix = np.floor(
                np.divide(
                    matlib.repmat(
                        np.log(data[feature_id, j_sample].todense()), 1, dimension_num
                    ),
                    gamma,
                )
                + beta[feature_id, :]
            )
            y_matrix = np.exp(np.multiply(gamma, t_matrix - beta[feature_id, :]))
            a_matrix = np.divide(
                -np.log(x[feature_id, :]), np.divide(y_matrix, u1[feature_id, :])
            )

            lowhash_count = ceil(feature_id.shape[0] * lowhash_fraction)
            lowhash_positions = np.argsort(a_matrix, axis=0)[:lowhash_count]
            lowhash_features = feature_id[lowhash_positions]

            bucket_indices = []
            for k in range(repeats):
                features = lowhash_features[:, k]
                bucket_indices.append(features + k * feature_num)

            lowhash_buckets[np.concatenate(bucket_indices), j_sample] = 1

            if verbose and j_sample % 1_000 == 0:
                print(j_sample, end=" ")

        lowhash_buckets = sparse.csr_matrix(lowhash_buckets)
        return lowhash_buckets

    def get_neighbors(
        self,
        n_neighbors: int = 20,
        lowhash_fraction=0.1,
        repeats=50,
        min_bucket_size=2,
        max_bucket_size=float("inf"),
        min_cooccurence_count=1,
        *,
        seed=1,
        verbose=True,
    ) -> np.ndarray:
        # Calculate LowHash
        lowhash_buckets = self._pcws_low_hash(
            lowhash_fraction=lowhash_fraction,
            repeats=repeats,
            seed=seed,
            verbose=verbose,
        )

        # Select neighbor candidates based on cooccurence counts
        row_sums = lowhash_buckets.sum(axis=1).A1  # type: ignore
        matrix = lowhash_buckets[
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
        n_rows = self.data.shape[0]
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
            nbr_matrix[i, : len(neighbors)] = neighbors
        return nbr_matrix



