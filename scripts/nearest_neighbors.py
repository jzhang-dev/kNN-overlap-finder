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
import pynear
import faiss
import time
import pynndescent
from data_io import parse_paf_file, get_sibling_id
from accelerate import open_gzipped

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

def process_nbr_matrix(
        nbr_indices,
        nbr_distance,
        n_neighbors
):
    new_matrix = np.full_like(nbr_indices, fill_value=-1)
    for read_id in range(nbr_indices.shape[0]):
        sibling_id = get_sibling_id(read_id)
        nbr_array = np.concatenate([nbr_indices[read_id, :], np.array([get_sibling_id(i) for i in nbr_indices[sibling_id, :]])])
        distance_array = np.concatenate([nbr_distance[read_id, :], nbr_distance[sibling_id, :]])

        # 去重，保留唯一邻居（并保留对应的最小/最大距离，取决于需求）
        unique_values, indices = np.unique(nbr_array, return_index=True)
        unique_distance_array = distance_array[indices]

        # 按距离排序，取前 n_neighbors 个邻居
        sorted_indices = np.argsort(unique_distance_array) 
        top_neighbors = unique_values[sorted_indices][:n_neighbors]

        new_matrix[read_id, :len(top_neighbors)] = top_neighbors
    return new_matrix

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
        print('search model finished')
        nbr_distance, nbr_indices = nbrs.kneighbors(data)

        new_matrix = process_nbr_matrix(nbr_indices,nbr_distance,n_neighbors)

        return new_matrix


class NNDescent(_NearestNeighbors):
    def get_neighbors(
        self,
        data: csr_matrix | np.ndarray,
        metric="cosine",
        n_neighbors: int = 20,
        *,
        index_n_neighbors:int=50,
        n_trees: int| None = 300,
        leaf_size: int| None = 200,
        n_iters: int |None = None,
        diversify_prob: float|None=1,
        pruning_degree_multiplier:float|None=1.5,
        low_memory: bool = True,
        n_jobs: int | None = 64,
        seed: int | None = 683985,
        verbose: bool = True,
    ):
        index = pynndescent.NNDescent(
            data,
            metric=metric,
            n_neighbors=index_n_neighbors,
            n_trees=n_trees,
            leaf_size=leaf_size,
            n_iters=n_iters,
            diversify_prob=diversify_prob,
            pruning_degree_multiplier=pruning_degree_multiplier,
            low_memory=low_memory,
            n_jobs=n_jobs,
            random_state=seed,
            verbose=verbose,
        )
        _nbr_indices, _nbr_distance = index.neighbor_graph
        nbr_indices = _nbr_indices[:,:n_neighbors]
        nbr_distance = _nbr_distance[:,:n_neighbors]
        new_matrix = process_nbr_matrix(nbr_indices,nbr_distance,n_neighbors)
        return new_matrix


class HNSW(_NearestNeighbors):
    def get_neighbors(
        self,
        data: csr_matrix | np.ndarray,
        n_neighbors: int,
        metric: Literal["euclidean", "cosine"] = "euclidean",
        *,
        n_jobs: int | None = 64,
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
        if n_jobs is not None:
            p.set_num_threads(n_jobs)
        p.init_index(max_elements=data.shape[0], ef_construction=ef_construction, M=M)
        ids = np.arange(data.shape[0])
        p.add_items(data, ids)
        p.set_ef(ef_search)
        nbr_indices, nbr_distance = p.knn_query(data, k=n_neighbors)
        new_matrix = process_nbr_matrix(nbr_indices,nbr_distance,n_neighbors)
        return new_matrix

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
        n_jobs: int = 64
    ) -> np.ndarray:

        faiss.omp_set_num_threads(n_jobs)
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
        nbr_distance, nbr_indices = index.search(data, n_neighbors)  # type: ignore
        new_matrix = process_nbr_matrix(nbr_indices,nbr_distance,n_neighbors)
        return new_matrix

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
        n_jobs: int = 64
    ) -> np.ndarray:
        
        faiss.omp_set_num_threads(n_jobs)

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
        nbr_distance, nbr_indices = index.search(data, n_neighbors)
        new_matrix = process_nbr_matrix(nbr_indices,nbr_distance,n_neighbors)
        return new_matrix


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
        n_jobs:int| None =None,
    ) -> np.ndarray:
        assert data.shape is not None
        kmer_num = data.shape[1]
        time1 = time.time()
        hash_table = self._get_hash_table(kmer_num, repeats=repeats, seed=seed)
        simhash = self.get_simhash(data, hash_table)
        time2 = time.time()
        print(f"SimHash generation time: {time2 - time1:.2f} seconds")
        vptree = pynear.VPTreeBinaryIndex()
        vptree.set(simhash)
        vptree_indices, vptree_distances = vptree.searchKNN(simhash, n_neighbors + 1)
        nbr_indices = np.array(vptree_indices)[:, :-1][:, ::-1]
        nbr_distance = np.array(vptree_distances)[:, :-1][:, ::-1]
        new_matrix = process_nbr_matrix(nbr_indices,nbr_distance,n_neighbors)
        print(f"SimHash neighbor search time: {time.time() - time2:.2f} seconds")
        return new_matrix
    
class PAFNearestNeighbors(_NearestNeighbors):
    def get_neighbors(
        self,
        n_rows: int,
        n_neighbors: int,
        paf_path: str,
        read_indices: Mapping[str, int],
        *,
        min_alignment_length: int = 0,
    ) -> np.ndarray:
        # Calculate cumulative alignment lengths
        alignment_lengths = collections.defaultdict(collections.Counter)
        i=0
        for record in parse_paf_file(paf_path):
            ##recorde process
            i+=1
            if i % 10000000 == 0:
                print(i)
                
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

        print('alignment_lengths generation done')
        # Construct neighbor matrix
        nbr_matrix = np.empty((n_rows, n_neighbors), dtype=np.int64)
        nbr_matrix[:, :] = -1
        for i in range(n_rows):
            if i % 10000 ==0:
                print(i)
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

class idPAFNearestNeighbors(_NearestNeighbors):
    def get_neighbors(
        self,
        n_rows: int,
        n_neighbors: int,
        paf_path: str,
        read_indices: Mapping[str, int],
        *,
        min_alignment_length: int = 0,
    ) -> np.ndarray:
        # Calculate cumulative alignment lengths
        alignment_lengths = collections.defaultdict(collections.Counter)
        i=0
        for record in parse_paf_file(paf_path):
            ##recorde process
            i+=1
            if i % 10000000 == 0:
                print(i)
                
            i1 = int(record.query_name)*2
            i2 = int(record.target_name)*2
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

        print('alignment_lengths generation done')
        # Construct neighbor matrix
        nbr_matrix = np.empty((n_rows, n_neighbors), dtype=np.int64)
        nbr_matrix[:, :] = -1
        for i in range(n_rows):
            if i % 10000 ==0:
                print(i)
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

class MHAPNearestNeighbors(_NearestNeighbors):
    def get_neighbors(
        self,
        n_rows: int,
        n_neighbors: int,
        paf_path: str,
        read_indices: Mapping[str, int],
        *,
        max_error_percentage: float = 0.3,
    ) -> np.ndarray:
        # Calculate cumulative alignment lengths
        share_minimizer_dict = collections.defaultdict(collections.Counter)
        i=0
        with open_gzipped(paf_path,'rt') as f:
            for line in f:
                li = line.strip().split(' ')
                if len(li) > 10:
                    i1 = read_indices.get(li[0])
                    i2 = read_indices.get(li[1])
                    if li[4] == '1':
                        i1 = get_sibling_id(i1)
                    if li[8] == '1':
                        i2 = get_sibling_id(i2)
                    if float(li[2]) < max_error_percentage:
                        share_minimizer_count = float(li[3])
                        share_minimizer_dict[i1][i2] = share_minimizer_count
                        share_minimizer_dict[i2][i1] = share_minimizer_count
                        i1, i2 = get_sibling_id(i1), get_sibling_id(i2)
                        share_minimizer_dict[i1][i2] = share_minimizer_count
                        share_minimizer_dict[i2][i1] = share_minimizer_count
        nbr_matrix = np.empty((n_rows, n_neighbors), dtype=np.int64)
        nbr_matrix[:, :] = -1
        for i in range(n_rows):
            row_nbr_dict = {
                j: share_minimizer_count
                for j, share_minimizer_count in share_minimizer_dict[i].items()
            }
            neighbors = list( 
                sorted(row_nbr_dict, key=lambda x: row_nbr_dict[x], reverse=True)
            )[:n_neighbors]
            nbr_matrix[i, : len(neighbors)] = neighbors
        return nbr_matrix


class wtdbg2NearestNeighbors(_NearestNeighbors):
    def get_neighbors(
        self,
        n_rows: int,
        n_neighbors: int,
        paf_path: str,
        read_indices: Mapping[str, int],
        *,
        min_alignment_length: int = 0,
    ) -> np.ndarray:
        # Calculate cumulative alignment lengths
        alignment_lengths = collections.defaultdict(collections.Counter)
        i=0
        with open_gzipped(paf_path,'rt') as f:
            for line in f:
                li = line.strip().split('\t')
                if len(li) > 12:
                    if int(li[10]) >= 150 and int(li[10])/int(li[11]) >= 0.3:
                        i1 = read_indices.get(li[0])
                        i2 = read_indices.get(li[5])
                        if i1 is None or i2 is None:
                            # Assume query or target is excluded
                            continue
                        if li[1] == "-":
                            i1 = get_sibling_id(i1)
                        if li[6] == "-":
                            i2 = get_sibling_id(i2)
                        length = int(li[10])
                        alignment_lengths[i1][i2] += length
                        alignment_lengths[i2][i1] += length
                        i1, i2 = get_sibling_id(i1), get_sibling_id(i2)
                        alignment_lengths[i1][i2] += length
                        alignment_lengths[i2][i1] += length
        if len(alignment_lengths) == 0:
            warn(f"No overlaps found from {paf_path}")

        print('alignment_lengths generation done')
        # Construct neighbor matrix
        nbr_matrix = np.empty((n_rows, n_neighbors), dtype=np.int64)
        nbr_matrix[:, :] = -1
        for i in range(n_rows):
            if i % 10000 ==0:
                print(i)
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
    
class MECAT2NearestNeighbors(_NearestNeighbors):
    def get_neighbors(
        self,
        n_rows: int,
        n_neighbors: int,
        paf_path: str,
        read_indices: Mapping[str, int],
        *,
        min_indentity: float = 75,
    ) -> np.ndarray:
        # Calculate cumulative alignment lengths
        identity_dict = collections.defaultdict(collections.Counter)
        i=0
        with open_gzipped(paf_path,'rt') as f:
            for line in f:
                li = line.strip().split('\t')
                if len(li) > 10:
                    i1 = int(li[0])*2
                    i2 = int(li[1])*2
                    if li[4] == '1':
                        i1 = get_sibling_id(i1)
                    if li[8] == '1':
                        i2 = get_sibling_id(i2)
                    if float(li[2]) > min_indentity:
                        identity = float(li[2])
                        identity_dict[i1][i2] = identity
                        identity_dict[i2][i1] = identity
                        i1, i2 = get_sibling_id(i1), get_sibling_id(i2)
                        identity_dict[i1][i2] = identity
                        identity_dict[i2][i1] = identity
        nbr_matrix = np.empty((n_rows, n_neighbors), dtype=np.int64)
        nbr_matrix[:, :] = -1
        for i in range(n_rows):
            row_nbr_dict = {
                j: identity
                for j, identity in identity_dict[i].items()
            }
            neighbors = list( 
                sorted(row_nbr_dict, key=lambda x: row_nbr_dict[x], reverse=True)
            )[:n_neighbors]
            nbr_matrix[i, : len(neighbors)] = neighbors
        return nbr_matrix

class FEDRANNNearestNeighbors(_NearestNeighbors):
    def get_neighbors(
        self,
        n_rows: int,
        n_neighbors: int,
        paf_path: str,
        read_indices: Mapping[str, int],
        *,
        min_indentity: float = 75,
    ) -> np.ndarray:
        # Calculate cumulative alignment lengths
        rank_dict = collections.defaultdict(collections.Counter)
        i=0
        with open_gzipped(paf_path,'rt') as f:
            next(f)
            for line in f:
                li = line.strip().split('\t')
                if len(li) == 5:
                    i1 = read_indices.get(li[0])
                    i2 = read_indices.get(li[2])
                    if li[1] == '-':
                        i1 = get_sibling_id(i1)
                    if li[3] == '-':
                        i2 = get_sibling_id(i2)
                    rank = int(li[4])
                    rank_dict[i1][i2] = rank
        nbr_matrix = np.empty((n_rows, n_neighbors), dtype=np.int64)
        nbr_matrix[:, :] = -1
        for i in range(n_rows):
            row_nbr_dict = {
                j: rank
                for j, rank in rank_dict[i].items()
            }
            neighbors = list( 
                sorted(row_nbr_dict, key=lambda x: row_nbr_dict[x], reverse=False)
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


    
