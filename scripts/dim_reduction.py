from dataclasses import dataclass
import scipy as sp
import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse import csr_matrix,isspmatrix_coo
from sklearn.preprocessing import normalize as normalize_function
import anndata
import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn import random_projection
from sklearn.manifold import Isomap  
import umap
from sklearn.decomposition import PCA
import sharedmem
import collections
from memory_profiler import profile
from typing import Union
import gc
from joblib import Parallel, delayed
import os
from sklearn.utils.extmath import safe_sparse_dot
import sharedmem
import math


class _SpectralMatrixFree:
    """
    Perform dimension reduction using Laplacian Eigenmaps.

    Matrix-free spectral embedding without computing the similarity matrix explicitly.

    Only cosine similarity is supported.

    Adapted from https://github.com/kaizhang/SnapATAC2/blob/51f040c095820fea43e9a6360d751bfc29faecc5/snapatac2-python/python/snapatac2/tools/_embedding.py#L434
    """

    def __init__(
        self,
        out_dim: int = 30,
        feature_weights=None,
    ):
        self.out_dim = out_dim
        self.feature_weights = feature_weights

    def fit(self, mat):
        if self.feature_weights is not None:
            mat = mat @ sp.sparse.diags(self.feature_weights)
        self.sample = mat
        self.in_dim = mat.shape[1]

        s = 1 / np.sqrt(np.ravel(sp.sparse.csr_matrix.power(mat, 2).sum(axis=1)))
        X = sp.sparse.diags(s) @ mat

        D = np.ravel(X @ X.sum(axis=0).T) - 1
        X = sp.sparse.diags(1 / np.sqrt(D)) @ X
        evals, evecs = self._eigen(X, 1 / D, k=self.out_dim)

        ix = evals.argsort()[::-1]
        self.evals = evals[ix]
        self.evecs = evecs[:, ix]

    def transform(self, weighted_by_sd: bool = True):
        evals = self.evals
        evecs = self.evecs

        if weighted_by_sd:
            idx = [i for i in range(evals.shape[0]) if evals[i] > 0]
            evals = evals[idx]
            evecs = evecs[:, idx] * np.sqrt(evals)
        return evals, evecs

    @staticmethod
    def _eigen(X, D, k):
        def f(v):
            return X @ (v.T @ X).T - D * v

        n = X.shape[0]
        A = sp.sparse.linalg.LinearOperator((n, n), matvec=f, dtype=np.float64)
        return sp.sparse.linalg.eigsh(A, k=k)


class _DimensionReduction:

    def transform(self, data: csr_matrix | NDArray, n_dimensions: int) -> NDArray:
        raise NotImplementedError


class SpectralEmbedding(_DimensionReduction):
    def transform(
        self, data: csr_matrix | NDArray, n_dimensions: int, weighted_by_sd: bool = True
    ) -> NDArray:
        reducer = _SpectralMatrixFree(out_dim=n_dimensions)
        reducer.fit(data)
        _, embedding = reducer.transform(weighted_by_sd=weighted_by_sd)
        return embedding
    
class TruncatedSVD(_DimensionReduction):
    def transform(
        self, data: csr_matrix | NDArray, n_dimensions: int):
        reducer = TruncatedSVD(n_components=n_dimensions)
        embedding = reducer.fit(data)
        return embedding
    
class PCA(_DimensionReduction):
    def transform(
        self, data: csr_matrix | NDArray, n_dimensions: int):
        pca = sklearn.decomposition.PCA(n_components=n_dimensions) 
        dim_redu = pca.fit_transform(data)  
        return dim_redu

class isomap(_DimensionReduction):
    def transform(self,data,n_dimensions):
        isomap = Isomap(n_components=n_dimensions)  
        dim_redu = isomap.fit_transform(data) 
        return dim_redu

class umapEmbedding():
    def transform(
        self, data: csr_matrix | NDArray, n_dimensions: int):
        umap_model = umap.umap_.UMAP(n_components=n_dimensions)
        embedding = umap_model.fit_transform(data)
        return embedding

class GaussianRandomProjection(_DimensionReduction):
    def transform(
        self, data: csr_matrix | NDArray, n_dimensions: int):
        reducer = random_projection.GaussianRandomProjection(n_components=n_dimensions)
        embedding = reducer.fit_transform(data)
        return embedding



class mp_SparseRandomProjection:
    def _make_random_matrix(
        self, n_components, n_features, density: float, seed: int = 2094
    ):
        rng = np.random.default_rng(seed)
        indices = []
        offset = 0
        indptr = [offset]
        for _ in range(n_components):
            # find the indices of the non-zero components for row i
            n_nonzero_i = rng.binomial(n_features, density)
            indices_i = rng.choice(n_features, n_nonzero_i, replace=False)
            indices.append(indices_i)
            offset += n_nonzero_i
            indptr.append(offset)

        indices = np.concatenate(indices)

        # Among non zero components the probability of the sign is 50%/50%
        data = rng.binomial(1, 0.5, size=np.size(indices)) * 2 - 1

        # build the CSR structure by concatenating the rows
        components = csr_matrix(
            (data, indices, indptr), shape=(n_components, n_features), dtype=np.float32
        )
        return np.sqrt(1 / density) / np.sqrt(n_components) * components

    def transform(
        self,
        data: csr_matrix | NDArray,
        n_dimensions: int,
        density: float | str = "auto",
        batch_size: int = 10000,
        seed: int = 521022,
        threads: int = 1,
    ) -> NDArray:
        assert data.shape is not None
        if density == "auto":
            _density = 1 / math.sqrt(data.shape[1])
        else:
            assert isinstance(density, float) and 0 < density <= 1
            _density = density

        if isspmatrix_coo(data):
            data = data.tocsr()  
            
        random_matrix = self._make_random_matrix(
            n_components=n_dimensions,
            n_features=data.shape[1],
            density=_density,
            seed=seed,
        )

        with sharedmem.MapReduce(np=threads) as pool:
            embeddings = sharedmem.empty(
                (data.shape[0], n_dimensions), dtype=random_matrix.dtype
            )
            def work(i0):
                batch_data = data[i0 : i0 + batch_size]
                batch_embeddings = safe_sparse_dot(batch_data, random_matrix.T, dense_output=True)
                embeddings[i0 : i0 + batch_size] = batch_embeddings

                return i0

            pool.map(work, range(0, data.shape[0], batch_size))

        assert isinstance(embeddings, np.ndarray)
        return embeddings


class mp_SparseRandomProjection_former:
    def _fit_transformer(self,
                         n_features: int,
                         n_dimensions,
                         density,
                         random_state
                         ):
        """初始化 scikit-learn 的投影器（仅需特征维度）"""
        self.transformer_ = random_projection.SparseRandomProjection(
            n_components=n_dimensions,
            density=density,
            random_state=random_state
        )
        # 用虚拟数据拟合（避免加载全矩阵）
        dummy_data = csr_matrix((1, n_features))
        self.transformer_.fit(dummy_data)


    def _process_batch(self, temp_dir, batch_idx: int, batch_data: csr_matrix) -> str:
        # 降维计算
        _reduced = self.transformer_.transform(batch_data)
        reduced = _reduced.toarray()
        random_num = np.random.randint(1,10000,size=1)[0]
        # 保存降维结果到临时文件
        output_path = os.path.join(temp_dir ,f"tmp{random_num}_batch_{batch_idx}.npy")
        np.save(output_path, reduced)
        
        # 立即释放内存
        del batch_data, reduced
        gc.collect()
        
        return output_path

    def transform(self,
                  data: csr_matrix | NDArray,
                  n_dimensions: int,
                  random_state: int = 521022,
                  density: float ='auto',
                  temp_dir: str = "./temp",
                  batch_size: int = 10000,
                  n_jobs: int = 1
                  ) -> np.ndarray:
        # 初始化投影器
        os.makedirs(temp_dir, exist_ok=True)
        n_samples, n_features = data.shape
        self._fit_transformer(n_features,n_dimensions,density,random_state)
        
        # 分块处理
        batch_indices = list(range(0, n_samples, batch_size))
        
        # 分块处理
        print(data.format)
        if isspmatrix_coo(data):
                data = data.tocsr()  
        batch_indices = range(0, n_samples, batch_size)
        batch_paths = Parallel(n_jobs=n_jobs)(
            delayed(self._process_batch)( temp_dir, idx // batch_size ,data[idx: idx + batch_size])
            for idx in batch_indices
        )
        # 合并所有分块结果
        result = np.vstack([np.load(path,allow_pickle=True) for path in batch_paths])
        print(result.shape)
        # 清理临时文件
        for path in batch_paths:
            os.remove(path)
        return result
def split_sparse_matrix_by_rows(sparse_matrix: csr_matrix, num_splits: int = 10):
    """
    将稀疏矩阵按行分成指定份数。
    
    参数:
        sparse_matrix: scipy.sparse.csr_matrix 或其他稀疏矩阵格式
        num_splits: 分割成的份数，默认为10
    返回:
        splits: 包含分割后的稀疏矩阵的列表
    """
    total_rows = sparse_matrix.shape[0]

    rows_per_split = total_rows // num_splits
    remainder = total_rows % num_splits
    
    splits = []
    
    start_idx = 0
    
    for i in range(num_splits):
        if i < remainder:
            end_idx = start_idx + rows_per_split + 1
        else:
            end_idx = start_idx + rows_per_split
        split_matrix = sparse_matrix[start_idx:end_idx, :]
        splits.append(split_matrix)
        start_idx = end_idx
    
    return splits

class Split_GRP(_DimensionReduction):
    @profile
    def transform(
        self, data: csr_matrix | NDArray, 
        n_dimensions: int,
        n_split: int = 20, 
        processes: int = 1):

        splits = split_sparse_matrix_by_rows(data,n_split)
        reduced_part_fm = collections.defaultdict()
        
        with sharedmem.MapReduce(np=processes) as pool:

            def work(i):
                part_fm = splits[i]
                print(f'processing part{i}, shape: {part_fm.shape}')
                reducer = random_projection.GaussianRandomProjection(n_components=n_dimensions)
                part_embedding = reducer.fit_transform(part_fm)
                return i, part_embedding

            def reduce(i, part_embedding):
                reduced_part_fm[i] = part_embedding

            pool.map(work, range(n_split), reduce=reduce)

        embedding = np.vstack([reduced_part_fm[i] for i in range(n_split)])
        return embedding

class SparseRP(_DimensionReduction):
    def transform(
        self, data: csr_matrix | NDArray, n_dimensions: int,density: float ='auto'):
        reducer = random_projection.SparseRandomProjection(n_components=n_dimensions, density=density)
        embedding = reducer.fit_transform(data)
        _embedding = embedding.toarray()
        return _embedding

class SimHash_Dimredu(_DimensionReduction):
    @staticmethod
    def _get_hash_table(
        feature_count: int, n_dimensions:int, seed: int
    ) -> NDArray[np.int8]:
        assert n_dimensions % 8 == 0, "Error: n_dimensions must be divisible by 8."

        rng = np.random.default_rng(seed)
        hash_table = rng.integers(
            0, 2, size=(feature_count, n_dimensions), dtype=np.int8
        )
        hash_table = hash_table * 2 - 1
        return hash_table
    
    def transform(self,data: NDArray | csr_matrix, n_dimensions = 3200 ,seed = 20141025):
        hash_table = self._get_hash_table(data.shape[1],n_dimensions,seed)
        simhash = (data @ hash_table).astype(np.uint8) 
        return simhash

class scBiMapEmbedding(_DimensionReduction):
    """
    From scBiMapping on Conda
    Author: Teng Qiu
    """

    def transform(
        self,
        data: csr_matrix | NDArray,
        n_dimensions: int,
        *,
        normalize=True, 
    ) -> NDArray:
        if data.min() < 0:
            raise ValueError(
                "The input matrix is regarded as a similarity matrix and thus should not contain negtive values"
            )
        eps=0.0000000001,
        Dx = sparse.diags(np.ravel(1 / (data.sum(axis=1) + eps)))

        y = (Dx @ csr_matrix(data.sum(axis=1))).T  # row vector
        Dy = sparse.diags(np.ravel((1 / (np.sqrt((y @ data).T.toarray()) + eps))))  # type: ignore
        C = np.sqrt(Dx) @ data @ Dy  # type: ignore
        _, _, evec = sparse.linalg.svds( # type: ignore
            C, k=n_dimensions, return_singular_vectors="vh", random_state=0
        )
        V = Dy @ evec.T  # eigenvectors for features
        U = Dx @ data @ V  # eigenvectors for cells

        if normalize:
            U = normalize_function(U, axis=1, norm="l2")
        return U  # type: ignore
