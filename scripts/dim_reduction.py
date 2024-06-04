import scipy as sp
import numpy as np

# TODO: weighted_by_sd

class SpectralMatrixFree:
    """Matrix-free spectral embedding without computing the similarity matrix explicitly.

    Only cosine similarity is supported.

    Adapted from https://github.com/kaizhang/SnapATAC2/blob/51f040c095820fea43e9a6360d751bfc29faecc5/snapatac2-python/python/snapatac2/tools/_embedding.py#L434
    """
    def __init__(
        self,
        out_dim: int = 30,
        feature_weights = None,
    ):
        self.out_dim = out_dim
        self.feature_weights = feature_weights

    def fit(self, mat):
        if self.feature_weights is not None:
            mat = mat @ sp.sparse.diags(self.feature_weights)
        self.sample = mat
        self.in_dim = mat.shape[1]

        s = 1 / np.sqrt(np.ravel(sp.sparse.csr_matrix.power(mat, 2).sum(axis = 1)))
        X = sp.sparse.diags(s) @ mat

        D = np.ravel(X @ X.sum(axis = 0).T) - 1
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


