from scipy.sparse._csr import csr_matrix
import pynndescent
import numpy as np
from dataclasses import dataclass, field

@dataclass
class _NearestNeighbors:
    def get_neighbors(
        self, data: csr_matrix | np.ndarray, n_neighbors: int
    ) -> np.ndarray:
        raise NotImplementedError()


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
        _nbr_indices, _ = index.neighbor_graph
        nbr_indices = _nbr_indices[:,:n_neighbors]
        return nbr_indices
