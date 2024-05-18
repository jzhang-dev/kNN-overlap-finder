from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse._csr import csr_matrix
from typing import Sequence, Type
from dataclasses import dataclass, field

from nearest_neighbors import _NearestNeighbors, get_overlap_candidates
from dim_reduction import SpectralMatrixFree
from graph import ReadGraph, get_read_graph_statistics
from align import cWeightedSemiglobalAligner, run_multiprocess_alignment


@dataclass
class NearestNeighborsConfig:
    method: Type[_NearestNeighbors]
    use_tfidf: bool = False
    dim_reduction: int | None = None
    n_neighbors: int = 20
    kw: dict = field(default_factory=dict)
    overlap_candidates: Sequence[tuple[int, int]] | None = field(
        default=None, repr=False
    )
    graph: ReadGraph | None = field(default=None, repr=False)
    time: float | None = None
    pre_align_stats: dict | None = None
    post_align_stats: dict | None = None

    def run(self, data, read_ids: Sequence[int]):
        import time

        start_time = time.time()
        if self.use_tfidf:
            data = TfidfTransformer(use_idf=True, smooth_idf=True).fit_transform(data)
        if self.dim_reduction is not None:
            reducer = SpectralMatrixFree(self.dim_reduction)
            reducer.fit(data)
            _, data = reducer.transform()
        neighbor_indices = self.method(data).get_neighbors(
            n_neighbors=self.n_neighbors, **self.kw
        )
        elapsed_time = time.time() - start_time
        self.time = elapsed_time
        self.overlap_candidates = get_overlap_candidates(
            neighbor_indices, read_ids, n_neighbors=self.n_neighbors
        )

    def evaluate_pre_alignment(self, reference_graph: ReadGraph):
        if self.overlap_candidates is None:
            raise TypeError()
        graph = ReadGraph.from_overlap_candidates(self.overlap_candidates)
        result = get_read_graph_statistics(graph, reference_graph=reference_graph)
        self.pre_align_stats = result

    def evaluate_post_alignment(
        self,
        read_features,
        feature_weights,
        reference_graph: ReadGraph,
        *,
        n_neighbors=6,
        min_alignment_score=0,
        processes=8,
        _cache: dict = {},
    ):
        if self.overlap_candidates is None:
            raise TypeError()
        candidates = self.overlap_candidates

        alignment_dict = run_multiprocess_alignment(
            candidates,
            read_features,
            feature_weights,
            aligner=cWeightedSemiglobalAligner,
            align_kw=dict(max_cells=int(1e9)),
            traceback=False,
            processes=processes,
            batch_size=200,
            min_free_memory_gb=48,
            max_total_wait_seconds=600,
            _cache=_cache,
        )

        graph = ReadGraph.from_pairwise_alignment(
            candidates,
            alignment_dict,
            n_neighbors=n_neighbors,
            min_alignment_score=min_alignment_score,
        )
        self.graph = graph
        result = get_read_graph_statistics(graph, reference_graph=reference_graph)
        self.post_align_stats = result
