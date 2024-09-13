#!/usr/bin/python3
# -*- coding: utf-8 -*-

import collections
from math import ceil
import time
from dataclasses import dataclass
from typing import Sequence, Collection, Mapping, Any, Type, MutableMapping
import psutil
import numpy as np
import sharedmem


import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.join(current_dir, "..", "lib")
print(lib_dir)
sys.path.append(lib_dir)
#import Aligner as cAligner  # type: ignore


@dataclass(init=False)
class Seq:
    elements: Sequence[int]
    positions: Sequence[float]
    weights: Sequence[float]
    length: int
    read_id: int | None = None

    def __init__(self, elements, positions=None, weights=None, length=None):
        if len(elements) == 0:
            raise ValueError()
        self.elements = elements

        if weights is None:
            # 自动计算 weights ； 仅供测试使用
            self.weights = np.ones(len(self.elements)).tolist()
        else:
            self.weights = weights

        if length is None:
            self.length = sum(self.weights)
        else:
            self.length = length

        if positions is None:
            # 自动计算 positions ； 仅供测试使用
            self.positions = np.cumsum([0] + list(self.weights))[:-1].tolist()
        else:
            self.positions = positions

        if not len(self.elements) == len(self.positions) == len(self.weights):
            raise ValueError()

    def __getitem__(self, indices: Sequence[int]) -> "Seq":
        sample_elements = [self.elements[i] for i in indices]
        sample_positions = [self.positions[i] for i in indices]
        sample_weights = [self.weights[i] for i in indices]
        return type(self)(
            length=self.length,
            elements=sample_elements,
            positions=sample_positions,
            weights=sample_weights,
        )

    def sample(self, elements: Collection[int]) -> "Seq":
        indices = [i for i, x in enumerate(self.elements) if x in elements]
        return self[indices]

    def __len__(self):
        return len(self.elements)


@dataclass
class AlignmentResult:
    score: int
    indices_1: Sequence[int]
    indices_2: Sequence[int]
    dp_matrix: np.ndarray | None = None
    arrow_matrix: np.ndarray | None = None

    @classmethod
    def from_empty(cls):
        return cls(0, [], [], None, None)

    def __iter__(self):
        yield self.score
        yield self.indices_1
        yield self.indices_2
        yield self.dp_matrix
        yield self.arrow_matrix
    
    @property
    def has_traceback(self) -> bool:
        if self.score <= 0: # No traceback needed
            return True
        if len(self.indices_1) > 0 and len(self.indices_2) > 0:
            return True
        return False

@dataclass
class _PairwiseAligner:
    seq1: Seq
    seq2: Seq

    @staticmethod
    def _get_cell_score(dp_matrix, i, j, x1, x2, w1, w2):
        # match
        if x1 == x2:
            match_score = dp_matrix[i - 1, j - 1] + max(w1, w2)
        else:
            match_score = dp_matrix[i - 1, j - 1] - max(w1, w2)

        # gaps
        gap1_score = dp_matrix[i - 1, j] - w1
        gap2_score = dp_matrix[i, j - 1] - w2

        max_score = max(match_score, gap1_score, gap2_score)

        if match_score == max_score:
            arrow = 1
        elif gap1_score == max_score:
            arrow = 2
        else:
            arrow = 3

        return max_score, arrow

    @staticmethod
    def _dp_traceback(arrow_matrix, i, j):
        aln1, aln2 = [], []
        while i > 0 and j > 0:
            arrow = arrow_matrix[i, j]
            if arrow == 1:
                aln1.append(i - 1)
                aln2.append(j - 1)
                i -= 1
                j -= 1
            elif arrow == 2:
                aln1.append(i - 1)
                aln2.append(-1)
                i -= 1
            elif arrow == 3:
                aln1.append(-1)
                aln2.append(j - 1)
                j -= 1
            else:
                raise ValueError()

        aln1.reverse()
        aln2.reverse()
        return aln1, aln2

    def align(self):
        raise NotImplementedError()


class WeightedSemiglobalAligner(_PairwiseAligner):
    def align(self, traceback=True):
        seq1, seq2 = self.seq1, self.seq2
        s1, s2 = seq1.elements, seq2.elements
        l1, l2 = len(s1), len(s2)
        weights_1 = seq1.weights
        weights_2 = seq2.weights

        # Initialize the DP matrix
        dp_matrix = np.zeros((l1 + 1, l2 + 1), dtype=np.int64)
        arrow_matrix = np.zeros(dp_matrix.shape, dtype=np.uint8)

        # Initialize the DP matrix
        dp_matrix = np.zeros((l1 + 1, l2 + 1), dtype=np.int64)
        arrow_matrix = np.zeros(dp_matrix.shape, dtype=np.uint8)

        # Fill the DP matrix
        for i in range(1, l1 + 1):
            x1 = s1[i - 1]
            w1 = weights_1[i - 1]
            current_row = dp_matrix[i, :]

            def fill_cell(j):
                x2 = s2[j - 1]
                w2 = weights_2[j - 1]
                score, arrow = self._get_cell_score(dp_matrix, i, j, x1, x2, w1, w2)
                current_row[j] = score
                arrow_matrix[i, j] = arrow
                return score

            for j in range(1, l2 + 1):
                fill_cell(j)

        # Traceback to find the optimal alignment
        # Start at the max value in the last row or column
        i = np.argmax(dp_matrix[:, -1])
        j = np.argmax(dp_matrix[-1, :])
        last_col_max_score = dp_matrix[i, -1]
        last_row_max_score = dp_matrix[-1, j]
        alignment_score = max(last_col_max_score, last_row_max_score)

        if last_col_max_score >= last_row_max_score:
            j = l2
        else:
            i = l1

        if traceback:
            aln1, aln2 = self._dp_traceback(arrow_matrix, i, j)
        else:
            aln1, aln2 = None, None

        return alignment_score, aln1, aln2, dp_matrix, arrow_matrix


class cWeightedSemiglobalAligner(_PairwiseAligner):
    """
    C++ implementation of `WeightedSemiglobalAligner`
    """

    def align(self, traceback=True, max_cells=int(1e9)) -> AlignmentResult:
        c_aligner = cAligner.AlignerWrapper()
        seq1, seq2 = self.seq1, self.seq2
        if len(seq1) * len(seq2) > max_cells:
            return AlignmentResult.from_empty()
        weights_1 = seq1.weights
        weights_2 = seq2.weights
        s1, s2 = seq1.elements, seq2.elements
        try:
            alignment_score, aln1, aln2 = c_aligner.align(
                s1, s2, weights_1, weights_2, traceback
            )
        except Exception:
            raise RuntimeError(f"Alignment failed. {seq1.read_id=} {seq2.read_id=}")
        dp_matrix, arrow_matrix = None, None
        return AlignmentResult(alignment_score, aln1, aln2, dp_matrix, arrow_matrix)



def wait_for_memory(
    min_free_memory_gb, *, mean_step_wait_seconds=10, max_total_wait_seconds=30
):
    """
    Block until at least `min_free_memory_gb` GB of memory is free or `max_wait_time` seconds have passed.

    Parameters:
    - min_free_memory_gb (float): The minimum amount of free memory in GB that we are waiting for.
    - max_wait_time (float): The maximum amount of time in seconds to wait.

    Returns:
    - bool: True if the required memory is available, False if the max wait time elapsed.
    """
    start_time = time.time()
    min_free_memory_bytes = min_free_memory_gb * (1024**3)

    while True:
        # Check the available memory
        available_memory = psutil.virtual_memory().available

        if available_memory >= min_free_memory_bytes:
            # print(f"Enough memory available: {available_memory / (1024 ** 3):.2f} GB")
            return True

        # Check if the maximum wait time has passed
        elapsed_time = time.time() - start_time
        if elapsed_time > max_total_wait_seconds:
            # print("Maximum wait time elapsed.")
            return False

        # Wait a bit before checking again
        time.sleep(np.random.random() * 2 * mean_step_wait_seconds)


def run_multiprocess_alignment(
    candidates: Sequence,
    read_features,
    feature_weights,
    *,
    aligner: Type[_PairwiseAligner],
    align_kw: Mapping = {},
    traceback=False,
    processes=4,
    batch_size=100,
    min_free_memory_gb=32,
    max_total_wait_seconds=120,
    mean_step_wait_seconds=None,
    shuffle=True,
    seed=1,
    verbose=True,
    _cache: MutableMapping[tuple[int, int], AlignmentResult] | None = None,
    _update_cache: bool = True,
) -> Mapping[tuple[int, int], AlignmentResult]:
    if mean_step_wait_seconds is None:
        mean_step_wait_seconds = processes

    if _cache is not None:
        
        cached_candidates = set(_cache) & set(candidates)
        if traceback:
            invalid_cache = set()
            for x in cached_candidates:
                cached_result = _cache[x]
                if cached_result.score > 0 and not cached_result.has_traceback:
                    invalid_cache.add(x)
            cached_candidates -= invalid_cache
        new_candidates = list(set(candidates) - cached_candidates)
        if verbose:
            print(f"{len(candidates)=}\t{len(cached_candidates)=}\t{len(new_candidates)=}")
        candidates = new_candidates

    if shuffle:
        candidates = list(candidates).copy()
        np.random.default_rng(seed).shuffle(candidates)
        
    candidate_count = len(candidates)
    alignment_scores = sharedmem.empty(candidate_count, dtype="int64")
    
    with sharedmem.MapReduce(np=processes) as pool:

        def align(i):

            k1, k2 = candidates[i]
            s1 = read_features[k1]
            s2 = read_features[k2]
            seq1 = Seq(s1, weights=[feature_weights[x] for x in s1])
            seq2 = Seq(s2, weights=[feature_weights[x] for x in s2])
            seq1.read_id = k1  # DEBUG
            seq2.read_id = k2
            wait_for_memory(
                min_free_memory_gb=min_free_memory_gb,
                max_total_wait_seconds=max_total_wait_seconds,
                mean_step_wait_seconds=mean_step_wait_seconds,
            )
            result = aligner(seq1, seq2).align(**align_kw, traceback=traceback)  # type: ignore
            return result

        def work(i0):
            output_size = len(alignment_scores)
            aligned_indices = []
            for i in range(i0, i0 + batch_size):
                if i >= output_size:
                    break

                alignment_scores[i] = -1  # DEBUG
                result = align(i)
                alignment_scores[i] = result.score
                if traceback:
                    aligned_indices.append((result.indices_1, result.indices_2))
            return i0, aligned_indices

        finished = 0
        alignment_dict = {}

        def reduce(i0, batch_aligned_indices):
            nonlocal alignment_dict
            for i in range(i0, i0 + batch_size):
                if i >= candidate_count:
                    break
                if traceback:
                    indices_1, indices_2 = batch_aligned_indices[i - i0]
                else:
                    indices_1, indices_2 = [], []
                score = int(alignment_scores[i])
                k1, k2 = candidates[i]
                alignment_dict[(k1, k2)] = AlignmentResult(
                    score, indices_1=indices_1, indices_2=indices_2
                )

            nonlocal finished
            finished += batch_size
            if verbose:
                delta_time = time.time() - start_time
                speed = finished / delta_time
                print(i0, f"{speed:.2f}", sep="\t", end="\t\t")

        if verbose:
            start_time = time.time()
        try:
            pool.map(work, range(0, candidate_count, batch_size), reduce=reduce)
        except Exception as e:
            print(e)
            unfinished = (alignment_scores == -1).nonzero()[0].tolist()
            raise RuntimeError(f"Alignment failed. Unfinished: {unfinished}")
    if verbose:
        print("")
    if _cache is not None:
        alignment_dict.update({x: _cache[x] for x in cached_candidates})
    if _update_cache and candidate_count > 0 and _cache is not None:
        _cache.update(alignment_dict)

    return alignment_dict



def test_alignment():
    print("Test simple alignment")
    aligners = [
        WeightedSemiglobalAligner,
        cWeightedSemiglobalAligner,
    ]
    s1 = [2, 3, 4, 5, 6]
    s2 = [1, 2, 3, 4, 5]
    seq1, seq2 = Seq(s1), Seq(s2)

    for aligner in aligners:
        kw: Mapping[str, Any] = dict(traceback=True)
        alignment_score, aln1, aln2, _, _ = aligner(seq1, seq2).align(**kw)
        assert alignment_score == 4
        assert aln1 == [0, 1, 2, 3]
        assert aln2 == [1, 2, 3, 4]

    print("Test unalignable")
    aligners = [
        WeightedSemiglobalAligner,
        cWeightedSemiglobalAligner,
    ]
    s1 = [2, 3, 4, 5, 6]
    s2 = [7, 8, 9, 10, 11]
    seq1, seq2 = Seq(s1), Seq(s2)

    for aligner in aligners:
        kw: Mapping[str, Any] = dict(traceback=True)
        alignment_score, aln1, aln2, _, _ = aligner(seq1, seq2).align(**kw)
        assert alignment_score == 0
        assert aln1 == []
        assert aln2 == []
