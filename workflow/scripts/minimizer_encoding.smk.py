from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from snakemake_stub import *


import gzip, json, collections
from typing import Sequence, Mapping, Collection
from Bio import SeqIO
import scipy.sparse as sp
import numpy as np
import pandas as pd
import mmh3
from collections import deque

def init_reverse_complement():
    TRANSLATION_TABLE = str.maketrans("ACTGactg", "TGACtgac")

    def reverse_complement(sequence: str) -> str:
        """
        >>> reverse_complement("AATC")
        'GATT'
        >>> reverse_complement("CCANT")
        'ANTGG'
        """
        sequence = str(sequence)
        return sequence.translate(TRANSLATION_TABLE)[::-1]

    return reverse_complement


reverse_complement = init_reverse_complement()


def load_reads(fasta_path: str):
    read_sequences = []
    read_names = []
    read_orientations = []

    with gzip.open(fasta_path, "rt") as handle:  # Open gzipped file in text mode
        for record in SeqIO.parse(handle, "fasta"):
            seq = str(record.seq)
            read_sequences.append(seq)
            read_names.append(record.id)
            read_orientations.append("+")

            # Include reverse complement
            read_sequences.append(reverse_complement(seq))
            read_names.append(record.id)
            read_orientations.append("-")

    return read_names, read_orientations, read_sequences


def get_minimizers(sequence: str,
                    k: int,
                    w: int,
                    hash_seed: int):
    """
    优化后的 minimizer 提取函数，时间复杂度 O(n)。

    参数:
    sequence (str): 输入序列。
    k (int): k-mer 的长度。
    w (int): 窗口大小（窗口内包含 w 个连续的 k-mer)。

    返回:
    list: 所有 minimizer 的列表，连续相同的 minimizer 只记录一次。
    dict: minimizer 的频率统计。
    """
    n = len(sequence)
    if n < k or w < 1:
        return [], {}

    # 预计算所有 k-mer 的哈希值
    hashes = []
    for i in range(n - k + 1):
        kmer = sequence[i:i+k]
        hashes.append(mmh3.hash(kmer,hash_seed))

    # 使用双端队列维护最小值索引
    dq = deque()
    minimizer_counter = collections.Counter()

    # 用于记录上一个 minimizer
    previous_minimizer_index = -1

    for i, current_hash in enumerate(hashes):
        # 移除队列中超出窗口范围的索引
        while dq and dq[0] <= i - w:
            dq.popleft()

        # 移除队列中比当前哈希大的索引（保持单调递增）
        while dq and hashes[dq[-1]] > current_hash:
            dq.pop()

        dq.append(i)

        # 当窗口填满后，记录当前窗口的最小值
        if i >= w - 1:
            minimizer_index = dq[0]

            # 如果当前 minimizer 与上一个 minimizer 不同，则记录
            if minimizer_index != previous_minimizer_index:
                minimizer = sequence[minimizer_index:minimizer_index+k]
                minimizer_counter[minimizer] += 1
                previous_minimizer_index = minimizer_index

    return minimizer_counter

def build_minimizer_matrix(read_sequences: Sequence[str], 
                           k: int, 
                           w: int,
                           min_multiplicity: int,
                           hash_seed: int
                           ):
    """
    构建 minimizer 矩阵。

    参数:
    sequences (list): 输入序列列表。
    k (int): k-mer 的长度。
    w (int): 窗口大小。

    返回:
    np.ndarray: 行为序列、列为 minimizer 的矩阵。
    dict: minimizer 到序号的映射。
    """
    # 提取所有 minimizer 并构建映射
    minimizer_to_index = {}
    current_index = 0
    minimizer_counters = []
    
    for seq in read_sequences:
        minimizer_counter = get_minimizers(seq, k, w, hash_seed)
        minimizer_counters.append(minimizer_counter)
        # 更新 minimizer 到序号的映射
        for minimizer in minimizer_counter:
            if minimizer not in minimizer_to_index:
                minimizer_to_index[minimizer] = current_index
                current_index += 1

    # 构建矩阵
    read_features = []
    num_sequences = len(read_sequences)
    num_minimizers = len(minimizer_to_index)
    print(f'Minimizer number: {num_minimizers}')
    row_ind = []
    col_ind = []
    data = []
    for i, minimizer_counter in enumerate(minimizer_counters):
        feature_i = []
        for minimizer, freq in minimizer_counter.items():
            j = minimizer_to_index[minimizer]
            feature_i.append(j)
            row_ind.append(i)
            col_ind.append(j)
            data.append(freq)
        read_features.append(feature_i)
    feature_matrix = sp.csr_matrix((data, (row_ind, col_ind)), shape=(num_sequences, num_minimizers))

    # 保留非零元素数 >= min_multiplicity 的列 只适用于组装场景
    col_nonzero = np.bincount(feature_matrix.indices, minlength=feature_matrix.shape[1])
    mask = col_nonzero >= min_multiplicity
    filtered_matrix = feature_matrix[:, mask]
    print(f'feature_matrix shape: {filtered_matrix.shape}')

    return filtered_matrix,read_features,minimizer_to_index


def encode_reads(
    fasta_path: str,
    k,
    w,
    *,
    min_multiplicity: int,
    hash_seed: int = 51261,
):
    """
    Encodes sequencing reads from a gzipped FASTA file as a sparse matrix.
    """
    # Load reads info

    # Load reads
    read_names, read_orientations, read_sequences = load_reads(fasta_path=fasta_path)

    # Build matrix
    feature_matrix,read_features,_ = build_minimizer_matrix(
        read_sequences=read_sequences,
        k=k,
        w=w,
        min_multiplicity=min_multiplicity,
        hash_seed=hash_seed
    )


    return feature_matrix, read_features

def main(snakemake: "SnakemakeContext"):

    input_fasta_file = snakemake.input["fasta"]

    output_npz_file = snakemake.output["npz"]
    output_json_file = snakemake.output["json"]
    min_multiplicity = snakemake.params['min_multiplicity']
    k = int(snakemake.wildcards["k"])
    w = int(snakemake.wildcards["w"])
    seed = snakemake.params["seed"]

    feature_matrix, read_features = encode_reads(
        fasta_path=input_fasta_file,
        k=k,
        w=w,
        min_multiplicity=min_multiplicity,
        hash_seed=seed
    )
    sp.save_npz(output_npz_file, feature_matrix)
    with gzip.open(output_json_file, "wt") as f:
        json.dump(read_features, f)

if __name__ == "__main__":
    main(snakemake)
