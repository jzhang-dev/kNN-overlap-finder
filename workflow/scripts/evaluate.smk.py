import sys
from pathlib import Path




import pickle, os, gzip, json
from importlib import reload
from dataclasses import dataclass, field
import collections
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pysam
import pynndescent
from sklearn.feature_extraction.text import TfidfTransformer
import umap
import scipy as sp



project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root / "scripts"))

from data_io import is_fwd_id, get_fwd_id, get_sibling_id # type: ignore

from nearest_neighbors import (
    ExactNearestNeighbors,
    NNDescent,
    WeightedLowHash,
) # type: ignore


from graph import ReadGraph, GenomicInterval # type: ignore

from truth import get_overlaps # type: ignore

from evaluate import NearestNeighborsConfig, mp_evaluate_configs # type: ignore



from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from snakemake_stub import *








def main(snakemake: "SnakemakeContext"):
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["figure.dpi"] = 300

    sample = snakemake.wildcards['sample']
    dataset = snakemake.wildcards['dataset']
    region = snakemake.wildcards['region']

    npz_path = snakemake.input['feature_matrix']
    tsv_path = snakemake.input['metadata']
    json_path = snakemake.input['read_feature']

    meta_df = pd.read_table(tsv_path).reset_index()
    read_indices = {read_name: read_id for read_id, read_name in meta_df['read_name'].items()}

    feature_matrix = sp.sparse.load_npz(npz_path)[meta_df.index, :]

    with gzip.open(json_path, "rt") as f:
        read_features = json.load(f)
        read_features = {i: read_features[i] for i in meta_df.index}

    feature_weights = {i: 1 for i in range(feature_matrix.shape[1])}




if __name__ == "__main__":
    main(snakemake)
