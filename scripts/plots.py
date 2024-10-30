from typing import Sequence, Mapping
import os
import networkx as nx
from data_io import is_fwd_id, get_fwd_id

import pandas as pd
import sharedmem
import scanpy as sc
import matplotlib.pyplot as plt
import anndata as ad

from graph import OverlapGraph


def get_graphviz_layout(graph, method="neato", figsize=(8, 6), seed=43, **kw):
    fig_width, fig_height = figsize
    graph_attributes = dict(
        dim=2, dimen=2, size=f'"{fig_width},{fig_height}"', start=seed, smoothing="none"
    )
    graph_attributes.update(kw)
    args = " ".join(f"-G{key}={value}" for key, value in graph_attributes.items())
    pos = nx.nx_agraph.graphviz_layout(graph, prog=method, args=args)
    return pos


def get_adjacency_matrix(graph):
    G = graph
    adj_matrix = nx.to_scipy_sparse_array(G)
    nodes = list(G.nodes)
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            node1, node2 = nodes[i], nodes[j]
            if G.has_edge(node1, node2):
                score = G[node1][node2]["alignment_score"]
                adj_matrix[i, j] = score


def get_umap_layout(graph, *, min_distance=1, random_state=0, verbose=False):
    # From Teng Qiu
    adj_matrix = nx.to_scipy_sparse_array(graph)

    mdata = ad.AnnData(obs=list(range(adj_matrix.shape[0])))  # type: ignore
    mdata.obsp["connectivities"] = adj_matrix

    if adj_matrix.max() > 10:
        print(
            "\033[0;33;40m",
            "Warning:  the values of Connectivity Matrix (representing the weights) are too large",
            "\033[0m",
        )
    mdata.uns["neighbors"] = {
        "connectivities_key": "connectivities",
        "params": {"method": "umap"},
    }

    sc.tl.umap(mdata, min_dist=min_distance, random_state=random_state)
    embeddings = mdata.obsm["X_umap"]

    pos = {x: (embeddings[i, 0], embeddings[i, 1]) for i, x in enumerate(graph.nodes)}  # type: ignore

    return pos


def plot_read_graph(
    ax,
    query_graph: OverlapGraph | None = None,
    metadata: pd.DataFrame | None = None,
    reference_graph: OverlapGraph | None = None,
    *,
    layout_method="sfdp",
    pos=None,
    figsize=(8, 6),
    node_size=3,
    seed=43,
):
    if query_graph is not None:
        g = query_graph
    elif query_graph is None and reference_graph is not None:
        g = reference_graph
    else:
        raise TypeError()

    # Edge colors
    GRAY = (0.8, 0.8, 0.8, 1)
    LIGHT_GRAY = (0.9, 0.9, 0.9, 1)
    RED = (1, 0.1, 0.1, 1)

    edge_colors = []
    for edge, attr in g.edges.items(): 
        color = "k"
        k1, k2 = edge

        if attr.get("redundant", False):
            color = LIGHT_GRAY
        elif reference_graph is not None and not reference_graph.has_edge(k1, k2):
            color = RED
        else:
            color = GRAY
        edge_colors.append(color)

    BLUE = (29 / 255, 89 / 255, 142 / 255, 1)
    GREEN = (53 / 255, 125 / 255, 35 / 255, 1)
    ORANGE = (252 / 255, 177 / 255, 3 / 255, 1)

    # Node colors
    node_colors = []
    for node in g.nodes:
        if len(g[node]) == 0:
            color = ORANGE
        elif metadata is None:
            color = BLUE
        elif metadata.at[node, "reference_strand"] == "+":
            color = BLUE
        elif metadata.at[node, "reference_strand"] == "-":
            color = GREEN
        else:
            raise ValueError()
        node_colors.append(color)

    # Layout
    if layout_method == "umap":
        pos = get_umap_layout(graph=g)
    else:
        pos = get_graphviz_layout(
            graph=g, figsize=figsize, seed=seed, method=layout_method
        )

    # Plot
    nx.draw_networkx(
        g,
        ax=ax,
        pos=pos,
        with_labels=False,
        node_size=node_size,
        edge_color=edge_colors,
        node_color=node_colors,
    )
    ax.axis("off")


def mp_plot_read_graphs(
    axes,
    query_graphs: Sequence[OverlapGraph],
    reference_graph: OverlapGraph,
    metadata: pd.DataFrame,
    *,
    layout_method="neato",
    figsize=(8, 6),
    node_size=3,
    seed: int = 4829,
    processes: int = 8,
    output_dir: str | None = None,
    verbose=True,
):
    def plot(i, pos):
        plot_read_graph(
            ax=axes[i],
            query_graph=query_graphs[i],
            reference_graph=reference_graph,
            metadata=metadata,
            pos=pos,
            node_size=node_size,
        )

    with sharedmem.MapReduce(np=processes) as pool:

        def work(i):
            if layout_method == "umap":
                pos = get_umap_layout(graph=query_graphs[i])
            else:
                pos = get_graphviz_layout(
                    graph=query_graphs[i],
                    figsize=figsize,
                    seed=seed,
                    method=layout_method,
                )
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"{i}.png")
                plot(i, pos)
                axes[i].figure.savefig(output_path)
            return i, pos

        def reduce(i, pos):
            if verbose:
                print(i, end=" ")
            plot(i, pos)

        pool.map(work, range(len(query_graphs)), reduce=reduce)
        if verbose:
            print("")
