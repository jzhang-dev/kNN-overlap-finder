from typing import Sequence, Mapping
import networkx as nx
from data_io import is_fwd_id, get_fwd_id

import sharedmem
import scanpy as sc
import matplotlib.pyplot as plt
import anndata as ad

from graph import ReadGraph


def get_graphviz_layout(graph, prog="neato", figsize=(8, 6), seed=43):
    fig_width, fig_height = figsize
    pos = nx.nx_agraph.graphviz_layout(
        graph, prog="neato", args=f'-Gsize="{fig_width},{fig_height}" -Gstart={seed}'
    )
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
    ax, query_graph, reference_graph=None, *, pos=None, figsize=(8, 6), seed=43
):
    g = query_graph

    BLUE = (29 / 255, 89 / 255, 142 / 255, 1)
    GRAY = (0.8, 0.8, 0.8, 1)
    RED = (1, 0.1, 0.1, 1)
    GREEN = (53 / 255, 125 / 255, 35 / 255, 1)

    edge_colors = []
    for edge, attr in g.edges.items():
        color = "k"
        k1, k2 = edge
        k1, k2 = get_fwd_id(k1), get_fwd_id(k2)
        if reference_graph and not reference_graph.has_edge(k1, k2):
            color = RED
        else:
            color = GRAY
        edge_colors.append(color)

    node_colors = [BLUE if x >= 0 else GREEN for x in g.nodes]

    if pos is None:
        pos = get_graphviz_layout(query_graph, figsize=figsize, seed=seed)

    nx.draw_networkx(
        g,
        ax=ax,
        pos=pos,
        with_labels=False,
        node_size=6,
        edge_color=edge_colors,
        node_color=node_colors,
    )
    ax.axis("off")


def mp_plot_read_graphs(
    read_graphs: Sequence[ReadGraph],
    reference_graph: ReadGraph,
    *,
    layout_method="neato",
    figsize=(8, 6),
    processes: int = 8,
):

    with sharedmem.MapReduce(np=processes) as pool:
        figures = []
        axes = []
        for _ in read_graphs:
            fig, ax = plt.subplots(figsize=figsize)
            figures.append(fig)
            axes.append(ax)

        def work(i):
            if layout_method == "neato":
                pos = get_graphviz_layout(
                    graph=read_graphs[i], figsize=figsize, seed=43
                )
            elif layout_method == "umap":
                pos = get_umap_layout(graph=read_graphs[i])
            else:
                raise ValueError()

            return i, pos

        def reduce(i, pos):
            print(i, end=" ")
            plot_read_graph(
                ax=axes[i],
                query_graph=read_graphs[i],
                reference_graph=reference_graph,
                pos=pos,
            )

        pool.map(work, range(len(read_graphs)), reduce=reduce)

    return figures, axes
