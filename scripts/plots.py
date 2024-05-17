import networkx as nx





def get_graphviz_layout(graph, prog="neato", figsize=(8, 6), seed=43):
    fig_width, fig_height = figsize
    pos = nx.nx_agraph.graphviz_layout(
        graph, prog="neato", args=f'-Gsize="{fig_width},{fig_height}" -Gstart={seed}'
    )
    return pos


def plot_read_graph(ax, read_graph, overlap_dict, *, pos=None, figsize=(8, 6), seed=43):
    g = read_graph

    BLUE = (29 / 255, 89 / 255, 142 / 255, 1)
    GRAY = (0.8, 0.8, 0.8, 1)
    RED = (1, 0.1, 0.1, 1)
    GREEN = (53 / 255, 125 / 255, 35 / 255, 1)

    edge_colors = []
    for edge, attr in g.edges.items():
        color = "k"
        k1, k2 = edge
        k1, k2 = get_fwd_id(k1), get_fwd_id(k2)
        if overlap_dict[k1][k2] <= 0:
            color = RED
        else:
            color = GRAY
        edge_colors.append(color)

    node_colors = [BLUE if x >= 0 else GREEN for x in g.nodes]

    if pos is None:
        pos = get_graphviz_layout(read_graph, figsize=figsize, seed=seed)

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
    overlap_dict: Mapping[tuple[int, int], int],
    *,
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

        layout = [None] * len(read_graphs)

        def work(i):
            pos = get_graphviz_layout(graph=read_graphs[i], figsize=figsize, seed=43)
            layout[i] = pos
            return i

        def reduce(i):
            print(i, end=" ")
            plot_read_graph(
                ax=axes[i],
                read_graph=read_graphs[i],
                overlap_dict=overlap_dict,
                pos=layout[i],
            )

        pool.map(work, range(len(read_graphs)), reduce=reduce)

    return figures, axes