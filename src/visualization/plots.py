import math
from os import PathLike
from typing import Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch


ArrayLike = Union[torch.Tensor, np.ndarray]


def _to_numpy(array: ArrayLike) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _degree_circle_layout(graph: nx.Graph) -> dict:
    degrees = dict(graph.degree())
    if not degrees:
        return nx.circular_layout(graph)

    nodes_sorted = sorted(degrees, key=degrees.get, reverse=True)
    n_nodes = len(nodes_sorted)
    angles = np.linspace(0.0, 2.0 * math.pi, n_nodes, endpoint=False)

    return {
        node: (math.cos(theta), math.sin(theta))
        for node, theta in zip(nodes_sorted, angles)
    }


def _compute_layout(graph: nx.Graph, dataset_name: Optional[str], seed: int) -> dict:
    if dataset_name and "pa" in dataset_name.lower():
        return _degree_circle_layout(graph)
    return nx.spring_layout(graph, seed=seed)


def _degree_colors(graph: nx.Graph) -> Optional[Sequence[float]]:
    degrees = dict(graph.degree())
    if not degrees:
        return None
    max_deg = max(degrees.values())
    if max_deg <= 0:
        return None
    return [degrees[node] / max_deg for node in graph.nodes()]


def plot_graph_comparison(
    adj_true: ArrayLike,
    adj_recon: ArrayLike,
    adj_noisy: ArrayLike,
    *,
    t_val: Optional[float] = None,
    cmap: str = "viridis",
) -> plt.Figure:
    adj_arrays = [_to_numpy(adj) for adj in (adj_true, adj_recon, adj_noisy)]

    adj_min = min(float(arr.min()) for arr in adj_arrays)
    adj_max = max(float(arr.max()) for arr in adj_arrays)

    fig, axs = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)

    title = f"Sampled t = {t_val:.2f}" if t_val is not None else "Graph Comparison"
    fig.suptitle(title, fontsize=14)

    subtitles = ("True Adjacency", "Reconstructed Adjacency", "Noisy Adjacency")
    images = []

    for ax, subtitle, data in zip(axs, subtitles, adj_arrays):
        im = ax.imshow(data, cmap=cmap, vmin=adj_min, vmax=adj_max)
        ax.set_title(subtitle)
        ax.axis("off")
        images.append(im)

    for ax, im in zip(axs, images):
        cbar = fig.colorbar(im, ax=ax, shrink=0.9, location="right", pad=0.01)
        cbar.set_label("Adjacency Value")

    return fig


def plot_graph_grid(
    graph_list: Sequence[nx.Graph],
    *,
    num_graphs: int = 100,
    n_cols: int = 10,
    n_rows: Optional[int] = None,
    title: Optional[str] = "Sampled Graphs",
    dataset_name: Optional[str] = None,
    layout_seed: int = 42,
    show_avg_degree: bool = True,
    base_node_size: int = 20,
    base_edge_width: float = 0.5,
) -> plt.Figure:
    if not graph_list:
        raise ValueError("graph_list is empty; nothing to plot.")

    num_graphs = max(1, min(num_graphs, len(graph_list)))
    n_cols = max(1, n_cols)
    if n_rows is None:
        n_rows = math.ceil(num_graphs / n_cols) if n_cols else 1

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3 * n_cols, 3 * n_rows),
    )
    axes = np.atleast_1d(axes).flatten()

    for idx, ax in enumerate(axes):
        ax.axis("off")
        if idx >= num_graphs:
            continue

        graph = graph_list[idx]
        pos = _compute_layout(graph, dataset_name, layout_seed)
        draw_kwargs = {
            "ax": ax,
            "pos": pos,
            "with_labels": False,
        }

        if dataset_name and "pa" in dataset_name.lower():
            colors = _degree_colors(graph)
            draw_kwargs.update(
                node_size=max(base_node_size, 40),
                width=0.4,
            )
            if colors is not None:
                draw_kwargs["node_color"] = colors
                draw_kwargs["cmap"] = "viridis"
        else:
            draw_kwargs.update(node_size=base_node_size, width=base_edge_width)

        nx.draw(graph, **draw_kwargs)

        if show_avg_degree:
            degree_dict = dict(graph.degree())
            n = max(1, len(degree_dict))
            avg_deg = sum(degree_dict.values()) / n
            ax.set_title(f"avg deg: {avg_deg:.2f}", fontsize=8)

    if title:
        fig.suptitle(title, fontsize=16)
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    else:
        fig.tight_layout()
    return fig


def plot_graph_snapshots(
    snapshots: Sequence[np.ndarray],
    *,
    grid_shape: Tuple[int, int] = (10, 10),
    threshold: float = 0.5,
    layout_seed: int = 42,
    node_size: int = 20,
    edge_width: float = 0.8,
) -> plt.Figure:
    if not snapshots:
        raise ValueError("snapshots sequence is empty.")

    n_rows, n_cols = grid_shape
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.8 * n_cols, 1.8 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    max_plots = n_rows * n_cols
    trimmed = list(snapshots)[:max_plots]

    last_graph = nx.from_numpy_array((trimmed[-1] > threshold).astype(int))
    if last_graph.number_of_nodes() == 0:
        pos = nx.spring_layout(nx.Graph(), seed=layout_seed)
    else:
        pos = nx.spring_layout(last_graph, seed=layout_seed)

    for ax, adj in zip(axes, trimmed):
        graph = nx.from_numpy_array((adj > threshold).astype(int))
        ax.axis("off")
        nx.draw_networkx(graph, pos=pos, with_labels=False, node_size=node_size, width=edge_width, ax=ax)

    for ax in axes[len(trimmed) :]:
        ax.axis("off")

    fig.tight_layout()
    return fig


def plot_heatmap_snapshots(
    snapshots: Sequence[np.ndarray],
    *,
    grid_shape: Tuple[int, int] = (10, 10),
    cmap: str = "Greys",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> plt.Figure:
    if not snapshots:
        raise ValueError("snapshots sequence is empty.")

    n_rows, n_cols = grid_shape
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.8 * n_cols, 1.8 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    max_plots = n_rows * n_cols
    trimmed = list(snapshots)[:max_plots]

    for ax, adj in zip(axes, trimmed):
        ax.imshow(adj, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.axis("off")

    for ax in axes[len(trimmed) :]:
        ax.axis("off")

    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, path: Union[str, PathLike[str]], *, dpi: int = 300, close: bool = True) -> None:
    fig.savefig(path, dpi=dpi)
    if close:
        plt.close(fig)


def close_figure(fig: plt.Figure) -> None:
    plt.close(fig)
