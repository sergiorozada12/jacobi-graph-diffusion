import argparse
import math
from pathlib import Path

import networkx as nx
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from configs.config_metrofi import MainConfig
from src.dataset.wireless import WirelessDatasetModule
from src.models.transformer_model import GraphTransformer
from src.sample.sampler import Sampler
from src.dataset.utils import compute_reference_metrics
from src.metrics.abstract import compute_ratios
from src.metrics.val import WirelessSamplingMetrics
from src.utils import adjs_to_graphs
from src.visualization.plots import (
    plot_edge_weight_histograms,
    plot_heatmap_snapshots,
    plot_weighted_adj_and_graph,
    save_figure,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate MetroFi interference graphs.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Override checkpoint path. Defaults to checkpoints/<dataset>/weights.pth or weights_ema.pth.",
    )
    return parser.parse_args()


def load_model(cfg, checkpoint_path):
    model = GraphTransformer(
        n_layers=cfg.model.n_layers,
        input_dims=cfg.model.input_dims,
        hidden_mlp_dims=cfg.model.hidden_mlp_dims,
        hidden_dims=cfg.model.hidden_dims,
        output_dims=cfg.model.output_dims,
        act_fn_in=torch.nn.ReLU(),
        act_fn_out=torch.nn.ReLU(),
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(cfg.general.device)
    model.eval()
    return model


def _collect_snapshots(graphs, max_node_num, max_snapshots, weight_key):
    snapshots = []
    max_value = 0.0

    for graph in graphs:
        if graph.number_of_nodes() == 0:
            continue

        nodes = list(graph.nodes())
        adj = nx.to_numpy_array(graph, nodelist=nodes, weight=weight_key, dtype=float)
        padded = np.zeros((max_node_num, max_node_num), dtype=np.float32)
        size = min(adj.shape[0], max_node_num)
        if size > 0:
            sub_adj = adj[:size, :size]
            padded[:size, :size] = sub_adj
            max_value = max(max_value, float(np.max(sub_adj)))
        snapshots.append(padded)

        if len(snapshots) >= max_snapshots:
            break

    return snapshots, max_value


def build_weighted_snapshots(graphs, max_node_num, max_snapshots=25):
    """Convert weighted graphs to padded adjacency heatmaps."""
    snapshots, max_value = _collect_snapshots(graphs, max_node_num, max_snapshots, "weight")
    weight_attr = "weight"
    if max_value <= 0.0:
        fallback_snapshots, fallback_max = _collect_snapshots(
            graphs,
            max_node_num,
            max_snapshots,
            "weight_norm",
        )
        if fallback_max > 0.0:
            snapshots, max_value = fallback_snapshots, fallback_max
            weight_attr = "weight_norm"
    return snapshots, max_value, weight_attr


def main():
    args = parse_args()

    cfg = OmegaConf.structured(MainConfig())
    cfg.train.training_mode = "weighted"
    cfg.model.output_dims = dict(cfg.model.score_output_dims)

    if torch.cuda.is_available():
        cfg.general.device = "cuda:0"
    else:
        if str(cfg.general.device).startswith("cuda"):
            print("CUDA not available, falling back to CPU.")
        cfg.general.device = "cpu"

    _ = pl.seed_everything(cfg.general.seed, workers=True)

    datamodule = WirelessDatasetModule(cfg)
    datamodule.setup()

    num_fixed_aps = datamodule.num_mac_addresses()
    if num_fixed_aps > 0:
        cfg.data.max_node_num = num_fixed_aps
        cfg.sampler.num_nodes = num_fixed_aps
        datamodule.max_node_num = num_fixed_aps
        print(f"Using fixed AP count from dataset metadata: {num_fixed_aps} nodes.")
    else:
        cfg.sampler.num_nodes = cfg.data.max_node_num
        print(f"Using configured max_node_num: {cfg.data.max_node_num} nodes.")

    # Prefer post-filter interference range when available so we rescale to the dataset bounds.
    if getattr(datamodule, "interference_range", None):
        interference_min, interference_max = datamodule.interference_range
    else:
        interference_min = getattr(datamodule, "interference_min", 0.0)
        interference_max = getattr(datamodule, "interference_max", 0.0)
    interference_min = float(0.0 if interference_min is None else interference_min)
    interference_max = float(interference_min if interference_max is None else interference_max)
    interference_scale = max(0.0, interference_max - interference_min)

    ckpt_dir = Path("checkpoints") / cfg.data.data
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        ema_path = ckpt_dir / "weights_ema.pth"
        weights_path = ckpt_dir / "weights.pth"
        if ema_path.exists():
            checkpoint_path = ema_path
        elif weights_path.exists():
            checkpoint_path = weights_path
        else:
            raise FileNotFoundError(
                f"Could not find checkpoint for '{cfg.data.data}'. "
                f"Looked for {ema_path} and {weights_path}. "
                "Pass --checkpoint to specify a custom file."
            )

    model = load_model(cfg, checkpoint_path)

    sampler = Sampler(cfg=cfg, model=model, node_dist=None)
    samples, _, adj_samples = sampler.sample(
        keep_isolates=True,
        return_adjs=True,
        use_node_dist=False,
        nodelist=list(range(cfg.data.max_node_num)),
        keep_zero_weights=True,
    )

    # Rescale dense adjacencies before graph conversion so zeros map to interference_min.
    adj_rescaled = adj_samples * interference_scale + interference_min
    adj_rescaled = adj_rescaled.numpy()

    graph_list = adjs_to_graphs(
        adj_rescaled,
        is_cuda=False,
        keep_isolates=True,
        keep_zero_weights=True,
        nodelist=list(range(adj_rescaled.shape[-1])),
    )

    # Attach normalized weights for bookkeeping.
    max_weight_norm = 0.0
    max_weight_raw = -float("inf")
    total_edges = 0
    for idx, G in enumerate(graph_list):
        norm_adj = adj_samples[idx].numpy()
        for u, v, data in G.edges(data=True):
            weight_norm = float(norm_adj[u, v])
            data["weight_norm"] = weight_norm
            data["interference"] = data["weight"]
            data["interference_raw"] = data["weight"]
            max_weight_norm = max(max_weight_norm, weight_norm)
            max_weight_raw = max(max_weight_raw, float(data["weight"]))
            total_edges += 1

    print(
        f"Rescaled {total_edges} edges to dataset range "
        f"[{interference_min:.4f}, {interference_max:.4f}] "
        f"(max weight_norm={max_weight_norm:.4f}, max weight={max_weight_raw:.4f})."
    )

    save_path = Path("samples") / "wireless.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if len(graph_list) > 0:
        first_graph_obj = graph_list[0]
        first_graph_adj = adj_rescaled[0]
        fig = plot_weighted_adj_and_graph(
            first_graph_adj,
            graph=first_graph_obj,
            dataset_name=cfg.data.data,
            cmap="coolwarm",
            vmin=interference_min,
            vmax=interference_max if interference_scale > 0 else None,
        )
        save_figure(fig, save_path, dpi=300)
        print(
            f"Saved figure to {save_path} using rescaled weights "
            f"(interference range: [{interference_min:.4f}, {interference_max:.4f}])."
        )
    else:
        print("No graphs generated; skipping figure save.")

    # Compute and print sampling metrics for the generated graphs.
    try:
        sampling_metrics = WirelessSamplingMetrics(datamodule)
        ref_metrics = compute_reference_metrics(datamodule, sampling_metrics)
        metrics_out = sampling_metrics.forward(
            graph_list,
            ref_metrics=ref_metrics,
            local_rank=0,
            test=True,
        )
        ref_test = ref_metrics.get("test") if isinstance(ref_metrics, dict) else None
        ratio_keys = list(metrics_out.keys())
        if "edge_pairs_used" in ratio_keys:
            ratio_keys.remove("edge_pairs_used")
        ratios = compute_ratios(metrics_out, ref_test or {}, ratio_keys)

        print("Wireless sampling metrics:", metrics_out)
        if ratios:
            print("Ratios vs reference:", ratios)

        # Save edge weight histogram comparison (test vs generated).
        try:
            ref_graphs_raw = []
            for g in datamodule.test_graphs:
                g_copy = g.copy()
                for u, v, data in g_copy.edges(data=True):
                    raw = data.get("interference_raw", data.get("weight", 0.0))
                    data["weight"] = float(raw)
                    data["interference"] = float(raw)
                ref_graphs_raw.append(g_copy)

            hist_fig = plot_edge_weight_histograms(
                ref_graphs_raw,
                graph_list,
                dataset_name="metrofi test vs generated edge weights",
            )
            hist_path = save_path.with_name("wireless_edge_weight_hist.png")
            save_figure(hist_fig, hist_path, dpi=300)
            print(f"Saved edge weight histogram comparison to {hist_path}.")
        except Exception as exc:
            print(f"Warning: could not plot edge weight histograms: {exc}")
    except Exception as exc:
        print(f"Warning: could not compute wireless sampling metrics: {exc}")

    heatmap_snapshots, vmax, weight_attr = build_weighted_snapshots(samples, cfg.data.max_node_num)
    if heatmap_snapshots:
        grid_cols = min(5, len(heatmap_snapshots))
        grid_rows = math.ceil(len(heatmap_snapshots) / grid_cols)
        vmin_heat = interference_min if interference_scale > 0 else 0.0
        vmax = max(vmax, interference_max) if interference_scale > 0 else (vmax if vmax > 0 else 1.0)
        heatmap_fig = plot_heatmap_snapshots(
            heatmap_snapshots,
            grid_shape=(grid_rows, grid_cols),
            cmap="magma",
            vmin=vmin_heat,
            vmax=vmax,
        )
        heatmap_path = save_path.with_name("wireless_weight_heatmaps.png")
        save_figure(heatmap_fig, heatmap_path, dpi=300)
        print(
            f"Saved weighted adjacency heatmaps to {heatmap_path} "
            f"(vmax: {vmax:.4f}, attr={weight_attr})."
        )


if __name__ == "__main__":
    main()
