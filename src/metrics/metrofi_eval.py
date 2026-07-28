from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from src.metrics.val import masked_adjacency_weight_metrics
from src.visualization.plots import (
    plot_pooled_edge_weight_histogram,
    save_conditional_adjacency_analysis,
    save_figure,
)


MASKED_VARIANT = "mask_during_generation"
FULL_VARIANT = "full_generation_masked_eval"


def save_metrofi_pooled_weights_json(
    results: Dict[str, Dict[str, Any]],
    out_path,
    *,
    interference_min: float,
    interference_max: float,
) -> Path:
    """Save masked pooled generated edge weights, converted from [0, 1] to dBm."""
    scale = float(interference_max) - float(interference_min)

    def _weights_dbm(variant: str):
        values = np.asarray(results[variant]["gen_edge_values"], dtype=np.float64)
        return (values * scale + float(interference_min)).tolist()

    payload = {
        "weights_conditional": _weights_dbm(MASKED_VARIANT),
        "weights_full": _weights_dbm(FULL_VARIANT),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f)
    return out_path


def flatten_masked_eval_metrics(results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    flat = {}
    for variant, payload in results.items():
        for key, value in payload["metrics"].items():
            flat[f"{variant}_{key}"] = value
    return flat


def select_metrofi_eval_tensors(dataset, n_graphs: int, device: str, seed: int):
    if not hasattr(dataset, "tensors") or len(dataset.tensors) < 3:
        raise ValueError("Expected MetroFi TensorDataset with X, adjacency, and observed_mask tensors.")

    n_metric = min(int(n_graphs), len(dataset))
    if n_metric <= 0:
        raise ValueError("No graphs selected for MetroFi masked evaluation.")

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=n_metric, replace=False)
    tensors = dataset.tensors
    gt_adj = tensors[1][indices].to(device).float()
    observed_mask = tensors[2][indices].to(device).bool()
    coords = tensors[3][indices].to(device).float() if len(tensors) >= 4 else None
    return indices, gt_adj, observed_mask, coords, rng


def apply_condition_mode(coords: Optional[torch.Tensor], rng, mode: str) -> Optional[torch.Tensor]:
    if coords is None:
        return None

    mode = str(mode).lower()
    if mode == "zero":
        return torch.zeros_like(coords)
    if mode == "shuffled":
        n_metric = coords.size(0)
        perm_np = rng.permutation(n_metric)
        if n_metric > 1 and np.array_equal(perm_np, np.arange(n_metric)):
            perm_np = np.roll(perm_np, 1)
        perm = torch.as_tensor(perm_np, device=coords.device, dtype=torch.long)
        return coords[perm]
    if mode != "true":
        raise ValueError(f"Unknown conditional eval condition mode: {mode}")
    return coords


def _sample_adj_variant(
    cfg,
    sampler,
    n_graphs: int,
    *,
    condition: Optional[torch.Tensor],
    observed_mask: Optional[torch.Tensor],
):
    old_test_graphs = cfg.sampler.test_graphs
    cfg.sampler.test_graphs = n_graphs
    try:
        graphs, fig, adj_samples = sampler.sample(
            keep_isolates=True,
            return_adjs=True,
            use_node_dist=False,
            nodelist=list(range(cfg.data.max_node_num)),
            keep_zero_weights=True,
            condition=condition,
            fixed_flags=observed_mask,
        )
    finally:
        cfg.sampler.test_graphs = old_test_graphs
    return graphs, fig, adj_samples[:n_graphs].float()


def run_metrofi_masked_eval_variants(
    cfg,
    sampler,
    gt_adj: torch.Tensor,
    observed_mask: torch.Tensor,
    *,
    condition: Optional[torch.Tensor] = None,
    include_masked_variant: bool = True,
    include_full_variant: bool = True,
) -> Dict[str, Dict[str, Any]]:
    n_graphs = gt_adj.size(0)
    results = {}

    if include_masked_variant:
        graphs, fig, gen_adj = _sample_adj_variant(
            cfg,
            sampler,
            n_graphs,
            condition=condition,
            observed_mask=observed_mask,
        )
        metrics, gt_edge_values, gen_edge_values = masked_adjacency_weight_metrics(
            gt_adj,
            gen_adj.to(gt_adj.device),
            observed_mask,
        )
        results[MASKED_VARIANT] = {
            "metrics": metrics,
            "gt_edge_values": gt_edge_values,
            "gen_edge_values": gen_edge_values,
            "gen_adj": gen_adj,
            "graphs": graphs,
            "fig": fig,
        }

    if include_full_variant:
        graphs, fig, gen_adj = _sample_adj_variant(
            cfg,
            sampler,
            n_graphs,
            condition=condition,
            observed_mask=None,
        )
        metrics, gt_edge_values, gen_edge_values = masked_adjacency_weight_metrics(
            gt_adj,
            gen_adj.to(gt_adj.device),
            observed_mask,
        )
        results[FULL_VARIANT] = {
            "metrics": metrics,
            "gt_edge_values": gt_edge_values,
            "gen_edge_values": gen_edge_values,
            "gen_adj": gen_adj,
            "graphs": graphs,
            "fig": fig,
        }

    return results


def build_metrofi_masked_eval_figures(
    results: Dict[str, Dict[str, Any]],
    *,
    title_prefix: str,
) -> Dict[str, Dict[str, Any]]:
    figures = {}
    for variant, payload in results.items():
        figures[variant] = {
            "hist": plot_pooled_edge_weight_histogram(
                payload["gt_edge_values"],
                payload["gen_edge_values"],
                dataset_name=f"{title_prefix} {variant} pooled edge weights",
            ),
            "sampled_graph": payload.get("fig"),
        }
    return figures


def save_metrofi_masked_eval_artifacts(
    results: Dict[str, Dict[str, Any]],
    gt_adj: torch.Tensor,
    observed_mask: torch.Tensor,
    out_dir,
    *,
    matrix_size: int,
    title_prefix: str,
    dpi: int = 150,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for variant, payload in results.items():
        variant_dir = out_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)

        gen_adj = payload["gen_adj"].detach().cpu()
        save_conditional_adjacency_analysis(
            gt_adj.detach().cpu(),
            gen_adj,
            observed_mask.detach().cpu(),
            variant_dir,
            matrix_size=matrix_size,
            dpi=dpi,
        )

        hist_fig = plot_pooled_edge_weight_histogram(
            payload["gt_edge_values"],
            payload["gen_edge_values"],
            dataset_name=f"{title_prefix} {variant} pooled edge weights",
        )
        save_figure(hist_fig, variant_dir / "pooled_edge_weight_hist.png", dpi=dpi)

        fig = payload.get("fig")
        if fig is not None:
            save_figure(fig, variant_dir / "sampled_graph.png", dpi=dpi)


def timestamped_eval_dir(base_dir, run_name: str, seed: int, n_graphs: int) -> Path:
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return Path(base_dir) / f"{run_name}_seed{seed}_n{n_graphs}_{run_tag}"
