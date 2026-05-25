import argparse
import math
import statistics
import json
import torch
import pytorch_lightning as pl
from pathlib import Path
from omegaconf import OmegaConf
import numpy as np


def print_header(title):
    print("\n" + "=" * 80)
    print(f" {title.upper()} ".center(80, "="))
    print("=" * 80)

def print_subheader(title):
    print(f"\n--- {title} ---")

def log_info(msg):
    print(f"🔹 [INFO] {msg}")

def log_success(msg):
    print(f"🟢 [SUCCESS] {msg}")

def log_warning(msg):
    print(f"⚠️ [WARNING] {msg}")

def log_error(msg):
    print(f"🔴 [ERROR] {msg}")

def print_row(label, val, ref=None):
    if ref is not None:
        if isinstance(val, (int, float)) and isinstance(ref, (int, float)):
            print(f"  ▸ {label:<35} | Gen: {val:<12.6f} | Ref: {ref:<12.6f}")
        else:
            print(f"  ▸ {label:<35} | Gen: {val:<12} | Ref: {ref:<12}")
    else:
        if isinstance(val, float):
            print(f"  ▸ {label:<35} | {val:<12.6f}")
        else:
            print(f"  ▸ {label:<35} | {val}")

# Core components and utilities loader
def load_model_components(model_name):
    if model_name == "pa":
        from configs.config_pa import MainConfig
        from src.metrics.val import PASamplingMetrics as MetricsClass
        from src.dataset.spectre import SpectreDatasetModule as DatasetClass
        wandb_name = "pa-graphon"
    elif model_name == "sbm":
        from configs.config_sbm import MainConfig
        from src.metrics.val import SBMSamplingMetrics as MetricsClass
        from src.dataset.spectre import SpectreDatasetModule as DatasetClass
        wandb_name = "sbm-graphon"
    elif model_name == "sbm_2comms":
        from configs.config_sbm_2comms import MainConfig
        from src.metrics.val import SBMSamplingMetrics as MetricsClass
        from src.dataset.spectre import SpectreDatasetModule as DatasetClass
        wandb_name = "sbm-2comms-graphon"
    elif model_name == "tree":
        from configs.config_tree import MainConfig
        from src.metrics.val import TreeSamplingMetrics as MetricsClass
        from src.dataset.spectre import SpectreDatasetModule as DatasetClass
        wandb_name = "tree"
    elif model_name == "tree_graphon":
        from configs.config_tree_graphon import MainConfig
        from src.metrics.val import TreeSamplingMetrics as MetricsClass
        from src.dataset.spectre import SpectreDatasetModule as DatasetClass
        wandb_name = "tree-graphon"
    elif model_name == "planar":
        from configs.config_planar import MainConfig
        from src.metrics.val import PlanarSamplingMetrics as MetricsClass
        from src.dataset.spectre import SpectreDatasetModule as DatasetClass
        wandb_name = "planar"
    elif model_name == "metrofi":
        from configs.config_metrofi import MainConfig
        from src.metrics.val import WirelessSamplingMetrics as MetricsClass
        from src.dataset.wireless import WirelessDatasetModule as DatasetClass
        wandb_name = "metrofi"
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return MainConfig, MetricsClass, DatasetClass, wandb_name

# Helper functions for generation and evaluation
def _validate_expected_num_graphs(samples, expected_num_graphs):
    if expected_num_graphs is None:
        return samples
    actual_num_graphs = len(samples)
    if actual_num_graphs < expected_num_graphs:
        raise ValueError(
            f"Expected {expected_num_graphs} graphs, but found {actual_num_graphs}."
        )
    if actual_num_graphs > expected_num_graphs:
        log_info(
            f"Found {actual_num_graphs} graphs, truncating to the first "
            f"{expected_num_graphs} to match --expected-num-graphs."
        )
        return samples[:expected_num_graphs]
    return samples

def _print_ratio_block(metrics, suffix="", title="Ratios"):
    print_header(title)
    keys = sorted([k for k in metrics if k.endswith('_ratio' + suffix) or k.endswith('average_ratio' + suffix)])
    if not keys:
        log_info("No ratios available in this block.")
        return
    for k in keys:
        print_row(k, metrics[k])

def _aggregate_fold_metrics(fold_metrics):
    aggregated = {}
    if not fold_metrics:
        return aggregated
    keys = fold_metrics[0].keys()
    for key in keys:
        values = [metrics[key] for metrics in fold_metrics if isinstance(metrics.get(key), (int, float))]
        if not values:
            continue
        aggregated[key] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    return aggregated

def _resolve_size_summary(samples, args, cfg):
    if args.min_nodes is not None and args.max_nodes is not None and args.min_nodes == args.max_nodes:
        return args.min_nodes
    if not samples:
        return None

    node_counts = sorted({graph.number_of_nodes() for graph in samples})
    if len(node_counts) == 1:
        return node_counts[0]
    return [node_counts[0], node_counts[-1]]

def _evaluate_folds(samples, n_folds, sampling_metrics, ref_metrics, extra_sampling_metrics):
    fold_size = math.ceil(len(samples) / n_folds)
    fold_metrics = []
    fold_metrics_extra = {}
    for fold_idx in range(n_folds):
        start = fold_idx * fold_size
        end = min(len(samples), start + fold_size)
        fold_samples = samples[start:end]
        if not fold_samples:
            continue
        sampling_metrics.reset()
        fold_result = sampling_metrics.forward(
            fold_samples,
            ref_metrics=ref_metrics,
            local_rank=0,
            test=True,
        )
        fold_metrics.append(fold_result)
        if extra_sampling_metrics:
            for suffix, (extra_metrics, extra_module) in extra_sampling_metrics.items():
                extra_module.reset()
                res = extra_module.forward(
                    fold_samples,
                    ref_metrics=extra_metrics,
                    local_rank=0,
                    test=True,
                )
                fold_metrics_extra.setdefault(suffix, []).append(res)
    return fold_metrics, fold_metrics_extra

# Sub-execution routines
def run_train(args, cfg, wandb_name, MetricsClass, DatasetClass):
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import WandbLogger
    from src.dataset.utils import DistributionNodes, compute_reference_metrics
    from src.train.trainer_graph import DiffusionGraphModule, DiffusionWeightedGraphModule
    from src.train.trainer_score import DiffusionScoreModule

    train_mode = getattr(cfg.train, "training_mode", "graph")
    if train_mode == "direct_score":
        cfg.model.output_dims = dict(cfg.model.score_output_dims)
    else:
        cfg.model.output_dims = dict(cfg.model.output_dims)

    datamodule = DatasetClass(cfg)
    datamodule.setup()

    node_dist = DistributionNodes(prob=datamodule.node_counts())

    sampling_metrics = MetricsClass(datamodule)
    ref_metrics = compute_reference_metrics(datamodule, sampling_metrics)

    if train_mode == "direct_score":
        module_cls = DiffusionScoreModule
    elif train_mode == "weighted":
        module_cls = DiffusionWeightedGraphModule
    else:
        module_cls = DiffusionGraphModule

    model = module_cls(
        cfg=cfg,
        sampling_metrics=sampling_metrics,
        ref_metrics=ref_metrics,
        node_dist=node_dist,
    )

    ckpt_dir = Path(f"checkpoints/{cfg.data.data}")
    ckpt_path = ckpt_dir / "last.ckpt"

    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="{epoch}",
        save_top_k=-1,
        every_n_epochs=cfg.general.save_checkpoint_every_n_epochs,
        save_last=True,
    )
    callbacks.append(checkpoint_callback)

    wandb_run_id = None
    logger = WandbLogger(
        project="jacobi-graph-diffusion",
        name=wandb_name,
        id=wandb_run_id,
        resume="must" if wandb_run_id else None,
    )

    trainer = Trainer(
        accelerator="auto" if cfg.general.device.startswith("cuda") else "cpu",
        devices="auto",
        strategy="ddp_find_unused_parameters_true",
        max_epochs=cfg.train.num_epochs,
        check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
        log_every_n_steps=1,
        callbacks=callbacks,
        logger=logger,
    )

    if ckpt_path.exists():
        trainer.fit(model, datamodule=datamodule, ckpt_path=str(ckpt_path))
    else:
        trainer.fit(model, datamodule=datamodule)

def run_gen(args, cfg, MetricsClass, DatasetClass, model_name):
    if model_name == "metrofi":
        run_gen_wireless(args, cfg, MetricsClass, DatasetClass)
    else:
        run_gen_spectre(args, cfg, MetricsClass, DatasetClass, model_name)

def run_gen_spectre(args, cfg, MetricsClass, DatasetClass, model_name):
    from src.models.transformer_model import GraphTransformer
    from src.sample.sampler import Sampler
    from src.dataset.utils import DistributionNodes, compute_reference_metrics, load_graphs_pickle, save_graphs_pickle, load_size_ref_metrics
    from src.visualization.plots import save_figure

    train_mode = getattr(cfg.train, "training_mode", "graph")
    if train_mode == "direct_score":
        cfg.model.output_dims = dict(cfg.model.score_output_dims)

    if args.min_nodes is not None or args.max_nodes is not None:
        if args.min_nodes is None or args.max_nodes is None:
            raise ValueError("Both --min-nodes and --max-nodes must be specified.")
        if args.max_nodes > cfg.data.max_node_num:
            cfg.data.max_node_num = args.max_nodes
        if args.max_nodes > cfg.sampler.num_nodes:
            cfg.sampler.num_nodes = args.max_nodes

    datamodule = DatasetClass(cfg)
    datamodule.setup()

    max_nodes_possible = cfg.data.max_node_num + 1
    base_prob = datamodule.node_counts(max_nodes_possible)
    if base_prob.numel() < max_nodes_possible:
        padding = torch.zeros(max_nodes_possible - base_prob.numel(), dtype=base_prob.dtype)
        base_prob = torch.cat([base_prob, padding], dim=0)

    if args.min_nodes is None and args.max_nodes is None:
        node_dist = DistributionNodes(prob=base_prob)
    else:
        if args.min_nodes < 1:
            raise ValueError("Minimum number of nodes must be at least 1.")
        if args.max_nodes < args.min_nodes:
            raise ValueError("Maximum number of nodes must be greater than or equal to the minimum.")
        
        probs = torch.zeros_like(base_prob)
        probs[args.min_nodes : args.max_nodes + 1] = 1.0
        total_mass = probs.sum()
        if not torch.isfinite(total_mass) or total_mass <= 0:
            raise ValueError("Probability mass over the requested range is zero; check the node range.")
        probs /= total_mass
        node_dist = DistributionNodes(prob=probs)

    sampling_metrics = MetricsClass(datamodule)
    ref_metrics = compute_reference_metrics(datamodule, sampling_metrics)

    if args.load_graphs_path is not None:
        samples = load_graphs_pickle(args.load_graphs_path)
        log_success(f"Loaded saved graphs from {args.load_graphs_path}")
        samples = _validate_expected_num_graphs(samples, args.expected_num_graphs)
    else:
        model = GraphTransformer(
            n_layers=cfg.model.n_layers,
            input_dims=cfg.model.input_dims,
            hidden_mlp_dims=cfg.model.hidden_mlp_dims,
            hidden_dims=cfg.model.hidden_dims,
            output_dims=cfg.model.output_dims,
            act_fn_in=torch.nn.ReLU(),
            act_fn_out=torch.nn.ReLU(),
        )
        
        if args.checkpoint is not None:
            weight_path = Path(args.checkpoint)
            if not weight_path.exists():
                raise FileNotFoundError(f"Provided checkpoint does not exist: {weight_path}")
        else:
            ckpt_dir = Path("checkpoints") / cfg.data.data
            ema_path = ckpt_dir / "weights_ema.pth"
            weights_path = ckpt_dir / "weights.pth"
            if (not args.no_ema) and cfg.train.use_ema and ema_path.exists():
                weight_path = ema_path
            elif weights_path.exists():
                weight_path = weights_path
            else:
                raise FileNotFoundError(
                    f"Could not find checkpoint for '{cfg.data.data}'. "
                    f"Looked for {ema_path} and {weights_path}."
                )

        state_dict = torch.load(weight_path, map_location="cpu")
        
        # Load state dict helper for Lightning checkpoints
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    new_state_dict[k.replace('model.', '', 1)] = v
                elif k.startswith('ema_model.'):
                    new_state_dict[k.replace('ema_model.', '', 1)] = v
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict
        elif isinstance(state_dict, dict) and "state_dict" not in state_dict:
            pl_state = state_dict
            model_state = {k[len("model."):]: v for k, v in pl_state.items() if k.startswith("model.")}
            ema_state = {k[len("ema_model."):]: v for k, v in pl_state.items() if k.startswith("ema_model.")}
            if (not args.no_ema) and cfg.train.use_ema and ema_state:
                state_dict = ema_state
            elif model_state:
                state_dict = model_state

        model.load_state_dict(state_dict)
        model = model.to(cfg.general.device)
        model.eval()

        sampler = Sampler(cfg=cfg, model=model, node_dist=node_dist)
        samples, fig = sampler.sample()

        plot_path = f"samples/test_{model_name}.png"
        Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, Path(plot_path), dpi=300)
        log_success(f"Saved generated graph plot to {plot_path}")
        
        save_path = args.save_graphs_path if args.save_graphs_path else f"samples/test_{model_name}_graphs.pkl"
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        save_graphs_pickle(samples, save_path)
        log_success(f"Saved generated graphs to {save_path}")
        samples = _validate_expected_num_graphs(samples, args.expected_num_graphs)

    extra_sampling_metrics = None
    if not args.no_average_ratio_to_size_ref:
        if args.min_nodes is not None and args.max_nodes is not None and args.min_nodes == args.max_nodes:
            try:
                size_ref_metrics, size_ref_sampling_metrics = load_size_ref_metrics(
                    cfg=cfg,
                    metrics_cls=MetricsClass,
                    target_nodes=args.min_nodes,
                )
                extra_sampling_metrics = {
                    "size_ref": (size_ref_metrics, size_ref_sampling_metrics)
                }
            except Exception as e:
                log_error(f"Failed to setup node-specific metrics: {e}")
                extra_sampling_metrics = None

    if args.n_folds < 1:
        raise ValueError("--n-folds must be at least 1.")

    fold_metrics, fold_metrics_extra = _evaluate_folds(
        samples, args.n_folds, sampling_metrics, ref_metrics, extra_sampling_metrics
    )

    if args.n_folds > 1:
        print_header(f"Per-fold metrics ({args.n_folds} folds)")
        for idx, fold_result in enumerate(fold_metrics, start=1):
            print_subheader(f"Fold {idx}")
            for key in sorted(fold_result.keys()):
                if isinstance(fold_result[key], (int, float)):
                    print_row(key, fold_result[key])
        metrics = _aggregate_fold_metrics(fold_metrics)
        metrics_extra = {
            suffix: _aggregate_fold_metrics(results)
            for suffix, results in fold_metrics_extra.items()
        }
    else:
        metrics = fold_metrics[0]
        metrics_extra = {
            suffix: results[0]
            for suffix, results in fold_metrics_extra.items()
            if results
        }

    print_header("Evaluation Metrics vs Training Distribution")
    if args.n_folds > 1:
        for k in sorted(ref_metrics['val'].keys()):
            if k in metrics:
                mean_val = metrics[k]['mean']
                std_val = metrics[k]['std']
                print(f"  ▸ {k:<35} | Gen: {mean_val:.6f} ± {std_val:.6f} | Ref: {ref_metrics['val'][k]:.6f}")
    else:
        for k in sorted(ref_metrics['val'].keys()):
            if k in metrics:
                print_row(k, metrics[k], ref_metrics['val'][k])

    if args.n_folds == 1:
        _print_ratio_block(metrics, title="Training-distribution ratios")

    if extra_sampling_metrics is not None and "size_ref" in metrics_extra and args.n_folds == 1:
        size_ref_res = metrics_extra["size_ref"]
        ref_vals = extra_sampling_metrics["size_ref"][0]
        print_header("Size-matched reference comparison")
        for k in sorted(ref_vals['val'].keys()):
            if k in size_ref_res:
                print_row(k, size_ref_res[k], ref_vals['val'][k])
        _print_ratio_block(size_ref_res, title="Size-matched reference ratios")

    if args.json_out:
        base_keys = ["degree", "clustering", "orbit", "spectre", "wavelet"]
        if args.n_folds > 1:
            entry = {
                "size": _resolve_size_summary(samples, args, cfg),
                "n_folds": args.n_folds,
                "metrics_mean_std": metrics,
            }
            if "size_ref" in metrics_extra:
                entry["metrics_mean_std_size_ref"] = metrics_extra["size_ref"]
            out_path = Path(args.json_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(entry, f, indent=2)
            return

        def get_metrics_dict(m):
            d = {k: m.get(k, 0.0) for k in base_keys}
            avg = sum(d.values()) / len(base_keys) if base_keys else 0.0
            d["average_mmd"] = avg
            return d, avg

        def get_ratios_dict(m):
            d = {k + "_ratio": m.get(k + "_ratio", 0.0) for k in base_keys}
            d["average_ratio"] = m.get("average_ratio", 0.0)
            return d

        m_dict, avg_mmd = get_metrics_dict(metrics)
        m_ratios = get_ratios_dict(metrics)

        # Accuracies mapping
        accuracy_key = "accuracy"
        if "pa_acc" in metrics:
            accuracy_key = "pa_acc"
        elif "sbm_acc" in metrics:
            accuracy_key = "sbm_acc"
        elif "tree_acc" in metrics:
            accuracy_key = "tree_acc"

        entry = {
            "size": _resolve_size_summary(samples, args, cfg),
            "accuracy": metrics.get(accuracy_key, 0.0),
            "acc_extra": {
                "forest_acc": metrics.get("forest_acc", 0.0),
                "connected_acc": metrics.get("connected_acc", 0.0),
            } if "forest_acc" in metrics else {},
            "average_mmd": avg_mmd,
            "metrics": m_dict,
            "metrics_ratio": m_ratios,
        }

        if "size_ref" in metrics_extra:
            ms_metrics = metrics_extra["size_ref"]
            ms_dict, msa_mmd = get_metrics_dict(ms_metrics)
            ms_ratios = get_ratios_dict(ms_metrics)
            entry["size_specific_metrics"] = ms_dict
            entry["size_specific_metrics_ratio"] = ms_ratios

        out_path = Path(args.json_out)
        data = {"dataset_name": cfg.data.data, "kernel": args.kernel, "entries": []}
        if out_path.exists():
            with open(out_path, "r") as f:
                try:
                    data = json.load(f)
                except Exception as e:
                    log_warning(f"Warning: could not load existing JSON at {out_path}: {e}")

        updated = False
        for i, e in enumerate(data.get("entries", [])):
            if e["size"] == entry["size"]:
                data["entries"][i] = entry
                updated = True
                break
        if not updated:
            if "entries" not in data:
                data["entries"] = []
            data["entries"].append(entry)

        data["entries"].sort(key=lambda x: x["size"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(data, f, indent=4)
        log_success(f"Results saved to {out_path}")

def run_gen_wireless(args, cfg, MetricsClass, DatasetClass):
    from src.models.transformer_model import GraphTransformer
    from src.sample.sampler import Sampler
    from src.dataset.utils import compute_reference_metrics
    from src.metrics.abstract import compute_ratios
    from src.utils import adjs_to_graphs
    from src.visualization.plots import (
        plot_edge_weight_histograms,
        plot_heatmap_snapshots,
        plot_weighted_adj_and_graph,
        save_figure,
    )

    cfg.train.training_mode = "weighted"
    cfg.model.output_dims = dict(cfg.model.score_output_dims)

    datamodule = DatasetClass(cfg)
    datamodule.setup()

    num_fixed_aps = datamodule.num_mac_addresses()
    if num_fixed_aps > 0:
        cfg.data.max_node_num = num_fixed_aps
        cfg.sampler.num_nodes = num_fixed_aps
        datamodule.max_node_num = num_fixed_aps
        log_info(f"Using fixed AP count from dataset metadata: {num_fixed_aps} nodes.")
    else:
        cfg.sampler.num_nodes = cfg.data.max_node_num
        log_info(f"Using configured max_node_num: {cfg.data.max_node_num} nodes.")

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
                f"Could not find checkpoint for '{cfg.data.data}'."
            )

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

    sampler = Sampler(cfg=cfg, model=model, node_dist=None)
    samples, _, adj_samples = sampler.sample(
        keep_isolates=True,
        return_adjs=True,
        use_node_dist=False,
        nodelist=list(range(cfg.data.max_node_num)),
        keep_zero_weights=True,
    )

    adj_rescaled = adj_samples * interference_scale + interference_min
    adj_rescaled = adj_rescaled.numpy()

    graph_list = adjs_to_graphs(
        adj_rescaled,
        is_cuda=False,
        keep_isolates=True,
        keep_zero_weights=True,
        nodelist=list(range(adj_rescaled.shape[-1])),
    )

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

    log_success(f"Rescaled {total_edges} edges to range [{interference_min:.4f}, {interference_max:.4f}]")

    save_path = Path("samples/wireless.png")
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

    try:
        sampling_metrics = MetricsClass(datamodule)
        ref_metrics = compute_reference_metrics(datamodule, sampling_metrics)

        sampled_generated = sampling_metrics._sample_subgraphs(graph_list, len(datamodule.test_graphs))
        metrics_out = sampling_metrics.forward(
            graph_list,
            ref_metrics=ref_metrics,
            local_rank=0,
            test=True,
            sampled_generated=sampled_generated,
        )
        ref_test = ref_metrics.get("test") if isinstance(ref_metrics, dict) else None
        ratio_keys = list(metrics_out.keys())
        if "edge_pairs_used" in ratio_keys:
            ratio_keys.remove("edge_pairs_used")
        ratios = compute_ratios(metrics_out, ref_test or {}, ratio_keys)

        print_header("Wireless sampling metrics")
        for k, v in sorted(metrics_out.items()):
            print_row(k, v)
        if ratios:
            print_header("Ratios vs reference")
            for k, v in sorted(ratios.items()):
                print_row(k, v)

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
                dataset_name="metrofi test vs generated full edges",
            )
            save_figure(hist_fig, save_path.with_name("wireless_edge_weight_hist_full.png"), dpi=300)

            hist_fig_sub = plot_edge_weight_histograms(
                ref_graphs_raw,
                sampled_generated,
                dataset_name="metrofi test vs generated subgraphs",
            )
            save_figure(hist_fig_sub, save_path.with_name("wireless_edge_weight_hist_subgraphs.png"), dpi=300)

        except Exception as exc:
            log_warning(f"Warning: could not plot histograms: {exc}")

        # Adjacency heatmap snapshots plotting helper
        def _collect_snapshots(graphs, max_node_num, max_snapshots, weight_key):
            snapshots = []
            max_value = 0.0
            for graph in graphs:
                if graph.number_of_nodes() == 0:
                    continue
                padded = np.zeros((max_node_num, max_node_num), dtype=np.float32)
                has_edges = False
                for u, v, data in graph.edges(data=True):
                    u_idx, v_idx = int(u), int(v)
                    if u_idx < max_node_num and v_idx < max_node_num:
                        w = float(data.get(weight_key, 0.0))
                        padded[u_idx, v_idx] = w
                        padded[v_idx, u_idx] = w
                        has_edges = True
                if has_edges:
                    local_max = float(np.max(padded))
                    max_value = max(max_value, local_max)
                snapshots.append(padded)
                if len(snapshots) >= max_snapshots:
                    break
            return snapshots, max_value

        def build_weighted_snapshots(graphs, max_node_num, max_snapshots=25):
            snapshots, max_value = _collect_snapshots(graphs, max_node_num, max_snapshots, "weight")
            if max_value <= 0.0:
                fallback_snapshots, fallback_max = _collect_snapshots(
                    graphs, max_node_num, max_snapshots, "weight_norm"
                )
                if fallback_max > 0.0:
                    snapshots, max_value = fallback_snapshots, fallback_max
            return snapshots, max_value

        heatmap_snapshots, vmax = build_weighted_snapshots(samples, cfg.data.max_node_num)
        if heatmap_snapshots:
            grid_cols = min(5, len(heatmap_snapshots))
            grid_rows = math.ceil(len(heatmap_snapshots) / grid_cols)
            vmin_heat = interference_min if interference_scale > 0 else 0.0
            vmax_heat = max(vmax, interference_max) if interference_scale > 0 else (vmax if vmax > 0 else 1.0)
            heatmap_fig = plot_heatmap_snapshots(
                heatmap_snapshots,
                grid_shape=(grid_rows, grid_cols),
                cmap="magma",
                vmin=vmin_heat,
                vmax=vmax_heat,
            )
            save_figure(heatmap_fig, save_path.with_name("wireless_weight_heatmaps_full.png"), dpi=300)

    except Exception as exc:
        log_error(f"Warning: could not compute metrics: {exc}")

def run_tune(args, cfg, model_name):
    from configs.config_tune import get_tune_config
    from src.parameter_search.tuning import TuningSettings, run_tuning

    search_space, objective, metrics_alias = get_tune_config(model_name, cfg)

    settings = TuningSettings(
        objective=objective,
        metric_key=None,
        metrics_alias=metrics_alias,
        max_trials=None,
        seed=None,
        device=cfg.general.device,
        num_graphs=None,
        ckpt_path=args.checkpoint,
        results_path=None,
        store_name=args.store_name,
        verbose=True,
        suppress_external_output=True,
        search_space=search_space,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
    )

    run_tuning(cfg, settings)

# CLI parser and main entrypoint
def main():

    # Setup subparsers
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["pa", "sbm", "sbm_2comms", "tree", "tree_graphon", "planar", "metrofi"],
        help="Specific diffusion model/config type to use."
    )
    parent_parser.add_argument("--seed", type=int, default=None, help="Override default config seed.")
    parent_parser.add_argument("--device", type=str, default=None, help="Override general device (e.g. cuda, cpu, cuda:0).")

    parser = argparse.ArgumentParser(description="Unified Jacobi Graph Diffusion Entrypoint")
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Mode to run.")

    # Train subparser
    subparsers.add_parser("train", parents=[parent_parser], help="Train the model.")

    # Gen subparser
    gen_parser = subparsers.add_parser("gen", parents=[parent_parser], help="Generate graphs using a checkpoint.")
    gen_parser.add_argument("--checkpoint", "--ckpt-path", type=str, default=None, help="Path to custom weights file (.ckpt or .pth).")
    gen_parser.add_argument("--save-graphs-path", type=str, default=None, help="Output path to save generated graphs as a pickle.")
    gen_parser.add_argument("--load-graphs-path", type=str, default=None, help="Optional path to a saved graphs pickle to evaluate instead of generating.")
    gen_parser.add_argument("--expected-num-graphs", type=int, default=None, help="If set, raise an error unless loaded/generated graph list size matches this.")
    gen_parser.add_argument("--json-out", type=str, default=None, help="Save final evaluation metrics to this JSON file.")
    gen_parser.add_argument("--kernel", type=str, default="emd", help="Kernel name (default: 'emd').")
    gen_parser.add_argument("--n-folds", type=int, default=1, help="Number of folds to evaluate.")
    gen_parser.add_argument("--min-nodes", type=int, default=None, help="Minimum number of nodes to sample.")
    gen_parser.add_argument("--max-nodes", type=int, default=None, help="Maximum number of nodes to sample.")
    gen_parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to generate (overrides cfg.sampler.test_graphs).")
    gen_parser.add_argument("--no-ema", action="store_true", help="Disable loading EMA checkpoint weights.")
    gen_parser.add_argument("--skip-size-ref", action="store_true", help="Skip loading size-specific reference metrics.")
    gen_parser.add_argument("--no-average-ratio-to-size-ref", dest="no_average_ratio_to_size_ref", action="store_true", help="Skip calculating size-matched reference ratios.")
    gen_parser.set_defaults(no_average_ratio_to_size_ref=False)

    # Tune subparser
    tune_parser = subparsers.add_parser("tune", parents=[parent_parser], help="Tune hyperparams via grid search.")
    tune_parser.add_argument("--checkpoint", "--ckpt-path", type=str, default=None, help="Forced checkpoint path during tuning.")
    tune_parser.add_argument("--min-nodes", type=int, default=None, help="Minimum number of nodes to sample during tuning.")
    tune_parser.add_argument("--max-nodes", type=int, default=None, help="Maximum number of nodes to sample during tuning.")
    tune_parser.add_argument("--store-name", type=str, default=None, help="Hyperparameter store filename (.json in hyperparams/).")

    args = parser.parse_args()

    # Dynamic imports based on chosen model
    MainConfig, MetricsClass, DatasetClass, wandb_name = load_model_components(args.model)

    # Instantiate config and apply CLI overrides
    cfg = OmegaConf.structured(MainConfig())
    if args.seed is not None:
        cfg.general.seed = args.seed
    if args.device is not None:
        cfg.general.device = args.device
    if args.mode == "gen" and args.num_samples is not None:
        cfg.sampler.test_graphs = args.num_samples

    # Global seeding
    _ = pl.seed_everything(cfg.general.seed, workers=True)

    # Delegate execution
    if args.mode == "train":
        run_train(args, cfg, wandb_name, MetricsClass, DatasetClass)
    elif args.mode == "gen":
        run_gen(args, cfg, MetricsClass, DatasetClass, args.model)
    elif args.mode == "tune":
        run_tune(args, cfg, args.model)

if __name__ == "__main__":
    main()
