import argparse
import json
import math
import statistics
from omegaconf import OmegaConf
from pathlib import Path
from typing import Any, Dict
import torch
import pytorch_lightning as pl

from src.models.transformer_model import GraphTransformer
from src.dataset.spectre import SpectreDatasetModule
from src.dataset.utils import (
    DistributionNodes,
    compute_reference_metrics,
    load_graphs_pickle,
    load_size_ref_metrics,
    save_graphs_pickle,
)
from src.sample.sampler import Sampler
from configs.config_sbm_2comms import MainConfig
from src.metrics.val import SBMSamplingMetrics
from src.visualization.plots import save_figure


def parse_args():
    parser = argparse.ArgumentParser(description="Generate SBM (2 communities) graphs with custom node-count distribution.")
    parser.add_argument(
        "--min-nodes",
        type=int,
        default=None,
        help="Minimum number of nodes to sample. Requires --max-nodes. Defaults to dataset distribution.",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        help="Maximum number of nodes to sample. Requires --min-nodes. Defaults to dataset distribution.",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="Optional checkpoint path to force (supports Lightning .ckpt and plain .pth).",
    )
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Disable EMA when loading checkpoints (use raw model weights).",
    )
    parser.add_argument("--order", type=int, default=None, help="Override cfg.sde.order.")
    parser.add_argument(
        "--sample-target",
        type=str,
        choices=("true", "false"),
        default=None,
        help="Override cfg.sde.sample_target.",
    )
    parser.add_argument("--num-scales", type=int, default=None, help="Override cfg.sde.num_scales.")
    parser.add_argument("--eps-sde", type=float, default=None, help="Override cfg.sde.eps_sde.")
    parser.add_argument("--eps-score", type=float, default=None, help="Override cfg.sde.eps_score.")
    parser.add_argument("--time-schedule", type=str, default=None, help="Override cfg.sde.time_schedule.")
    parser.add_argument(
        "--time-schedule-power",
        type=float,
        default=None,
        help="Override cfg.sde.time_schedule_power.",
    )
    parser.add_argument("--eps-time", type=float, default=None, help="Override cfg.sampler.eps_time.")
    parser.add_argument(
        "--predictor",
        type=str,
        choices=("em", "heun", "milstein"),
        default=None,
        help="Override cfg.sampler.predictor.",
    )
    parser.add_argument(
        "--use-corrector",
        type=str,
        choices=("true", "false"),
        default=None,
        help="Override cfg.sampler.use_corrector.",
    )
    parser.add_argument("--snr", type=float, default=None, help="Override cfg.sampler.snr.")
    parser.add_argument("--scale-eps", type=float, default=None, help="Override cfg.sampler.scale_eps.")
    parser.add_argument("--n-steps", type=int, default=None, help="Override cfg.sampler.n_steps.")
    parser.add_argument(
        "--load-graphs-path",
        type=str,
        default=None,
        help="Optional path to a saved graph pickle. If provided, skip generation and only run evaluation.",
    )
    parser.add_argument(
        "--save-graphs-path",
        type=str,
        default=None,
        help="Optional path where newly generated graphs will be saved as a pickle.",
    )
    parser.add_argument(
        "--no-average-ratio-to-size-ref",
        dest="average_ratio_to_size_ref",
        action="store_false",
        help="Disable size-matched reference ratio computation.",
    )
    parser.set_defaults(average_ratio_to_size_ref=True)
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional path to save results in JSON format.",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="emd",
        help="Kernel name to store in JSON.",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=1,
        help="Number of folds to evaluate. With 1, behaves like standard single evaluation.",
    )
    return parser.parse_args()


def _load_graph_transformer_state_dict(raw_ckpt: Dict[str, Any], use_ema: bool) -> Dict[str, Any]:
    # Lightning checkpoints often store params in `state_dict` prefixed by `model.`/`ema_model.`
    if "state_dict" in raw_ckpt and isinstance(raw_ckpt["state_dict"], dict):
        pl_state = raw_ckpt["state_dict"]
        model_state = {k[len("model."):]: v for k, v in pl_state.items() if k.startswith("model.")}
        ema_state = {k[len("ema_model."):]: v for k, v in pl_state.items() if k.startswith("ema_model.")}

        if use_ema and ema_state:
            return ema_state
        if model_state:
            return model_state
        return pl_state

    return raw_ckpt


def build_node_distribution(cfg, datamodule, min_nodes=None, max_nodes=None):
    max_nodes_possible = cfg.data.max_node_num + 1
    base_prob = datamodule.node_counts(max_nodes_possible)
    if base_prob.numel() < max_nodes_possible:
        padding = torch.zeros(max_nodes_possible - base_prob.numel(), dtype=base_prob.dtype)
        base_prob = torch.cat([base_prob, padding], dim=0)

    if min_nodes is None and max_nodes is None:
        return DistributionNodes(prob=base_prob)

    if min_nodes is None or max_nodes is None:
        raise ValueError("Both --min-nodes and --max-nodes must be provided together.")
    if min_nodes < 1:
        raise ValueError("Minimum number of nodes must be at least 1.")
    if max_nodes < min_nodes:
        raise ValueError("Maximum number of nodes must be greater than or equal to the minimum.")

    max_node_num = cfg.data.max_node_num
    if max_nodes > max_node_num:
        raise ValueError(
            f"Requested max_nodes={max_nodes} exceeds cfg.data.max_node_num={max_node_num}. "
            "Please increase the configuration before building the distribution."
        )

    probs = torch.zeros_like(base_prob)
    probs[min_nodes : max_nodes + 1] = 1.0
    total_mass = probs.sum()
    if not torch.isfinite(total_mass) or total_mass <= 0:
        raise ValueError("Probability mass over the requested range is zero; check the node range.")
    probs /= total_mass
    return DistributionNodes(prob=probs)


def resolve_saved_graphs_output_path(args):
    if args.save_graphs_path is not None:
        return Path(args.save_graphs_path)
    return Path("samples/test_sbm_2comms_graphs.pkl")


def _print_ratio_block(metrics, suffix="", title="Ratios"):
    print('------------------------------------------------------------------------------------')
    print(title)
    print('------------------------------------------------------------------------------------')
    keys = sorted([k for k in metrics if k.endswith('_ratio' + suffix) or k.endswith('average_ratio' + suffix)])
    if not keys:
        print("No ratios available in this block.")
        return
    for k in keys:
        print(f"{k} - {metrics[k]}")


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


def main():
    args = parse_args()
    cfg = OmegaConf.structured(MainConfig())
    if getattr(cfg.train, "training_mode", "graph") == "direct_score":
        cfg.model.output_dims = dict(cfg.model.score_output_dims)
    if args.no_ema:
        cfg.train.use_ema = False
    if args.order is not None:
        cfg.sde.order = args.order
    if args.sample_target is not None:
        cfg.sde.sample_target = args.sample_target.lower() == "true"
    if args.num_scales is not None:
        cfg.sde.num_scales = args.num_scales
    if args.eps_sde is not None:
        cfg.sde.eps_sde = args.eps_sde
    if args.eps_score is not None:
        cfg.sde.eps_score = args.eps_score
    if args.time_schedule is not None:
        cfg.sde.time_schedule = args.time_schedule
    if args.time_schedule_power is not None:
        cfg.sde.time_schedule_power = args.time_schedule_power
    if args.eps_time is not None:
        cfg.sampler.eps_time = args.eps_time
    if args.predictor is not None:
        cfg.sampler.predictor = args.predictor
    if args.use_corrector is not None:
        cfg.sampler.use_corrector = args.use_corrector.lower() == "true"
    if args.snr is not None:
        cfg.sampler.snr = args.snr
    if args.scale_eps is not None:
        cfg.sampler.scale_eps = args.scale_eps
    if args.n_steps is not None:
        cfg.sampler.n_steps = args.n_steps

    if args.min_nodes is not None or args.max_nodes is not None:
        if args.min_nodes is None or args.max_nodes is None:
            raise ValueError("Both --min-nodes and --max-nodes must be specified.")
        if args.max_nodes > cfg.data.max_node_num:
            cfg.data.max_node_num = args.max_nodes
        if args.max_nodes > cfg.sampler.num_nodes:
            cfg.sampler.num_nodes = args.max_nodes

    if torch.cuda.is_available():
        if not str(cfg.general.device).startswith("cuda"):
            cfg.general.device = "cuda"
    elif str(cfg.general.device).startswith("cuda"):
        print("CUDA not available, falling back to CPU.")
        cfg.general.device = "cpu"

    _ = pl.seed_everything(cfg.general.seed, workers=True)

    datamodule = SpectreDatasetModule(cfg)
    datamodule.setup()

    node_dist = build_node_distribution(cfg, datamodule, args.min_nodes, args.max_nodes)

    sampling_metrics = SBMSamplingMetrics(datamodule)
    ref_metrics = compute_reference_metrics(datamodule, sampling_metrics)

    if args.load_graphs_path is not None:
        samples = load_graphs_pickle(args.load_graphs_path)
        print(f"Loaded saved graphs from {args.load_graphs_path}")
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

        if args.ckpt_path is not None:
            weight_path = Path(args.ckpt_path)
            if not weight_path.exists():
                raise FileNotFoundError(f"Provided checkpoint path does not exist: {weight_path}")
        else:
            ckpt_dir = Path("checkpoints") / cfg.data.data
            ema_path = ckpt_dir / "weights_ema.pth"
            weights_path = ckpt_dir / "weights.pth"
            if cfg.train.use_ema and ema_path.exists():
                weight_path = ema_path
            elif weights_path.exists():
                weight_path = weights_path
            else:
                raise FileNotFoundError(
                    f"Could not find checkpoint for '{cfg.data.data}'. "
                    f"Looked for {ema_path} and {weights_path}."
                )

        raw_ckpt = torch.load(weight_path, map_location="cpu")
        state_dict = _load_graph_transformer_state_dict(raw_ckpt, use_ema=cfg.train.use_ema) if isinstance(raw_ckpt, dict) else raw_ckpt
        model.load_state_dict(state_dict)
        model = model.to(cfg.general.device)
        model.eval()

        pl.seed_everything(cfg.general.seed, workers=True)
        sampler = Sampler(cfg=cfg, model=model, node_dist=node_dist)
        samples, fig = sampler.sample()

        save_path = Path("samples/test_sbm_2comms.png")
        save_figure(fig, save_path, dpi=300)
        save_graphs_path = resolve_saved_graphs_output_path(args)
        save_graphs_pickle(samples, save_graphs_path)
        print(f"Saved generated graphs to {save_graphs_path}")

    extra_sampling_metrics = None
    if args.average_ratio_to_size_ref:
        if args.min_nodes is not None and args.max_nodes is not None and args.min_nodes == args.max_nodes:
            try:
                size_ref_metrics, size_ref_sampling_metrics = load_size_ref_metrics(
                    cfg=cfg,
                    metrics_cls=SBMSamplingMetrics,
                    target_nodes=args.min_nodes,
                )
                extra_sampling_metrics = {
                    "size_ref": (size_ref_metrics, size_ref_sampling_metrics)
                }
            except Exception as e:
                print(f"INFO: Could not load size-matched reference metrics: {e}")
                print("      Only training-distribution ratios will be reported.")
                args.average_ratio_to_size_ref = False
        else:
            if args.min_nodes != args.max_nodes:
                 print("INFO: --min-nodes != --max-nodes. Skipping size-matched reference comparison.")
            args.average_ratio_to_size_ref = False

    if args.n_folds < 1:
        raise ValueError("--n-folds must be at least 1.")

    fold_metrics, fold_metrics_extra = _evaluate_folds(
        samples, args.n_folds, sampling_metrics, ref_metrics, extra_sampling_metrics
    )

    if args.n_folds > 1:
        print('------------------------------------------------------------------------------------')
        print(f'Per-fold metrics ({len(fold_metrics)} folds)')
        print('------------------------------------------------------------------------------------')
        for idx, fold_result in enumerate(fold_metrics, start=1):
            print(f"Fold {idx}")
            for key in sorted(fold_result.keys()):
                if isinstance(fold_result[key], (int, float)):
                    print(f"{key} - {fold_result[key]}")

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

    print('------------------------------------------------------------------------------------')
    if args.n_folds > 1:
        for k in ref_metrics['val']:
            if k in metrics:
                print(f"{k} ref. / gen mean±std - {ref_metrics['val'][k]} / {metrics[k]['mean']} ± {metrics[k]['std']}")
    else:
        for k in ref_metrics['val']:
            print(f"{k} ref. / gen. - {ref_metrics['val'][k]} / {metrics[k]}")

    if args.n_folds == 1:
        _print_ratio_block(metrics, title="Training-distribution ratios")

    if extra_sampling_metrics is not None and "size_ref" in metrics_extra and args.n_folds == 1:
        size_ref_res = metrics_extra["size_ref"]
        # We need the reference values from the dict we loaded
        ref_vals = extra_sampling_metrics["size_ref"][0]
        
        print('------------------------------------------------------------------------------------')
        print('Size-matched reference comparison')
        print('------------------------------------------------------------------------------------')
        for k in sorted(ref_vals['val'].keys()):
            if k in size_ref_res:
                print(f"{k} size-ref / gen. - {ref_vals['val'][k]} / {size_ref_res[k]}")
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
        
        entry = {
            "size": _resolve_size_summary(samples, args, cfg),
            "accuracy": metrics.get("sbm_acc", 0.0),
            "acc_extra": {},
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

        # Load existing or create new
        out_path = Path(args.json_out)
        data = {"dataset_name": cfg.data.data, "kernel": args.kernel, "entries": []}
        if out_path.exists():
            with open(out_path, "r") as f:
                try:
                    data = json.load(f)
                except Exception as e:
                    print(f"Warning: could not load existing JSON at {out_path}: {e}")

        # Update or append
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
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
