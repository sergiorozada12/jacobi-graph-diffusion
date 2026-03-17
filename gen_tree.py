import argparse
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
#from configs.config_tree import MainConfig
from configs.config_tree_graphon import MainConfig
from src.metrics.val import TreeSamplingMetrics
from src.visualization.plots import save_figure


def parse_args():
    parser = argparse.ArgumentParser(description="Generate tree graphs with optional custom node-count distribution.")
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
    return Path("samples/test_graphs.pkl")


def _print_ratio_block(metrics, suffix="", title="Ratios"):
    print('------------------------------------------------------------------------------------')
    print(title)
    print('------------------------------------------------------------------------------------')
    for k in metrics:
        if not k.endswith('_ratio' + suffix) and not k.endswith('average_ratio' + suffix):
            continue
        print(f"{k} - {metrics[k]}")


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

    sampling_metrics = TreeSamplingMetrics(datamodule)
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

        save_path = Path("samples/test.png")
        save_figure(fig, save_path, dpi=300)
        save_graphs_path = resolve_saved_graphs_output_path(args)
        save_graphs_pickle(samples, save_graphs_path)
        print(f"Saved generated graphs to {save_graphs_path}")

    extra_ref_metrics = None
    if args.average_ratio_to_size_ref:
        if args.min_nodes is None or args.max_nodes is None:
            raise ValueError("Size-ref ratio computation requires both --min-nodes and --max-nodes.")
        if args.min_nodes != args.max_nodes:
            raise ValueError("Size-ref ratio computation requires --min-nodes and --max-nodes to be equal.")
        size_ref_metrics = load_size_ref_metrics(
            cfg=cfg,
            metrics_cls=TreeSamplingMetrics,
            target_nodes=args.min_nodes,
        )
        extra_ref_metrics = {"size_ref": size_ref_metrics}

    sampling_metrics.reset()
    metrics = sampling_metrics.forward(
        samples,
        ref_metrics=ref_metrics,
        extra_ref_metrics=extra_ref_metrics,
        local_rank=0,
        test=True,
    )

    print('------------------------------------------------------------------------------------')
    for k in ref_metrics['val']:
        print(f"{k} ref. / gen. - {ref_metrics['val'][k]} / {metrics[k]}")

    _print_ratio_block(metrics, title="Training-distribution ratios")

    if extra_ref_metrics is not None:
        print('------------------------------------------------------------------------------------')
        print('Size-matched reference comparison')
        print('------------------------------------------------------------------------------------')
        for k in size_ref_metrics['val']:
            print(f"{k} size-ref / gen. - {size_ref_metrics['val'][k]} / {metrics[k]}")
        _print_ratio_block(metrics, suffix="_size_ref", title="Size-matched reference ratios")

if __name__ == "__main__":
    main()
