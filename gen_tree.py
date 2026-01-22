import argparse
from omegaconf import OmegaConf
from pathlib import Path
import torch
import pytorch_lightning as pl

from src.models.transformer_model import GraphTransformer
from src.dataset.spectre import SpectreDatasetModule
from src.dataset.utils import DistributionNodes, compute_reference_metrics
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
    return parser.parse_args()


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


def main():
    args = parse_args()
    cfg = OmegaConf.structured(MainConfig())
    if getattr(cfg.train, "training_mode", "graph") == "direct_score":
        cfg.model.output_dims = dict(cfg.model.score_output_dims)

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

    model = GraphTransformer(
        n_layers=cfg.model.n_layers,
        input_dims=cfg.model.input_dims,
        hidden_mlp_dims=cfg.model.hidden_mlp_dims,
        hidden_dims=cfg.model.hidden_dims,
        output_dims=cfg.model.output_dims,
        act_fn_in=torch.nn.ReLU(),
        act_fn_out=torch.nn.ReLU(),
    )
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
    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(cfg.general.device)
    model.eval()

    pl.seed_everything(cfg.general.seed, workers=True)
    sampler = Sampler(cfg=cfg, model=model, node_dist=node_dist)
    samples, fig = sampler.sample()

    save_path = Path("samples/test.png")
    save_figure(fig, save_path, dpi=300)

    sampling_metrics.reset()
    metrics = sampling_metrics.forward(
        samples,
        ref_metrics=ref_metrics,
        local_rank=0,
        test=True,
    )

    print('------------------------------------------------------------------------------------')
    for k in ref_metrics['val']:
        print(f"{k} ref. / gen. - {ref_metrics['val'][k]} / {metrics[k]}")

    print('------------------------------------------------------------------------------------')
    for k in metrics:
        if '_ratio' not in k:
            continue
        print(f"{k} - {metrics[k]}")

if __name__ == "__main__":
    main()
