from omegaconf import OmegaConf
from pathlib import Path
import torch
import pytorch_lightning as pl

from src.models.transformer_model import GraphTransformer
from src.dataset.spectre import SpectreDatasetModule
from src.dataset.utils import DistributionNodes, compute_reference_metrics
from src.sample.sampler import Sampler
from configs.config_sbm_2comms import MainConfig
from src.metrics.val import SBMSamplingMetrics
from src.visualization.plots import save_figure


def main():
    cfg = OmegaConf.structured(MainConfig())

    if torch.cuda.is_available():
        if not str(cfg.general.device).startswith("cuda"):
            cfg.general.device = "cuda"
    elif str(cfg.general.device).startswith("cuda"):
        print("CUDA not available, falling back to CPU.")
        cfg.general.device = "cpu"

    _ = pl.seed_everything(cfg.general.seed, workers=True)

    datamodule = SpectreDatasetModule(cfg)
    datamodule.setup()

    node_dist = DistributionNodes(prob=datamodule.node_counts())

    sampling_metrics = SBMSamplingMetrics(datamodule)
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

    save_path = Path("samples/test_sbm_2comms.png")
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
