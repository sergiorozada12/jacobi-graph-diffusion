from omegaconf import OmegaConf
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from src.models.transformer_model import GraphTransformer
from src.dataset.synth import SynthGraphDatasetModule, compute_reference_metrics
from src.dataset.utils import DistributionNodes
from src.sample.sampler import Sampler
from configs.config_tree import MainConfig
from src.metrics.val import TreeSamplingMetrics


def main():
    cfg = OmegaConf.structured(MainConfig())
    _ = pl.seed_everything(cfg.general.seed)

    datamodule = SynthGraphDatasetModule(cfg)
    datamodule.setup()

    node_dist = DistributionNodes(prob=datamodule.node_counts())

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
    model.load_state_dict(torch.load(f"checkpoints/{cfg.data.data}/weights.pth"))

    sampler = Sampler(cfg=cfg, model=model, node_dist=node_dist)
    samples, fig = sampler.sample()

    save_path = f"samples/test.png"
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

    sampling_metrics.reset()
    metrics = sampling_metrics.forward(
        samples,
        ref_metrics=ref_metrics,
        local_rank=0,
        test=False,
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
