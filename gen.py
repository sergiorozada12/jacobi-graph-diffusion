from omegaconf import OmegaConf
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from src.models.transformer_model import GraphTransformer
from src.dataset.synth import SynthGraphDatasetModule
from src.sample.sampler import Sampler
from configs.config_sbm import MainConfig


def main():
    cfg = OmegaConf.structured(MainConfig())
    _ = pl.seed_everything(cfg.general.seed)

    datamodule = SynthGraphDatasetModule(cfg)
    datamodule.setup()

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

    sampler = Sampler(cfg=cfg, model=model)
    _, fig = sampler.sample()

    save_path = f"samples/test.png"
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    main()
