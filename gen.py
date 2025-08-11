from omegaconf import OmegaConf
import torch
import pytorch_lightning as pl

from src.models.transformer_model import GraphTransformer
from src.dataset.synth import SynthGraphDatasetModule
from src.sample.sampler import Sampler
from configs.config_ego import MainConfig


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
    model.load_state_dict(torch.load(f"checkpoints/{cfg.data.data}.pth"))

    sampler = Sampler(cfg=cfg, datamodule=datamodule, model=model)
    sampler.sample()

if __name__ == "__main__":
    main()
