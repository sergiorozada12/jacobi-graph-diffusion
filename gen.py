from omegaconf import OmegaConf
import torch
import pytorch_lightning as pl

from src.models.marginal_network import MarginalNetwork
from src.dataset.synth import SynthGraphDatasetModule
from src.sample.sampler import Sampler
from configs.config_ego import MainConfig


def main():
    cfg = OmegaConf.structured(MainConfig())
    _ = pl.seed_everything(cfg.general.seed)

    datamodule = SynthGraphDatasetModule(cfg)
    datamodule.setup()

    model = MarginalNetwork(**cfg.model)
    model.load_state_dict(torch.load(f"checkpoints/{cfg.data.data}.pth"))

    sampler = Sampler(cfg=cfg, datamodule=datamodule, model=model)
    sampler.sample()

if __name__ == "__main__":
    main()
