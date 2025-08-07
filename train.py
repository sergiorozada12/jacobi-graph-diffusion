from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from src.train.trainer import DiffusionGraphModule
from src.dataset.synth import SynthGraphDatasetModule
from configs.config_sbm import MainConfig


def main():
    cfg = OmegaConf.structured(MainConfig())
    _ = pl.seed_everything(cfg.general.seed)

    datamodule = SynthGraphDatasetModule(cfg)
    datamodule.setup()

    model = DiffusionGraphModule(cfg)

    logger = WandbLogger(project="jacobi-graph-diffusion", name="tree-test")
    trainer = Trainer(
        accelerator=cfg.general.device,
        devices=[1],
        max_epochs=cfg.train.num_epochs,
        enable_checkpointing=False,
        check_val_every_n_epoch=20,
        log_every_n_steps=1,
        logger=logger,
    )

    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()
