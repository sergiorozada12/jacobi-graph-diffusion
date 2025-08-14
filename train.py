from omegaconf import OmegaConf

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from configs.config_planar import MainConfig
from src.train.trainer import DiffusionGraphModule
from src.dataset.synth import SynthGraphDatasetModule, compute_reference_metrics
from src.metrics.val import PlanarSamplingMetrics


def main():
    cfg = OmegaConf.structured(MainConfig())
    _ = pl.seed_everything(cfg.general.seed)

    datamodule = SynthGraphDatasetModule(cfg)
    datamodule.setup()

    sampling_metrics = PlanarSamplingMetrics(datamodule)
    ref_metrics = compute_reference_metrics(datamodule, sampling_metrics)

    model = DiffusionGraphModule(
        cfg=cfg,
        sampling_metrics=sampling_metrics,
        ref_metrics=ref_metrics
    )

    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{cfg.data.data}",
        filename="{epoch}",
        save_top_k=-1,
        every_n_epochs=cfg.general.save_checkpoint_every_n_epochs,
        save_last=True,
    )
    callbacks.append(checkpoint_callback)

    logger = WandbLogger(project="jacobi-graph-diffusion", name="planar-new-test")
    trainer = Trainer(
        accelerator=cfg.general.device,
        devices=[1],
        strategy="ddp_find_unused_parameters_true",  # Needed to load old checkpoints
        max_epochs=cfg.train.num_epochs,
        check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
        log_every_n_steps=1,
        callbacks=callbacks,
        logger=logger,
    )

    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()
