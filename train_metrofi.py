from omegaconf import OmegaConf
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from configs.config_metrofi import MainConfig

from src.train.trainer_graph import DiffusionWeightedGraphModule
from src.dataset.wireless import WirelessDatasetModule
from src.dataset.utils import DistributionNodes, compute_reference_metrics
from src.metrics.val import WirelessSamplingMetrics


def main():
    cfg = OmegaConf.structured(MainConfig())
    _ = pl.seed_everything(cfg.general.seed)

    cfg.model.output_dims = dict(cfg.model.score_output_dims)

    datamodule = WirelessDatasetModule(cfg)
    datamodule.setup()

    node_dist = DistributionNodes(prob=datamodule.node_counts())

    sampling_metrics = WirelessSamplingMetrics(datamodule)
    ref_metrics = compute_reference_metrics(datamodule, sampling_metrics)

    module_cls = DiffusionWeightedGraphModule
    model = module_cls(
        cfg=cfg,
        sampling_metrics=sampling_metrics,
        ref_metrics=ref_metrics,
        node_dist=node_dist,
    )

    ckpt_dir = Path(f"checkpoints/{cfg.data.data}")
    ckpt_path = ckpt_dir / "last.ckpt"

    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="{epoch}",
        save_top_k=-1,
        every_n_epochs=cfg.general.save_checkpoint_every_n_epochs,
        save_last=True,
    )
    callbacks.append(checkpoint_callback)

    wandb_run_id = None
    logger = WandbLogger(
        project="jacobi-graph-diffusion",
        name="metrofi",
        id=wandb_run_id,
        resume="must" if wandb_run_id else None,
    )

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

    if ckpt_path.exists():
        trainer.fit(model, datamodule=datamodule, ckpt_path=str(ckpt_path))
    else:
        trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()
