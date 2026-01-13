from omegaconf import OmegaConf
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from configs.config_tree import MainConfig
#from configs.config_sbm import MainConfig
#from configs.config_pa import MainConfig
#from configs.config_sbm_2comms import MainConfig
#from configs.config_planar import MainConfig


from src.train.trainer_graph import DiffusionGraphModule, DiffusionWeightedGraphModule
from src.train.trainer_score import DiffusionScoreModule
from src.dataset.synth import SynthGraphDatasetModule
from src.dataset.spectre import SpectreDatasetModule
from src.dataset.utils import DistributionNodes, compute_reference_metrics
from src.metrics.val import TreeSamplingMetrics, SBMSamplingMetrics, PlanarSamplingMetrics, PASamplingMetrics


def main():
    cfg = OmegaConf.structured(MainConfig())
    _ = pl.seed_everything(cfg.general.seed)

    train_mode = getattr(cfg.train, "training_mode", "graph")
    if train_mode == "direct_score":
        cfg.model.output_dims = dict(cfg.model.score_output_dims)
    else:
        cfg.model.output_dims = dict(cfg.model.output_dims)

    datamodule = SpectreDatasetModule(cfg)
    datamodule.setup()

    node_dist = DistributionNodes(prob=datamodule.node_counts())

    sampling_metrics = TreeSamplingMetrics(datamodule)
    # sampling_metrics = SBMSamplingMetrics(datamodule)
    # sampling_metrics = PlanarSamplingMetrics(datamodule)
    # sampling_metrics = PASamplingMetrics(datamodule)
    ref_metrics = compute_reference_metrics(datamodule, sampling_metrics)

    if train_mode == "direct_score":
        module_cls = DiffusionScoreModule
    elif train_mode == "weighted":
        module_cls = DiffusionWeightedGraphModule
    else:
        module_cls = DiffusionGraphModule
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

    # wandb_run_id = "or4yxkkd" # SBM
    # wandb_run_id = "k3heg161" # Tree 2
    # wandb_run_id = "dhoxuvon" # Planar 2
    # wandb_run_id = "zjd7dbh7" # Planar
    # wandb_run_id = "y1dt6kl0" # Tree
    wandb_run_id = None
    logger = WandbLogger(
        project="jacobi-graph-diffusion",
        name="tree-spectre",
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
