from omegaconf import OmegaConf
from pathlib import Path
import os

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from configs.config_qm9 import MainConfig
from src.train.trainer_mol import DiffusionMolModule
from src.dataset.qm9 import QM9DatasetModule
from src.dataset.utils import DistributionNodes
from src.metrics.mol_metrics import BasicMolecularMetrics

def main():
    cfg = OmegaConf.structured(MainConfig())
    _ = pl.seed_everything(cfg.general.seed)

    datamodule = QM9DatasetModule(cfg)
    datamodule.setup()

    node_dist = DistributionNodes(prob=datamodule.node_counts())

    # Ported DeFoG metrics
    train_smiles = datamodule.train_smiles()
    sampling_metrics = BasicMolecularMetrics(dataset_info=datamodule.dataset_info, train_smiles=train_smiles)
    ref_metrics = None # Molecular metrics don't typically use the structural ref_metrics

    model = DiffusionMolModule(
        cfg=cfg,
        sampling_metrics=sampling_metrics,
        ref_metrics=ref_metrics,
        node_dist=node_dist,
        dataset_info=datamodule.dataset_info,
    )

    ckpt_dir = Path(f"checkpoints/{cfg.data.data}")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="{epoch}",
        save_top_k=-1,
        every_n_epochs=cfg.general.save_checkpoint_every_n_epochs,
        save_last=True,
    )
    callbacks.append(checkpoint_callback)

    logger = WandbLogger(
        project="jacobi-graph-diffusion",
        name="qm9-molecular",
    )

    trainer = Trainer(
        accelerator=cfg.general.device,
        devices=[1],
        max_epochs=cfg.train.num_epochs,
        check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
        log_every_n_steps=1,
        callbacks=callbacks,
        logger=logger,
    )

    ckpt_path = ckpt_dir / "last.ckpt"
    if ckpt_path.exists():
        print(f"Resuming from {ckpt_path}")
        trainer.fit(model, datamodule=datamodule, ckpt_path=str(ckpt_path))
    else:
        trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()
