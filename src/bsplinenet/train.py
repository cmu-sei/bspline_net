"""
bspline_net (Github)

Copyright 2026 Carnegie Mellon University.

NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

Licensed under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.

[DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.

This Software includes and/or makes use of Third-Party Software each subject to its own license.

DM26-0750
"""
#!/usr/bin/env python3
"""Main training script for B-spline trajectory learning.

Usage:
    python -m bsplinenet.train stage=knots
    python -m bsplinenet.train stage=cp
    python -m bsplinenet.train stage=joint knot_checkpoint=path/to/knot.ckpt cp_checkpoint=path/to/cp.ckpt

    # With overrides
    python -m bsplinenet.train stage=knots model.knots.lr=0.001 data.batch_size=64

    # Hyperparameter sweep
    python -m bsplinenet.train --multirun stage=knots model.knots.lr=0.01,0.001,0.0001
"""

import os
from datetime import datetime
from functools import partial
from pathlib import Path

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from .callbacks import get_visualization_callback
from .cp_model import LitCPModel
from .data import compute_derived_data_config, get_datamodule
from .joint_model import LitJointModel
from .knot_model import LitKnotModel
from .traj_model import LitTrajModel

# from torchinfo import summary

# import tensorflow as tf
# tf.config.set_visible_devices([], 'GPU')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_model(stage: str, cfg: DictConfig):
    """Get the appropriate model for the training stage."""
    if stage == "knots":
        return LitKnotModel(cfg)
    elif stage == "traj":
        return LitTrajModel(cfg)
    elif stage == "cp":
        return LitCPModel.from_checkpoints(cfg, traj_checkpoint=cfg.get("traj_checkpoint"))
    elif stage == "joint":
        return LitJointModel.from_checkpoints(
            cfg,
            knot_checkpoint=cfg.get("knot_checkpoint"),
            cp_checkpoint=cfg.get("cp_checkpoint"),
        )
    else:
        raise ValueError(f"Unknown stage: {stage}. Must be one of: knots, cp, joint")


def get_callbacks(cfg: DictConfig, stage: str, logger):
    """Get training callbacks."""
    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
    ]

    checkpoint_dir = Path("checkpoints") / cfg.trainer.log_dir / stage / cfg.experiment_name
    callbacks.append(
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"{stage}" + "-{epoch:03d}-{val/loss:.5f}",
            monitor="val/loss" if stage == "knots" or stage == "cp" else "val/traj_loss",
            mode="min",
            save_top_k=cfg.trainer.save_top_k,
            save_last=True,
        )
    )

    # Early stopping
    if cfg.trainer.early_stopping.enabled:
        callbacks.append(
            EarlyStopping(
                monitor=cfg.trainer.scheduler.monitor,
                patience=cfg.trainer.early_stopping.patience,
                mode="min",
                check_finite=cfg.trainer.early_stopping.check_finite,
                strict=False,
            )
        )

    if cfg.trainer.vis_every_n_epochs > 0:
        callbacks.append(get_visualization_callback(stage, cfg.trainer.vis_every_n_epochs))
    return callbacks


def get_logger(cfg: DictConfig, stage: str):
    """Get training logger."""
    return TensorBoardLogger(
        save_dir=str(Path("outputs") / cfg.trainer.log_dir),
        name=stage,
        version=cfg.get("experiment_name", None),
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""
    # Compute derived config values
    cfg = compute_derived_data_config(cfg)
    if cfg.get("experiment_name", None) is None:
        cfg.experiment_name = f"version_0_{datetime.now():%Y%m%d_%H%M%S}"
    # Print config
    print("=" * 60)
    print(f"Training Stage: {cfg.stage}")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))

    # Seed
    pl.seed_everything(cfg.seed, workers=True)

    # Get datamodule
    datamodule = get_datamodule(cfg.stage, cfg)

    # Get model
    model = get_model(cfg.stage, cfg)
    # if cfg.stage =="knots":
    #    summary(model.model, input_size=(1, 12), depth=3)
    # if cfg.stage=="cp":
    #    summary(model.model, input_size=((2,108),(2, 12)), depth=3)

    # Get callbacks and logger
    logger = get_logger(cfg, cfg.stage)
    callbacks = get_callbacks(cfg, cfg.stage, logger)

    # Create trainer
    trainer = pl.Trainer(
        min_epochs=cfg.trainer.min_epochs,
        max_epochs=cfg.trainer.max_epochs,
        # accelerator=cfg.trainer.accelerator,
        strategy=cfg.trainer.strategy,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        callbacks=callbacks,
        logger=logger,
        deterministic=cfg.trainer.deterministic,
    )

    # Train
    try:
        trainer.fit(model, datamodule=datamodule)
        print(trainer.callbacks)
        # if trainer.strategy=="ddp_spawn":
        #    trainer.callbacks[-1].path

        # Print best checkpoint
        print("\n" + "=" * 60)
        print("Training Complete")
        print("=" * 60)
        print(f"Best model: {trainer.checkpoint_callback.best_model_path}")
        best_score = trainer.checkpoint_callback.best_model_score
        if best_score is not None:
            print(f"Best val/loss: {best_score:.6f}")
        else:
            print("No best model score available")

        # Return metric for Optuna (when using --multirun)
        val_loss = trainer.callback_metrics.get("val/loss")
        return val_loss.item() if val_loss else float("inf")

    except Exception as e:
        print(f"Trial failed.\n{e}")
        return float("inf")


if __name__ == "__main__":
    main()
