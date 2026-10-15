"""
bspline_net (Github)

Copyright 2026 Carnegie Mellon University.

NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

Licensed under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.

[DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.

This Software includes and/or makes use of Third-Party Software each subject to its own license.

DM26-0750
"""
# PAPER: IEEE 2026 (knot placement); also used by the TMLR analysis
"""Knot placement model for adaptive B-spline knot spacing.

Predicts spacing/activity from initial conditions, which is then
converted to interior knot positions.
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig

from .utils import build_knot_vector, density_to_knots


class ActivityModel(nn.Module):
    """Predicts adaptive spacing from initial conditions.

    Architecture:
        - MLP to map IC to per-timestep features
        - Conv1d layers to smooth and refine spacing prediction

    Args:
        n_dim: State space dimension
        n_samples: Number of time samples
        hidden: Hidden layer size
        kernel_size: Conv kernel size
    """

    def __init__(
        self,
        n_dim: int,
        n_samples: int,
        hidden: int = 128,
        n_layers_encoder: int = 2,
        ch: int = 6,
        seq_start: int = 32,
        kernel_1: int = 5,
        kernel_2: int = 3,
        dropout_rate: float = 0.10,
    ):
        super().__init__()
        # ch = 16 #n_dim // 2
        layers_encoder = [nn.Linear(n_dim, hidden), nn.SiLU()]
        for i in range(n_layers_encoder):
            layers_encoder.extend([nn.Linear(hidden, hidden), nn.SiLU()])
        self.encoder = nn.Sequential(*layers_encoder)

        self.to_sequence = nn.Sequential(
            nn.Linear(hidden, ch * seq_start),
            nn.SiLU(),
            nn.Unflatten(1, (ch, seq_start)),  # (B, ch, seq_start)
            nn.BatchNorm1d(ch),
        )

        self.upsample_blocks = nn.Sequential(
            # 32 -> 128
            nn.Upsample(size=128, mode="linear", align_corners=False),
            nn.Conv1d(ch, ch, kernel_size=kernel_1, padding=kernel_1 // 2),
            nn.SiLU(),
            # 128 -> 256
            nn.Upsample(size=256, mode="linear", align_corners=False),
            nn.Conv1d(ch, ch, kernel_size=kernel_1, padding=kernel_1 // 2),
            nn.SiLU(),
            # 256 -> 501
            nn.Upsample(size=n_samples, mode="linear", align_corners=False),
            nn.Conv1d(ch, ch, kernel_size=kernel_2, padding=kernel_2 // 2),
            nn.SiLU(),
        )

        # Residual refinement
        self.refine = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=kernel_2, padding=kernel_2 // 2),
            nn.SiLU(),
        )

        # Channel mix:
        self.output = nn.Conv1d(ch, 1, kernel_size=1)

    def forward(self, ic: torch.Tensor) -> torch.Tensor:
        x = self.encoder(ic)
        x = self.to_sequence(x)
        x = self.upsample_blocks(x)
        x = self.refine(x) + x  # residual
        x = self.output(x)  # (B, 1, n_samples)
        return x.squeeze(1)  # (B, n_samples)
        return torch.abs(x)  # Estimated ||f''||_\inf


class LitKnotModel(pl.LightningModule):
    """Lightning wrapper for knot placement model.

    Trains the ActivityModel to predict spacing that matches
    the computed adaptive spacing from trajectories.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.max_error = cfg.model.knots.max_error

        self.model = ActivityModel(
            n_dim=len(cfg.data.R),
            n_samples=cfg.data.n_samples,
            hidden=cfg.model.knots.hidden,
            n_layers_encoder=2,
            ch=6,
            seq_start=32,
            kernel_1=cfg.model.knots.kernel_1,
            kernel_2=cfg.model.knots.kernel_2,
        )

        # Store R as buffer so it moves with model
        self.register_buffer("R", torch.tensor(cfg.data.R, dtype=torch.float32))

    def forward(self, norm_ics: torch.Tensor) -> torch.Tensor:
        """Predict spacing from normalized ICs."""
        return self.model(norm_ics)

    def predict_knots(self, norm_ics: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict knot positions and full knot vector.

        Args:
            norm_ics: Normalized initial conditions (batch, n_dim)

        Returns:
            knot_positions: Interior knot positions (batch, n_interior_knots)
            knot_vector: Full clamped knot vector (batch, n_knots_total)
        """
        jerk_activity_pred = self(norm_ics)
        # use formula e_max =  h^2 ||f||_inf /8 where h is spacing (1/density)
        theory_density = torch.sqrt(torch.relu(jerk_activity_pred) / 8 / self.max_error)
        knot_positions, _ = density_to_knots(
            theory_density,
            self.cfg.data.max_density,
            self.cfg.data.min_density,
            self.cfg.data.time_horizon,
        )
        knot_vector = build_knot_vector(knot_positions, self.cfg.data.spline_degree)
        return jerk_activity_pred, knot_positions, knot_vector

    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        """Shared step for training and validation."""
        norm_ics = batch["norm_ics"]
        jerk_activity_max = batch["jerk_activity_max"]
        jerk_activity_max_pred = self(norm_ics)
        loss = ((jerk_activity_max_pred - jerk_activity_max) ** 2).mean()
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=(stage == "train"), on_epoch=True)
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.model.knots.lr,
        )
        return optimizer
