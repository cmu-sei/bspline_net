"""
bspline_net (Github)

Copyright 2026 Carnegie Mellon University.

NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

Licensed under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.

[DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.

This Software includes and/or makes use of Third-Party Software each subject to its own license.

DM26-0750
"""
# NOTE: DANGLING CODE. This joint (end-to-end knot + control point) model was never completed
# and is NOT used in either paper (IEEE or TMLR). It is kept only because train.py (stage=joint) and the tests import it.
# It reads cfg.model.cp.rank / metapos, which are no longer in configs/model/cp/default.yaml,
# so stage=joint is not expected to run as-is.
"""Joint model for fine-tuning knot and control point models together."""

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from .cp_model import CPDeepONetModel, LitCPModel
from .knot_model import ActivityModel, LitKnotModel
from .utils import build_knot_vector, compute_trajectory_loss, density_to_knots


class LitJointModel(pl.LightningModule):
    """Joint Lightning model for end-to-end training.

    Combines knot placement and control point prediction models,
    allowing gradients to flow through both.
    """

    def __init__(
        self,
        cfg: DictConfig,
        knot_model: ActivityModel = None,
        cp_model: CPDeepONetModel = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["knot_model", "cp_model"])
        self.cfg = cfg

        # Initialize or use provided models
        if knot_model is None:
            self.knot_model = ActivityModel(
                n_dim=len(cfg.data.R),
                n_samples=cfg.data.n_samples,
                hidden=cfg.model.knots.hidden,
                kernel_1=cfg.model.knots.kernel_1,
                kernel_2=cfg.model.knots.kernel_2,
            )
        else:
            self.knot_model = knot_model

        if cp_model is None:
            self.cp_model = CPDeepONetModel(
                n_dim=len(cfg.data.R),
                n_cp_max=cfg.data.n_cp_max,
                spline_degree=cfg.data.spline_degree,
                widths_ic=cfg.model.cp.widths_ic,
                widths_kcp=cfg.model.cp.widths_kcp,
                R=torch.tensor(cfg.data.R, dtype=torch.float32),
                rank=cfg.model.cp.rank,
                metapos=cfg.model.cp.metapos,
            )
        else:
            self.cp_model = cp_model

        # Store R as buffer
        self.register_buffer("R", torch.tensor(cfg.data.R, dtype=torch.float32))

        # Loss weights
        self.activity_loss_weight = cfg.model.joint.get("activity_loss_weight", 1.0)
        self.trunk_recon_weight = cfg.model.cp.get("trunk_recon_weight", 0.1)
        self.trunk_smooth_weight = cfg.model.cp.get("trunk_smooth_weight", 0.1)
        self.trunk_ortho_weight = cfg.model.cp.get("trunk_ortho_weight", 0.01)
        self.branch_smooth_weight = cfg.model.cp.get("branch_smooth_weight", 0.1)
        self.ic_loss_weight = cfg.model.cp.get("ic_loss_weight", 0.1)

    @classmethod
    def from_checkpoints(
        cls,
        cfg: DictConfig,
        knot_checkpoint: str = None,
        cp_checkpoint: str = None,
    ) -> "LitJointModel":
        """Load joint model from separate checkpoints.

        Args:
            cfg: Configuration
            knot_checkpoint: Path to knot model checkpoint
            cp_checkpoint: Path to CP model checkpoint

        Returns:
            LitJointModel with loaded weights
        """
        model = cls(cfg)

        if knot_checkpoint:
            knot_lit = LitKnotModel.load_from_checkpoint(knot_checkpoint, cfg=cfg)
            model.knot_model = knot_lit.model

        if cp_checkpoint:
            cp_lit = LitCPModel.load_from_checkpoint(cp_checkpoint, cfg=cfg)
            model.cp_model = cp_lit.model

        return model

    def forward(
        self, norm_ics: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Full forward pass through both models.

        Args:
            norm_ics: Normalized initial conditions (batch, n_dim)

        Returns:
            jerk_activity_max_pred: Predicted density of max jerk (2nd derivative) (batch, n_samples)
            knot_vector: Full knot vector (batch, n_knots)
            pred_cp: Predicted control points (batch, n_cp_max, n_dim)
            aux_losses: Auxiliary losses from CP model
        """
        # Knot model
        jerk_activity_max_pred = self.knot_model(norm_ics)
        knot_positions, _ = density_to_knots(
            jerk_activity_max_pred,
            self.cfg.data.max_density,
            self.cfg.data.min_density,
            self.cfg.data.time_horizon,
        )
        knot_vector = build_knot_vector(knot_positions, self.cfg.data.spline_degree)

        # CP model
        pred_cp, aux_losses = self.cp_model(knot_vector, norm_ics)  # , self.R)

        return jerk_activity_max_pred, knot_vector, pred_cp, aux_losses

    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        """Shared step for training and validation."""
        ics = batch["ics"]
        norm_ics = batch["norm_ics"]
        trajectories = batch["trajectories"]
        norm_trajectories = batch["norm_trajectories"]
        jerk_activity_max = batch["jerk_activity_max"]

        # Forward pass
        jerk_activity_max_pred, knot_vector, pred_cp, aux_losses = self(norm_ics)
        norm_pred_cp = pred_cp / self.R

        # Spacing loss
        activity_loss = (
            ((jerk_activity_max_pred - jerk_activity_max) ** 2) / jerk_activity_max**2
        ).mean()

        # Trajectory loss (use spacing_pred as weighting, detached)
        traj_loss, traj_loss_norm, pred_trajectories = compute_trajectory_loss(
            norm_pred_cp,
            knot_vector,
            norm_trajectories,
            jerk_activity_max_pred.detach(),
            d=self.cfg.data.spline_degree,
        )

        # IC loss
        ic_loss = ((pred_cp[:, 0, :] - ics) ** 2).mean()

        # Trunk auxiliary losses
        trunk_loss = (
            self.trunk_recon_weight * aux_losses["recon"]
            + self.trunk_smooth_weight * aux_losses["smooth"]
            + self.trunk_ortho_weight * aux_losses["ortho"]
        )

        # Branch smoothness loss
        branch_loss = self.branch_smooth_weight * aux_losses["branch_smooth"]

        # Total loss
        total_loss = (
            self.activity_loss_weight * activity_loss
            + traj_loss
            + traj_loss_norm
            + self.ic_loss_weight * ic_loss
            + trunk_loss
            + branch_loss
        )

        # Logging
        self.log(f"{stage}/loss", total_loss, prog_bar=True, on_epoch=True)
        self.log(f"{stage}/activity_loss", activity_loss, on_epoch=True)
        self.log(f"{stage}/traj_loss", traj_loss, on_epoch=True)
        self.log(f"{stage}/traj_loss_norm", traj_loss_norm, on_epoch=True)
        self.log(f"{stage}/ic_loss", ic_loss, on_epoch=True)
        self.log(f"{stage}/trunk_loss", trunk_loss, on_epoch=True)
        self.log(f"{stage}/branch_loss", branch_loss, on_epoch=True)

        return total_loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {"params": self.knot_model.parameters(), "lr": self.cfg.model.joint.knot_lr},
                {"params": self.cp_model.parameters(), "lr": self.cfg.model.joint.cp_lr},
            ]
        )
        return optimizer
