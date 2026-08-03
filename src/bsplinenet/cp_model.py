"""
bspline_net (Github)

Copyright 2026 Carnegie Mellon University.

NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

Licensed under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.

[DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.

This Software includes and/or makes use of Third-Party Software each subject to its own license.

DM26-0750
"""
"""Control point prediction model.

The trunk learns to approximate the pseudoinverse of the B-spline basis matrix
B(knots). It is trained against the property `cp_embed @ B = I`, with no
dependence on trajectories or the branch.

The branch (TrajectoryModel) is loaded from a pretrained checkpoint and kept
frozen. It is used only at inference and for visualization, where the full
forward returns `pred_cp = R * (cp_embed @ ic_embed)` and reconstructs a
trajectory via the basis.
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torchinfo import summary

from .traj_model import TrajectoryModel
from .utils import bspline_basis


class CPDeepONetModel(nn.Module):
    """Trunk approximator for B^+ plus a frozen branch used for inference.

    The trunk maps knot windows to per-CP weights of shape
    (batch, n_cp, rollout_steps). At inference, those weights act on the
    branch's predicted trajectory `ic_embed` via a bilinear product to produce
    control points.

    The training objective for the trunk is `||cp_embed @ B - I||^2` per
    sample, evaluated only over the sample's actual n_cp (variable across
    samples). This decouples trunk training from the branch.
    """

    def __init__(
        self,
        n_dim: int,
        spline_degree: int,
        widths_ic: list[int],
        widths_kcp: list[int],
        R,
        rollout_steps: int,
        lifted_dim: int,
        spectral_epsilon_min: float = 0.01,
        use_decoder: bool = True,
        window_extra: int = 0,
    ):
        """
        Args:
            n_dim: state-space dimension.
            spline_degree: B-spline degree (d).
            widths_ic, widths_kcp, rollout_steps, lifted_dim, spectral_epsilon_min, use_decoder:
                forwarded to TrajectoryModel for the branch.
            window_extra: extra knots on each side of the base (d+2)-knot window.
                With window_extra=0, each CP sees only its native support; with
                window_extra=k, the trunk's MLP sees (d+2 + 2k) knots, with
                zero-padding at the global knot-vector ends.
            R: per-dim normalization tensor of shape (n_dim,).
        """
        super().__init__()
        self.n_dim = n_dim
        self.spline_degree = spline_degree
        self.rollout_steps = rollout_steps
        self.window_extra = window_extra
        self.window_size = (spline_degree + 2) + 2 * window_extra

        self.register_buffer("R", R if torch.is_tensor(R) else torch.tensor(R, dtype=torch.float32))

        # Branch (used at inference/visualization only; never trained here).
        self.traj_model = TrajectoryModel(
            n_dim=n_dim,
            widths_ic=widths_ic,
            rollout_steps=rollout_steps,
            lifted_dim=lifted_dim,
            spectral_epsilon_min=spectral_epsilon_min,
            use_decoder=use_decoder,
        )

        # Trunk: per-window MLP producing per-CP, per-rollout-slot weights.
        # Input per CP: a window of `self.window_size` knot positions.
        # Output per CP: a vector of length `rollout_steps`.
        layers = []
        prev = self.window_size
        for w in widths_kcp:
            layers.extend([nn.Linear(prev, w), nn.SiLU()])
            prev = w
        layers.append(nn.Linear(prev, rollout_steps))
        self.cp_trunk = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.cp_trunk:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def _windowed_knots(self, knot_vector: torch.Tensor) -> torch.Tensor:
        """Build per-CP windows of size `self.window_size` with zero-padding
        on the boundary.

        knot_vector: (batch, n_knots)
        returns: (batch, n_cp_max, window_size)

        Each CP c's window is centered on the (d+2)-knot span [c, c+d+1] and
        extended by `window_extra` on each side. Out-of-range positions are
        filled with zeros.
        """
        d = self.spline_degree
        k = self.window_extra
        batch, n_knots = knot_vector.shape
        n_cp_max = n_knots - d - 1  # standard B-spline relationship

        # Zero-pad on both ends so that windows near the boundary are well-defined.
        pad = k
        if pad > 0:
            padded = F.pad(knot_vector, (pad, pad), mode="constant", value=0.0)
        else:
            padded = knot_vector

        # Unfold to extract windows.
        windows = padded.unfold(dimension=1, size=self.window_size, step=1)
        # The first `n_cp_max` windows correspond to CPs 0..n_cp_max-1.
        return windows[:, :n_cp_max, :]

    def trunk_forward(self, knot_vector: torch.Tensor) -> torch.Tensor:
        """Run the trunk only; returns cp_embed of shape (batch, n_cp_max, rollout_steps)."""
        windows = self._windowed_knots(knot_vector)
        return self.cp_trunk(windows)

    def _closed_form_cp_embed(
        self, knot_vector: torch.Tensor, end_points: torch.Tensor
    ) -> torch.Tensor:
        """Build per-sample closed-form B^+ (regularized least squares), padded
        into a (batch, n_cp_max, rollout_steps) tensor matching the trunk's
        output shape. Unused rows (beyond each sample's n_cp_i) are zero.

        Regularized: (B^T B + lam I)^(-1) B^T, with lam small for numerical
        stability without meaningfully biasing the inverse.
        """
        d = self.spline_degree
        device = knot_vector.device
        batch, n_knots = knot_vector.shape
        n_cp_max = n_knots - d - 1
        t_rollout = torch.linspace(0, 1, self.rollout_steps, device=device)
        lam = 1e-6

        cp_embed = torch.zeros(batch, n_cp_max, self.rollout_steps, device=device)
        for i in range(batch):
            ep_i = int(end_points[i].item())
            n_cp_i = ep_i - d - 1
            if n_cp_i <= 0:
                continue
            B_i = bspline_basis(t_rollout, knot_vector[i, :ep_i], d)  # (rollout_steps, n_cp_i)
            BtB = B_i.T @ B_i  # (n_cp_i, n_cp_i)
            I_i = torch.eye(n_cp_i, device=device)
            # Solve (B^T B + lam I) X = B^T  -> X = B^+
            B_pinv = torch.linalg.solve(BtB + lam * I_i, B_i.T)  # (n_cp_i, rollout_steps)
            cp_embed[i, :n_cp_i, :] = B_pinv
        return cp_embed

    def forward(
        self,
        knot_vector: torch.Tensor,
        norm_ics: torch.Tensor,
        use_pinv: bool = False,
    ):
        """Full forward: trunk (or closed-form pinv) + branch, producing CPs
        and bookkeeping for visualization.

        Args:
            knot_vector: (batch, n_knots) knot positions.
            norm_ics: (batch, n_dim) normalized initial conditions.
            use_pinv: if True, replace the learned trunk's cp_embed with the
                closed-form B^+ computed per sample. Useful at inference for
                comparing to the analytical solution or when the learned trunk
                is not trusted.

        Returns: (pred_cp, info, cp_embed, ic_embed) matching the visualization
        callback's expected unpacking.
        """
        if use_pinv:
            end_points = _compute_end_points(knot_vector, self.spline_degree)
            cp_embed = self._closed_form_cp_embed(knot_vector, end_points)
        else:
            cp_embed = self.trunk_forward(knot_vector)  # (batch, n_cp_max, rollout_steps)

        traj_out = self.traj_model(norm_ics)
        if isinstance(traj_out, dict):
            ic_embed = traj_out["med"]
        else:
            ic_embed = traj_out
        # Bilinear product, denormalized by R for downstream plotting.
        pred_cp = self.R * torch.einsum("bcs, bsd -> bcd", cp_embed, ic_embed)
        info = {"used_pinv": use_pinv}
        return pred_cp, info, cp_embed, ic_embed


def _trajectory_loss(
    cp_embed: torch.Tensor,
    ic_embed: torch.Tensor,
    norm_traj: torch.Tensor,
    knot_vector: torch.Tensor,
    spline_degree: int,
    rollout_steps: int,
    end_points: torch.Tensor,
) -> torch.Tensor:
    """Mean per-sample ||B @ (cp_embed @ ic_embed) - traj_at_rollout||^2 on the
    actual n_cp per sample (trunk gets to use only the valid CPs).

    cp_embed:  (batch, n_cp_max, rollout_steps)
    ic_embed:  (batch, rollout_steps, n_dim)
    norm_traj: (batch, n_samples_full, n_dim) -- raw true trajectory before
        subsampling to rollout slots; we resample it here.
    knot_vector: (batch, n_knots)
    end_points:  (batch,) int
    """
    batch = knot_vector.shape[0]
    device = knot_vector.device
    d = spline_degree
    t_rollout = torch.linspace(0, 1, rollout_steps, device=device)

    # Subsample true trajectory at rollout slots.
    n_samples_full = norm_traj.shape[1]
    idx = torch.linspace(0, n_samples_full - 1, rollout_steps, device=device).round().long()
    f_true = norm_traj[:, idx, :]  # (batch, rollout_steps, n_dim)

    losses = []
    for i in range(batch):
        ep_i = int(end_points[i].item())
        kv_i = knot_vector[i, :ep_i]
        n_cp_i = ep_i - d - 1
        if n_cp_i <= 0:
            continue
        B_i = bspline_basis(t_rollout, kv_i, d)  # (rollout_steps, n_cp_i)
        cp_i = cp_embed[i, :n_cp_i, :]  # (n_cp_i, rollout_steps)
        # Predicted CPs (bilinear, but in normalized state).
        pred_cp_i = cp_i @ ic_embed[i]  # (n_cp_i, n_dim)
        # Reconstructed trajectory at rollout slots.
        recon_i = B_i @ pred_cp_i  # (rollout_steps, n_dim)
        losses.append(((recon_i - f_true[i]) ** 2).mean())

    if not losses:
        return torch.zeros((), device=device, requires_grad=True)
    return torch.stack(losses).mean()


def _trunk_loss(
    cp_embed: torch.Tensor,
    knot_vector: torch.Tensor,
    spline_degree: int,
    rollout_steps: int,
    end_points: torch.Tensor,
    sym_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Identity residual plus optional pinv-symmetry regularizer.

    Identity term: ||cp_embed @ B - I||^2, mean per sample.
        Pins cp_embed to be a left-inverse of B.

    Symmetry term: ||B @ cp_embed - (B @ cp_embed)^T||^2, mean per sample.
        For a left-inverse cp_embed, this is zero iff cp_embed = B^+ (the
        Moore-Penrose pseudoinverse). Together with the identity term, this
        pins cp_embed uniquely to B^+ without needing pinv as a target.

    Returns: (total, identity_only, sym_only) for logging.
    """
    batch = knot_vector.shape[0]
    device = knot_vector.device
    d = spline_degree
    t_rollout = torch.linspace(0, 1, rollout_steps, device=device)

    id_losses, sym_losses = [], []
    for i in range(batch):
        ep_i = int(end_points[i].item())
        kv_i = knot_vector[i, :ep_i]
        n_cp_i = ep_i - d - 1
        if n_cp_i <= 0:
            continue
        B_i = bspline_basis(t_rollout, kv_i, d)  # (rollout_steps, n_cp_i)
        cp_i = cp_embed[i, :n_cp_i, :]  # (n_cp_i, rollout_steps)
        I_i = torch.eye(n_cp_i, device=device)

        # Identity term: cp_embed B = I
        prod = cp_i @ B_i  # (n_cp_i, n_cp_i)
        id_losses.append(((prod - I_i) ** 2).mean())

        # Symmetry term: B cp_embed = (B cp_embed)^T
        if sym_weight > 0.0:
            proj = B_i @ cp_i  # (rollout_steps, rollout_steps)
            sym_losses.append(((proj - proj.T) ** 2).mean())

    if not id_losses:
        zero = torch.zeros((), device=device, requires_grad=True)
        return zero, zero, zero

    identity_only = torch.stack(id_losses).mean()
    sym_only = torch.stack(sym_losses).mean() if sym_losses else torch.zeros((), device=device)
    total = identity_only + sym_weight * sym_only
    return total, identity_only.detach(), sym_only.detach()


# Back-compat alias.
def _identity_residual_loss(*args, **kwargs):
    total, _, _ = _trunk_loss(*args, **kwargs)
    return total


def _compute_end_points(knot_vector: torch.Tensor, spline_degree: int) -> torch.Tensor:
    """Determine the per-sample number of valid knots based on the convention
    that padding knots have value 1.0 (matching existing data pipeline).
    Returns a (batch,) int tensor with end_point per sample.
    """
    d = spline_degree
    batch, n_knots = knot_vector.shape
    end_points = torch.full((batch,), n_knots, dtype=torch.long, device=knot_vector.device)
    for i in range(batch):
        kv = knot_vector[i]
        eq_ones = (kv == 1).nonzero(as_tuple=False).flatten()
        if eq_ones.numel() > d + 1:
            end_points[i] = int(eq_ones[d + 1].item())
    return end_points


class LitCPModel(pl.LightningModule):
    """Lightning module for trunk-only training against `cp_embed @ B = I`.

    The branch is loaded from a pretrained checkpoint via `from_checkpoints`,
    frozen at load time, and used only for visualization/inference. It does not
    participate in training.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        cp_cfg = cfg.model.cp
        self.model = CPDeepONetModel(
            n_dim=len(cfg.data.R),
            spline_degree=cfg.data.spline_degree,
            widths_ic=cp_cfg.widths_ic,
            widths_kcp=cp_cfg.widths_kcp,
            R=torch.tensor(cfg.data.R, dtype=torch.float32),
            rollout_steps=cp_cfg.rollout_steps,
            lifted_dim=cp_cfg.lifted_dim,
            spectral_epsilon_min=cp_cfg.get("spectral_epsilon_min", 0.01),
            use_decoder=cp_cfg.get("use_decoder", True),
            window_extra=cp_cfg.get("window_extra", 0),
        )
        self.spline_degree = cfg.data.spline_degree
        self.rollout_steps = cp_cfg.rollout_steps
        self.sym_weight = float(cp_cfg.get("sym_weight", 0.0))
        self.traj_weight = float(cp_cfg.get("traj_weight", 0.0))

    # ----- Checkpoint loading -----

    @classmethod
    def from_checkpoints(
        cls,
        cfg: DictConfig,
        traj_checkpoint: str = None,
    ) -> "LitCPModel":
        """Instantiate the module and load a pretrained branch checkpoint.

        The branch is always frozen here: this module exists solely to train
        the trunk, and the branch is along for the ride for visualization.
        """
        model = cls(cfg)
        if traj_checkpoint:
            model._load_traj_checkpoint(traj_checkpoint)
        model._freeze_branch()
        return model

    def _load_traj_checkpoint(self, ckpt_path: str):
        """Load TrajectoryModel weights from a LitTrajModel checkpoint."""
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        src = ckpt["state_dict"]
        prefix = "model."
        traj_state = {k[len(prefix) :]: v for k, v in src.items() if k.startswith(prefix)}
        self.model.traj_model.load_state_dict(traj_state, strict=True)
        print(f"Loaded TrajectoryModel weights from {ckpt_path}")

    def _freeze_branch(self):
        for p in self.model.traj_model.parameters():
            p.requires_grad = False
        print("Branch (TrajectoryModel) frozen; trunk-only training.")

    # ----- Lightning hooks -----

    def setup(self, stage=None):
        if self.global_rank == 0:
            try:
                # Summary based on trunk-only input (knot vector).
                n_knots = self.spline_degree + 1 + 30  # rough placeholder for display
                self.print(self.model.cp_trunk)
            except Exception as e:
                self.print(f"summary failed: {e}")

    def forward(self, knot_vector: torch.Tensor, norm_ics: torch.Tensor):
        return self.model(knot_vector, norm_ics)

    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        knot_vector = batch["knot_vector"]
        cp_embed = self.model.trunk_forward(knot_vector)
        end_points = _compute_end_points(knot_vector, self.spline_degree)

        total, id_only, sym_only = _trunk_loss(
            cp_embed,
            knot_vector,
            self.spline_degree,
            self.rollout_steps,
            end_points,
            sym_weight=self.sym_weight,
        )

        # Optional trajectory-reconstruction loss. The frozen branch is run to
        # produce ic_embed; cp_embed @ ic_embed @ B should match the true
        # (subsampled) trajectory. Gradients only flow into trunk parameters
        # since the branch is frozen.
        traj_loss = None
        if self.traj_weight > 0.0:
            norm_ics = batch["norm_ics"]
            norm_traj = batch["norm_trajectories"]
            with torch.no_grad():
                ic_embed = self.model.traj_model(norm_ics)
            traj_loss = _trajectory_loss(
                cp_embed,
                ic_embed,
                norm_traj,
                knot_vector,
                self.spline_degree,
                self.rollout_steps,
                end_points,
            )
            total = total + self.traj_weight * traj_loss

        on_step = stage == "train"
        self.log(
            f"{stage}/loss",
            total.detach(),
            prog_bar=True,
            on_epoch=True,
            on_step=on_step,
            sync_dist=True,
        )
        self.log(
            f"{stage}/identity_loss",
            id_only,
            prog_bar=False,
            on_epoch=True,
            on_step=on_step,
            sync_dist=True,
        )
        if self.sym_weight > 0.0:
            self.log(
                f"{stage}/sym_loss",
                sym_only,
                prog_bar=False,
                on_epoch=True,
                on_step=on_step,
                sync_dist=True,
            )
        if traj_loss is not None:
            self.log(
                f"{stage}/traj_loss",
                traj_loss.detach(),
                prog_bar=True,
                on_epoch=True,
                on_step=on_step,
                sync_dist=True,
            )
        return total

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        # Only the trunk has trainable params; the branch is frozen.
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=self.cfg.model.cp.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.cfg.trainer.scheduler.factor,
            patience=self.cfg.trainer.scheduler.patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.cfg.trainer.scheduler.monitor,
            },
        }
