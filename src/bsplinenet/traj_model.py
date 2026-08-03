"""
bspline_net (Github)

Copyright 2026 Carnegie Mellon University.

NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

Licensed under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.

[DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.

This Software includes and/or makes use of Third-Party Software each subject to its own license.

DM26-0750
"""
"""Standalone trajectory model training."""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torchinfo import summary

from .utils import compute_trajectory_loss_terms

# Swap out SpectralDeco for a recurrent neural network of approximately thesame size (based on N and T)
# This enables us to discern the importance of the network architecture to performance.
# designed for same inputs and outputs for ease of use.
import torch
import torch.nn as nn


class BaselineRnn(nn.Module):
    """
    Real-valued recurrent replacement for SpectralDeco, matched interface
    and matched parameter count (~33K, vs SpectralDeco's 33,024 at N=128):
        input:  phi, shape (B, N=128), real   -- the lifted state
        output: prediction, shape (B, 2N, T), real

    hidden_dim=72 was solved to bring total params (input_proj + cell + decoder)
    as close as possible to SpectralDeco at nominal (128 lifted dimensions)
    """

    def __init__(self, N=128, T=None, epsilon_min: float = 0.01, hidden_dim: int = 72):
        super().__init__()
        self.N = N
        self.T = T
        self.epsilon_min = epsilon_min  
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(N, hidden_dim)
        self.cell = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, 2 * N)
        self.activation = nn.Tanh()
        nn.init.eye_(self.cell.weight)
        self.cell.weight.data *= 0.9
        nn.init.zeros_(self.cell.bias)

    def forward(self, phi):
        # phi: (B, N), real
        h = self.activation(self.input_proj(phi))  # (B, hidden_dim)

        outputs = []
        for _ in range(self.T):
            h = self.activation(self.cell(h))       # (B, hidden_dim)
            outputs.append(self.decoder(h))          # (B, 2N)

        prediction = torch.stack(outputs, dim=-1)    # (B, 2N, T)
        return prediction
        
        
    
class SpectralDeco(nn.Module):
    def __init__(self, N, T, epsilon_min: float = 0.01):
        super().__init__()
        self.N = N
        self.T = T
        self.epsilon_min = epsilon_min

        # Initialize raw so effective Re(lambda) = linspace(-2, -0.05).
        # Requires epsilon_min < 0.05 so the softcap is well-defined at init.
        if epsilon_min >= 0.05:
            raise ValueError(
                f"spectral_epsilon_min={epsilon_min} must be < 0.05 to match "
                f"original eigenvalue init range [-0.5, -0.05]."
            )
        # torch.manual_seed(0)
        s_imag = torch.linspace(1, 0, N)
        envelope = torch.sin(torch.pi * s_imag)
        jitter = 0.30 * envelope * torch.randn(N)
        s_real = torch.clip(s_imag + jitter, 0, 1)  # still clip just in case
        target_real_init = -(epsilon_min + s_real * (30.0 - epsilon_min))

        raw_init = -epsilon_min - torch.log(torch.expm1(-(target_real_init + epsilon_min)))
        self.log_lambda_real_raw = nn.Parameter(raw_init)
        self.log_lambda_imag = nn.Parameter(s_imag * 25 * torch.pi)

        # Skew-Hermitian generator S of the unitary Q = matrix_exp(S).
        # Parameterized by an unconstrained N x N complex matrix M; we form
        # S = M - M^H in forward, which is exactly the skew-Hermitian projection.
        # This gives N^2 real degrees of freedom with no gauge redundancy.
        # Small init -> Q near identity, so phi passes through roughly unchanged at start.
        self.M_real = nn.Parameter(torch.randn(N, N) * 0.1)
        self.M_imag = nn.Parameter(torch.randn(N, N) * 0.1)

        self.register_buffer("t", torch.linspace(0, 1, T))

    @property
    def log_lambda_real(self) -> torch.Tensor:
        return -self.epsilon_min - F.softplus(-(self.log_lambda_real_raw + self.epsilon_min))

    def forward(self, phi):
        # phi: (B, N), real

        # Build unitary Q via matrix exponential of skew-Hermitian S.
        M = torch.complex(self.M_real, self.M_imag)
        S = M - M.conj().T  # skew-Hermitian
        Q = torch.matrix_exp(S)  # exactly unitary

        # Project phi into eigenbasis. Q is unitary so Q^{-1} = Q^H.
        phi_c = phi.to(torch.cfloat)
        coeffs = phi_c @ Q.conj()  # (B, N), complex

        # lambda^t: (T, N), complex
        log_lambda = torch.complex(self.log_lambda_real, self.log_lambda_imag)
        lambda_t = torch.exp(self.t.unsqueeze(1) * log_lambda.unsqueeze(0))

        # Evolve each mode: (B, N) and (T, N) -> (B, N, T)
        evolved = coeffs.unsqueeze(-1) * lambda_t.T.unsqueeze(0)

        # Apply Q to map back to lifted space.
        out = torch.einsum("ij,bjt->bit", Q, evolved)  # (B, N, T), complex
        return torch.cat([out.real, out.imag], dim=1)  # (B, 2N, T), real


class TrajectoryModel(nn.Module):
    """Predicts trajectories from initial conditions via lift -> spectral
    rollout -> scheduled, anchored decoder.

    The state is structurally embedded in the first n_dim lifted components,
    so the readout `lifted[:, :, :n_dim]` exactly preserves the IC at t=0.
    The decoder operates on the non-state portion and is anchored to produce
    zero correction at t=0, then scaled by a non-learnable `alpha` buffer that
    is externally scheduled (see LitTrajModel). Decoder weights are also
    initialized very small.

    Args:
        n_dim: State space dimension.
        widths_ic: Hidden layer widths for IC encoder (use [] for pure linear lift).
        rollout_steps: Number of virtual time slots in the rollout.
        lifted_dim: Lifted space dimension (must exceed n_dim).
        spectral_epsilon_min: Minimum decay rate for the spectral op.
        use_decoder: Whether to include the learned decoder correction.
    """

    def __init__(
        self,
        n_dim: int,
        widths_ic: list[int],
        rollout_steps: int,
        lifted_dim: int,
        spectral_epsilon_min: float = 0.01,
        use_decoder: bool = True,
        quantile_levels: list[float] | None = None,
        use_rnn: bool = False,
    ):
        super().__init__()

        assert lifted_dim > n_dim, (
            f"lifted_dim ({lifted_dim}) must exceed n_dim ({n_dim}) to leave "
            f"room for extra observables"
        )
        self.n_dim = n_dim
        self.lifted_dim = lifted_dim
        n_extra = lifted_dim - n_dim

        ic_lift = []
        prev_dim = n_dim
        for width in widths_ic:
            ic_lift.extend([nn.Linear(prev_dim, width), nn.SiLU()])
            prev_dim = width
        ic_lift.append(nn.Linear(prev_dim, n_extra))
        self.ic_lift = nn.Sequential(*ic_lift)

        if use_rnn:
            self.ic_rollout = BaselineRnn(lifted_dim, rollout_steps, epsilon_min=spectral_epsilon_min)
            self.use_rnn = True
        else:
            self.ic_rollout = SpectralDeco(lifted_dim, rollout_steps, epsilon_min=spectral_epsilon_min)
            self.use_rnn = False

        self.quantile_levels = quantile_levels  # e.g. [0.025, 0.975] or None
        if quantile_levels is None:
            # Backward-compatible point predictor: single decoder.
            if use_decoder:
                self.decoder = nn.Linear(2 * lifted_dim - n_dim, n_dim, bias=False)
                self.register_buffer("alpha", torch.tensor(0.0))
            else:
                self.decoder = None
                self.register_buffer("alpha", torch.tensor(0.0))
            self.decoder_lo = None
            self.decoder_hi = None
        else:
            # Quantile predictor: median decoder + two non-negative offset decoders.
            assert use_decoder, "quantile mode requires use_decoder=True"
            assert len(quantile_levels) == 2, (
                "quantile_levels must be a 2-list [low, high], " f"got {quantile_levels}"
            )
            self.decoder = nn.Linear(2 * lifted_dim - n_dim, n_dim, bias=False)  # median
            self.decoder_lo = nn.Linear(2 * lifted_dim - n_dim, n_dim, bias=False)  # low offset
            self.decoder_hi = nn.Linear(2 * lifted_dim - n_dim, n_dim, bias=False)  # high offset
            self.register_buffer("alpha", torch.tensor(0.0))

        self._init_weights()

    def _init_weights(self):

        for module in self.ic_lift:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
        for dec in (self.decoder, self.decoder_lo, self.decoder_hi):
            if dec is not None:
                nn.init.normal_(dec.weight, std=1e-3)

    def set_alpha(self, value: float):
        self.alpha.fill_(float(value))

    def forward(self, norm_ics: torch.Tensor, return_correction: bool = False):
        """Returns ic_embed of shape (batch, rollout_steps, n_dim)."""
        extras = self.ic_lift(norm_ics)  # (batch, n_extra)
        phi = torch.cat([norm_ics, extras], dim=-1)  # (batch, lifted_dim)
        rolled = self.ic_rollout(phi)  # (batch, 2*lifted_dim, rollout_steps)
        lifted = rolled.permute(0, 2, 1)  # (batch, rollout_steps, 2*lifted_dim)
        ic_embed = lifted[:, :, : self.n_dim]
        decoder_input = lifted[:, :, self.n_dim :].detach()

        if self.decoder is None:
            correction = torch.zeros_like(ic_embed)
            median = ic_embed
        else:
            correction = self.decoder(decoder_input)
            correction = correction - correction[:, :1, :]  # zero at t=0
            correction = self.alpha * correction  # scheduled gate
            median = ic_embed + correction
        if self.quantile_levels is None:
            out = median
        else:
            lo_raw = self.decoder_lo(decoder_input)
            lo = median - self.alpha * (F.relu(lo_raw - lo_raw[:, :1, :]))

            hi_raw = self.decoder_hi(decoder_input)
            hi = median + self.alpha * (F.relu(hi_raw - hi_raw[:, :1, :]))

            out = {"lo": lo, "med": median, "hi": hi}

        if return_correction:
            if isinstance(out, dict):
                out["correction"] = correction
            else:
                return out, correction
        return out


class LitTrajModel(pl.LightningModule):
    """Trains the TrajectoryModel in isolation against trajectory MSE."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        traj_cfg = cfg.model.traj
        self.model = TrajectoryModel(
            n_dim=len(cfg.data.R),
            widths_ic=traj_cfg.widths_ic,
            rollout_steps=traj_cfg.rollout_steps,
            lifted_dim=traj_cfg.lifted_dim,
            spectral_epsilon_min=traj_cfg.get("spectral_epsilon_min", 0.01),
            use_decoder=traj_cfg.get("use_decoder", True),
            quantile_levels=traj_cfg.get("quantile_levels", None),
            use_rnn = traj_cfg.get("use_rnn", False)
        )
        self.use_rnn = traj_cfg.get("use_rnn", False)
        self.warmup_start = traj_cfg.get("alpha_warmup_start_epoch", 50)
        self.warmup_end = traj_cfg.get("alpha_warmup_end_epoch", 150)
        self.integral_weight = float(traj_cfg.get("integral_weight", 0.0))

    def setup(self, stage=None):
        if self.global_rank == 0:
            try:
                n_dim = len(self.cfg.data.R)
                self.print(summary(self, input_size=(2, n_dim), verbose=0))
            except Exception as e:
                self.print(f"torchinfo summary failed: {e}")

    def _alpha_schedule(self, epoch: int) -> float:
        if epoch < self.warmup_start:
            return 0.0
        if epoch >= self.warmup_end:
            return 1.0
        span = max(1, self.warmup_end - self.warmup_start)
        return (epoch - self.warmup_start) / span

    def on_train_epoch_start(self):
        self.model.set_alpha(self._alpha_schedule(self.current_epoch))

    def forward(self, norm_ics: torch.Tensor) -> torch.Tensor:
        return self.model(norm_ics)

    def on_after_backward(self):
        if self.global_step % 50 != 0:
            return
        spec = self.model.ic_rollout
        if self.use_rnn:
            grads = {}
        else:
            grads = {
                "grad/lam_real_raw": spec.log_lambda_real_raw.grad,
                "grad/lam_imag": spec.log_lambda_imag.grad,
                "grad/M_real": spec.M_real.grad,
                "grad/M_imag": spec.M_imag.grad,
                "grad/ic_lift_first": next(self.model.ic_lift.parameters()).grad,
            }
        if self.model.decoder is not None:
            grads["grad/decoder"] = self.model.decoder.weight.grad
        for name, g in grads.items():
            if g is not None:
                self.log(name, g.abs().mean(), on_step=True, on_epoch=False)

    def _trajectory_loss(self, ic_embed, norm_trajectories):
        n_samples = norm_trajectories.shape[1]
        rollout_steps = ic_embed.shape[1]
        idx = torch.linspace(0, n_samples - 1, rollout_steps, device=ic_embed.device).round().long()
        target = norm_trajectories[:, idx, :]
        return compute_trajectory_loss_terms(ic_embed, target)

    def _pinball_loss(self, pred: torch.Tensor, target: torch.Tensor, tau: float) -> torch.Tensor:
        diff = target - pred
        return torch.maximum(tau * diff, (tau - 1) * diff).mean()

    def _integral_loss(self, mean_pred):
        cfg = self.cfg.data
        R = torch.tensor(cfg.R, device=mean_pred.device)
        pos_i = torch.tensor([1, 3, 5, 7, 9, 11], device=mean_pred.device)
        vel_i = torch.tensor([0, 2, 4, 6, 8, 10], device=mean_pred.device)
        T = mean_pred.shape[1]
        tt = torch.linspace(0.0, cfg.time_horizon, cfg.n_samples, device=mean_pred.device)
        idx = torch.linspace(0, cfg.n_samples - 1, T, device=mean_pred.device).round().long()
        t_phys = tt[idx]
        dt = (t_phys[1:] - t_phys[:-1]).view(1, -1, 1)
        pos = mean_pred[:, :, pos_i] * R[pos_i]
        vel = mean_pred[:, :, vel_i] * R[vel_i]
        incr = pos[:, 1:, :] - pos[:, :-1, :]
        integ = 0.5 * (vel[:, 1:, :] + vel[:, :-1, :]) * dt
        return (((incr - integ) / R[pos_i]) ** 2).mean()

    def _log(self, stage, name, value, on_step, prog_bar=False):
        self.log(
            f"{stage}/{name}",
            value.detach() if torch.is_tensor(value) else value,
            prog_bar=prog_bar,
            on_epoch=True,
            on_step=on_step,
            sync_dist=True,
        )

    def _shared_step(self, batch, stage):
        norm_ics = batch["norm_ics"]
        norm_trajectories = batch["norm_trajectories"]
        out = self(norm_ics)
        is_quantile = isinstance(out, dict)
        mean_pred = out["med"] if is_quantile else out
        on_step = stage == "train"

        # Point (trajectory) loss on the mean prediction.
        traj_loss, traj_loss_norm = self._trajectory_loss(mean_pred, norm_trajectories)
        loss = traj_loss + traj_loss_norm

        # Always compute the consistency residual for monitoring; only fold it
        # into the loss when integral_weight > 0.
        integral_loss = self._integral_loss(mean_pred)
        if self.integral_weight > 0:
            loss = loss + self.integral_weight * self.model.alpha * integral_loss

        # Quantile (pinball) losses, only in quantile mode.
        if is_quantile:
            n_samples = norm_trajectories.shape[1]
            idx = (
                torch.linspace(0, n_samples - 1, mean_pred.shape[1], device=mean_pred.device)
                .round()
                .long()
            )
            target = norm_trajectories[:, idx, :]
            qlo, qhi = self.model.quantile_levels
            loss_lo = self._pinball_loss(out["lo"], target, qlo)
            loss_hi = self._pinball_loss(out["hi"], target, qhi)
            loss = loss + loss_lo + loss_hi
            self._log(stage, "low_loss", loss_lo, on_step)
            self._log(stage, "high_loss", loss_hi, on_step)

        self._log(stage, "loss", loss, on_step, prog_bar=True)
        self._log(stage, "traj_loss", traj_loss, on_step)
        self._log(stage, "traj_loss_norm", traj_loss_norm, on_step)
        self._log(stage, "integral_loss", integral_loss, on_step)
        if self.model.decoder is not None:
            self._log(stage, "alpha", self.model.alpha, on_step)
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        spec = self.model.ic_rollout
        spec_params = list(spec.parameters())
        spec_ids = {id(p) for p in spec_params}
        other_params = [p for p in self.parameters() if id(p) not in spec_ids]
        base_lr = self.cfg.model.traj.lr
        spec_lr_mult = self.cfg.model.traj.get("spec_lr_mult", 10)
        optimizer = torch.optim.Adam(
            [
                {"params": other_params, "lr": base_lr},
                {"params": spec_params, "lr": base_lr * spec_lr_mult},
            ]
        )
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
