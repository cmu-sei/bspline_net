"""
bspline_net (Github)

Copyright 2026 Carnegie Mellon University.

NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

Licensed under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.

[DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.

This Software includes and/or makes use of Third-Party Software each subject to its own license.

DM26-0750
"""
# src/projectm/callbacks.py
"""Visualization callbacks for training.

Helpers grouped by concern: knot, trajectory, CP. Per-model orchestrators
(`run_knot_plots`, `run_traj_plots`, `run_cp_plots`) fetch from a batch and
log to TensorBoard. Stage callbacks are thin wrappers.
"""

import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from pytorch_lightning.callbacks import Callback

from .utils import bspline_basis


def fig_to_tensor(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    img_array = np.array(Image.open(buf))
    plt.close(fig)
    return torch.from_numpy(img_array).permute(2, 0, 1)


def _log_fig(trainer, tag, fig):
    if trainer.logger:
        trainer.logger.experiment.add_image(tag, fig_to_tensor(fig), trainer.current_epoch)
    plt.close(fig)


# ---------- Knot ----------


def plot_spacing_overlay(ax, t_np, computed, predicted, title=None):
    ax.plot(t_np, computed, "b-", label="Computed", alpha=0.7)
    ax.plot(t_np, predicted, "r--", label="Predicted", alpha=0.7)
    ax.set_xlabel("t")
    ax.set_ylabel("Activity")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_knot_positions(ax, knots_np, title=None):
    ax.eventplot([knots_np], lineoffsets=0, linelengths=0.5, colors="r")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel("t")
    ax.set_ylabel(f"n={len(knots_np)}")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)


def run_knot_plots(
    trainer, pl_module, knot_predictor, norm_ics, jerk_activity_max, num_samples, tag_prefix="val"
):
    """knot_predictor: callable norm_ics -> (predicted_activity, knot_pos_pred)."""
    with torch.no_grad():
        predicted_activity, knot_pos_pred = knot_predictor(norm_ics)
    n = min(num_samples, predicted_activity.shape[0])
    t_np = np.linspace(0, 1, predicted_activity.shape[1])

    fig, axes = plt.subplots(n, 2, figsize=(12, 3 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    for i in range(n):
        plot_spacing_overlay(
            axes[i, 0],
            t_np,
            jerk_activity_max[i].cpu().numpy(),
            predicted_activity[i].cpu().numpy(),
            title=f"Sample {i+1}: Activity",
        )
        if i == 0:
            axes[i, 0].legend()
        knots = knot_pos_pred[i].cpu().numpy()
        knots = knots[knots < 1.0]
        plot_knot_positions(axes[i, 1], knots, title=f"Sample {i+1}: Knot Positions")
    plt.tight_layout()
    _log_fig(trainer, f"{tag_prefix}/spacing_and_knots", fig)


# ---------- Trajectory ----------


def plot_trajectory_overlay(
    pred,
    true,
    time_horizon,
    num_samples,
    dims_to_plot,
    axes=None,
    cp_positions=None,
    cp_values=None,
    extra_lines=None,
    bounds=None,
):
    """Grid of true (blue) vs predicted (red) overlays. If axes provided, draw onto
    them; else build a figure. cp_positions/cp_values: per-sample lists for green markers.

    extra_lines: optional list of (label, color, style, traj_array) tuples, where
    traj_array has shape (num_samples_shown, T, n_dim). Each tuple adds an extra
    line to every panel.
    """
    t_np = np.linspace(0, time_horizon, pred.shape[1])
    n = min(num_samples, pred.shape[0])
    own_fig = axes is None
    if own_fig:
        fig, axes = plt.subplots(n, len(dims_to_plot), figsize=(4 * len(dims_to_plot), 3 * n))
        if n == 1:
            axes = axes.reshape(1, -1)
    else:
        fig = None
    for i in range(n):
        for j, dim in enumerate(dims_to_plot):
            ax = axes[i, j]
            if extra_lines is not None:
                for label, color, style, traj in extra_lines:
                    ax.plot(
                        t_np,
                        traj[i, :, dim],
                        linestyle=style,
                        linewidth=3,
                        color=color,
                        alpha=0.7,
                        label=label,
                    )
            ax.plot(t_np, true[i, :, dim], "k-", label="True", alpha=0.90, linewidth=1.5)
            ax.plot(
                t_np,
                pred[i, :, dim],
                "--",
                label="Predicted",
                alpha=0.95,
                linewidth=1.5,
                color="red",
            )
            if bounds is not None:
                lo, hi = bounds
                ax.fill_between(
                    t_np,
                    lo[i, :, dim],
                    hi[i, :, dim],
                    color="blue",
                    alpha=0.15,
                    label="Interval",
                )

            if cp_positions is not None and cp_values is not None:
                ax.scatter(cp_positions[i], cp_values[i][:, dim], c="k", s=15, zorder=5, label="CP")
            ax.set_xlabel("Time [s]")
            if i == 0:
                ax.set_title(f"Dimension {dim}")
            if j == 0:
                ax.set_ylabel(f"Sample {i+1}\nDim {dim}")
            ax.grid(True, alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    if own_fig:
        plt.tight_layout()
    return fig


def plot_spectral_diagnostics(spec):

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=200)
    if not hasattr(spec, "log_lambda_imag"):
        return fig
    with torch.no_grad():
        lam_real = spec.log_lambda_real.cpu().numpy()
        lam_imag = spec.log_lambda_imag.cpu().numpy()
        M = torch.complex(spec.M_real, spec.M_imag)
        Q_abs = torch.matrix_exp(M - M.conj().T).abs().cpu().numpy()
        t_grid = spec.t.cpu().numpy()
        mode_mag = np.exp(t_grid[:, None] * lam_real[None, :])

    sc = axes[0].scatter(lam_real, lam_imag, c=np.arange(len(lam_real)), cmap="viridis", s=40)
    axes[0].axvline(0, color="k", lw=0.5, alpha=0.3)
    axes[0].set_xlabel("Re(lambda)  [decay rate]")
    axes[0].set_ylabel("Im(lambda)  [ang. freq]")
    axes[0].set_title("Eigenvalue spectrum")
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(sc, ax=axes[0], label="mode idx")

    for n in range(mode_mag.shape[1]):
        axes[1].plot(t_grid, mode_mag[:, n], alpha=0.7)
    axes[1].set_xlabel("t (rollout)")
    axes[1].set_ylabel("|lambda^t|")
    axes[1].set_title("Mode decay over rollout")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)

    im = axes[2].imshow(Q_abs, aspect="auto", cmap="magma")
    axes[2].set_xlabel("eigenvector idx")
    axes[2].set_ylabel("lifted dim")
    axes[2].set_title("|Q|  (unitary basis)")
    plt.colorbar(im, ax=axes[2])
    plt.tight_layout()
    return fig


def plot_correction_vs_error_hist2d(correction, pred, true, dims_to_plot):
    """2D log-log hist of |correction| vs |error|, per (sample, dim)."""
    eps = 1e-8
    log_corr = np.log10(np.sqrt((correction**2).sum(axis=1)) + eps)
    log_err = np.log10(np.sqrt(((pred - true) ** 2).sum(axis=1)) + eps)

    n_cols = min(4, len(dims_to_plot))
    n_rows = (len(dims_to_plot) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
    flat = axes.flatten()

    x_lo, x_hi = np.percentile(log_corr[:, dims_to_plot].ravel(), [1, 99])
    y_lo, y_hi = np.percentile(log_err[:, dims_to_plot].ravel(), [1, 99])
    bins = [np.linspace(x_lo, x_hi, 40), np.linspace(y_lo, y_hi, 40)]

    for k, dim in enumerate(dims_to_plot):
        ax = flat[k]
        h = ax.hist2d(log_corr[:, dim], log_err[:, dim], bins=bins, cmap="viridis")
        ax.set_xlabel("log10 |correction|")
        ax.set_ylabel("log10 |error|")
        ax.set_title(f"Dim {dim}")
        lo, hi = min(x_lo, y_lo), max(x_hi, y_hi)
        ax.plot([lo, hi], [lo, hi], "r--", alpha=0.5, linewidth=1)
        plt.colorbar(h[3], ax=ax)
    for k in range(len(dims_to_plot), len(flat)):
        flat[k].axis("off")
    plt.tight_layout()
    return fig


def run_traj_plots(
    trainer,
    pl_module,
    traj_model,
    norm_ics,
    norm_trajectories,
    time_horizon,
    num_samples,
    dims_to_plot,
    tag_prefix="val",
    skip_overlay=False,
):
    """Trajectory overlay (unless skipped) + spectral + correction-vs-error."""
    with torch.no_grad():
        traj_out = traj_model(norm_ics, return_correction=True)

    if isinstance(traj_out, dict):
        # Quantile mode: dict has lo/med/hi (and maybe correction).
        ic_embed = traj_out["med"]
        correction = traj_out.get("correction")
        lo_np = traj_out["lo"].cpu().numpy()
        hi_np = traj_out["hi"].cpu().numpy()
        bounds = (lo_np, hi_np)
    else:
        # Point mode (existing tuple return).
        ic_embed, correction = traj_out
        bounds = None

    idx = (
        torch.linspace(0, norm_trajectories.shape[1] - 1, ic_embed.shape[1], device=ic_embed.device)
        .round()
        .long()
    )
    pred_np = ic_embed.cpu().numpy()
    true_np = norm_trajectories[:, idx, :].cpu().numpy()
    correction_np = correction.cpu().numpy() if correction is not None else None

    if not skip_overlay:
        _log_fig(
            trainer,
            f"{tag_prefix}/trajectories",
            plot_trajectory_overlay(
                pred_np,
                true_np,
                time_horizon,
                num_samples,
                dims_to_plot,
                bounds=bounds,
            ),
        )
    _log_fig(
        trainer, f"{tag_prefix}/spectral_deco", plot_spectral_diagnostics(traj_model.ic_rollout)
    )
    if traj_model.decoder is not None and correction_np is not None:
        fig = plot_correction_vs_error_hist2d(correction_np, pred_np, true_np, dims_to_plot)
        if fig is not None:
            _log_fig(trainer, f"{tag_prefix}/correction_vs_error", fig)


# ---------- CP ----------


def plot_embeddings(cp_embed, ic_embed):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=300)
    axes = axes.flatten()
    axes[0].imshow(cp_embed[0, :, :].cpu().detach().numpy())
    axes[1].imshow(ic_embed[0, :, :].cpu().detach().numpy().T)
    axes[2].imshow(cp_embed[:, 1, :].cpu().detach().numpy().T, aspect="auto")
    axes[3].imshow(cp_embed[:, :, 1].cpu().detach().numpy().T, aspect="auto")
    axes[0].set_title(f"CP Embed {cp_embed[0].shape}")
    axes[1].set_title(f"IC Embed {ic_embed[0].shape}")
    plt.tight_layout()
    return fig


def plot_pinv_residual_histogram(cp_embed, knot_vector, spline_degree, rollout_steps):
    """Per-sample histogram of log10(||cp_embed @ B - I||_F / n_cp), which is
    the trunk's pseudoinverse residual (the trunk's training objective in
    normalized form). Smaller = trunk is closer to a perfect left-inverse of B.

    Overlays the closed-form pinv's residual on the same axes for reference.
    """
    d = spline_degree
    device = cp_embed.device
    batch = knot_vector.shape[0]
    t_rollout = torch.linspace(0, 1, rollout_steps, device=device)

    trunk_resids, pinv_resids = [], []
    for i in range(batch):
        kv = knot_vector[i]
        eq_ones = (kv == 1).nonzero(as_tuple=False).flatten()
        end_point = len(kv) if eq_ones.numel() <= d + 1 else int(eq_ones[d + 1].item())
        n_cp_i = end_point - d - 1
        if n_cp_i <= 0:
            continue

        B_i = bspline_basis(t_rollout, kv[:end_point], d)  # (rollout_steps, n_cp_i)
        I_i = torch.eye(n_cp_i, device=device)

        # Trunk's residual
        cp_i = cp_embed[i, :n_cp_i, :]  # (n_cp_i, rollout_steps)
        trunk_resid = (cp_i @ B_i - I_i).norm() / n_cp_i
        trunk_resids.append(trunk_resid.item())

        # Closed-form pinv residual (sanity check; should be near machine epsilon)
        BtB = B_i.T @ B_i
        B_pinv = torch.linalg.solve(BtB + 1e-6 * I_i, B_i.T)
        pinv_resid = (B_pinv @ B_i - I_i).norm() / n_cp_i
        pinv_resids.append(pinv_resid.item())

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    eps = 1e-12
    log_trunk = np.log10(np.array(trunk_resids) + eps)
    log_pinv = np.log10(np.array(pinv_resids) + eps)

    lo = min(log_trunk.min(), log_pinv.min())
    hi = max(log_trunk.max(), log_pinv.max())
    bins = np.linspace(lo - 0.1, hi + 0.1, 40)
    ax.hist(log_trunk, bins=bins, alpha=0.6, color="C0", label="Trunk")
    ax.hist(log_pinv, bins=bins, alpha=0.6, color="C1", label="Pinv (closed-form)")
    ax.set_xlabel("log10 ||cp_embed @ B - I||_F / n_cp")
    ax.set_ylabel("Count")
    ax.set_title("Per-sample pseudoinverse residual")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def _closed_form_cp_and_spline(knots_row, end_point, d, source_signal_rollout, t_full, lam=1e-6):
    """Compute closed-form CPs from a per-rollout-slot signal, then evaluate the
    resulting spline at the full sample grid.

    knots_row: 1D tensor of knot positions (already trimmed at end_point).
    end_point: int, length of valid portion of knots_row.
    d: spline degree.
    source_signal_rollout: (rollout_steps, n_dim) - signal we're fitting CPs to.
    t_full: 1D tensor of full sample times in [0, 1].
    lam: Tikhonov regularization for stability.

    Returns: (cp_closed: (n_cps, n_dim), spline_full: (n_samples, n_dim)).
    """
    n_cps = end_point - d - 1
    rollout_steps = source_signal_rollout.shape[0]
    t_rollout = torch.linspace(0, 1, rollout_steps, device=knots_row.device)
    B_rollout = bspline_basis(t_rollout, knots_row[:end_point], d)  # (rollout_steps, n_cps)
    # Regularized least-squares: solve (B^T B + lam I) cp = B^T signal
    BtB = B_rollout.T @ B_rollout
    Btf = B_rollout.T @ source_signal_rollout
    I = torch.eye(n_cps, device=knots_row.device)
    cp_closed = torch.linalg.solve(BtB + lam * I, Btf)  # (n_cps, n_dim)
    # Evaluate spline at full grid
    B_full = bspline_basis(t_full, knots_row[:end_point], d)
    spline_full = B_full @ cp_closed  # (n_samples, n_dim)
    return cp_closed, spline_full


def run_cp_plots(
    trainer,
    pl_module,
    cp_model,
    knot_vector,
    norm_ics,
    ics,
    trajectories,
    time_horizon,
    spline_degree,
    num_samples,
    dims_to_plot,
    tag_prefix="val",
):
    """Embeddings, IC residual hist, trajectory reconstruction with CP markers.

    Also overlays two closed-form spline reconstructions:
      - navy: CPs from pinv(B) @ ic_embed (the branch's predicted trajectory).
              Tells us whether the trunk is needed at all.
      - magenta: CPs from pinv(B) @ true_trajectory_at_rollout.
                 Diagnostic of pure basis inversion quality.
    """
    d = spline_degree
    with torch.no_grad():
        pred_cp, _, cp_embed, ic_embed = cp_model(knot_vector, norm_ics)

    _log_fig(trainer, f"{tag_prefix}/embeddings", plot_embeddings(cp_embed, ic_embed))
    _log_fig(
        trainer,
        f"{tag_prefix}/pinv_residual",
        plot_pinv_residual_histogram(cp_embed, knot_vector, spline_degree, ic_embed.shape[1]),
    )
    n = min(num_samples, trajectories.shape[0])
    n_samples_data = trajectories.shape[1]
    t_full = torch.linspace(0, 1, n_samples_data, device=trajectories.device)

    # Subsample truth at rollout slots so we can compute closed-form CPs from it.
    rollout_steps = ic_embed.shape[1]
    rollout_idx = (
        torch.linspace(0, n_samples_data - 1, rollout_steps, device=ic_embed.device).round().long()
    )
    norm_trajectories = ic_embed.new_zeros(0)  # placeholder; resolved below
    # We need norm_trajectories for the true-rollout signal; reconstruct it from
    # trajectories / R if not already in scope. trajectories is unnormalized.
    R = cp_model.R if hasattr(cp_model, "R") else pl_module.R
    norm_trajectories_at_rollout = (
        trajectories[:, rollout_idx, :] / R
    )  # (batch, rollout_steps, n_dim)

    cp_positions, cp_values, pred_traj_list = [], [], []
    closed_ic_list, closed_true_list = [], []
    for i in range(n):
        kv = knot_vector[i]
        end_knots = torch.where(kv == 1)[0]
        end_point = len(kv) if len(end_knots) <= d + 1 else end_knots[d + 1].item()
        n_cps = end_point - d - 1
        kv_np = kv[:end_point].cpu().numpy()
        cp_t = np.array([kv_np[ii + 1 : ii + d + 1].mean() for ii in range(n_cps)]) * time_horizon

        # Existing trunk-based reconstruction
        B = bspline_basis(t_full, kv[:end_point], d)
        pred_traj_list.append((B @ pred_cp[i, :n_cps]).cpu().numpy())
        cp_positions.append(cp_t)
        cp_values.append(pred_cp[i, :n_cps].cpu().numpy())

        # Closed-form from ic_embed (branch's predicted trajectory)
        _, spline_from_ic = _closed_form_cp_and_spline(kv, end_point, d, ic_embed[i], t_full)
        closed_ic_list.append((spline_from_ic * R).cpu().numpy())

        # Closed-form from true trajectory at rollout slots
        _, spline_from_true = _closed_form_cp_and_spline(
            kv, end_point, d, norm_trajectories_at_rollout[i], t_full
        )
        closed_true_list.append((spline_from_true * R).cpu().numpy())

    pred_traj_np = np.stack(pred_traj_list, axis=0)
    true_traj_np = trajectories[:n].cpu().numpy()
    closed_ic_np = np.stack(closed_ic_list, axis=0)
    closed_true_np = np.stack(closed_true_list, axis=0)

    extra_lines = [
        ("pinv(B) ic_embed", "royalblue", "--", closed_ic_np),
        ("pinv(B) true", "lime", "--", closed_true_np),
    ]

    _log_fig(
        trainer,
        f"{tag_prefix}/trajectories",
        plot_trajectory_overlay(
            pred_traj_np,
            true_traj_np,
            time_horizon,
            n,
            dims_to_plot,
            cp_positions=cp_positions,
            cp_values=cp_values,
            extra_lines=extra_lines,
        ),
    )


# ---------- Stage callbacks ----------


def _val_batch(trainer, pl_module):
    batch = next(iter(trainer.datamodule.val_dataloader()))
    return {
        k: v.to(pl_module.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }


class KnotVisualizationCallback(Callback):
    def __init__(self, every_n_epochs=5, num_samples=4):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.num_samples = num_samples

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        batch = _val_batch(trainer, pl_module)

        def predictor(norm_ics):
            pa, kp, _ = pl_module.predict_knots(norm_ics)
            return pa, kp

        run_knot_plots(
            trainer,
            pl_module,
            predictor,
            batch["norm_ics"],
            batch["jerk_activity_max"],
            self.num_samples,
        )


class TrajVisualizationCallback(Callback):
    def __init__(
        self, every_n_epochs=5, num_samples=4, dims_to_plot=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    ):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.num_samples = num_samples
        self.dims_to_plot = dims_to_plot

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        batch = _val_batch(trainer, pl_module)

        run_traj_plots(
            trainer,
            pl_module,
            pl_module.model,
            batch["norm_ics"],
            batch["norm_trajectories"],
            pl_module.cfg.data.time_horizon,
            self.num_samples,
            self.dims_to_plot,
        )


class CPVisualizationCallback(Callback):
    def __init__(
        self, every_n_epochs=5, num_samples=4, dims_to_plot=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    ):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.num_samples = num_samples
        self.dims_to_plot = dims_to_plot

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        batch = _val_batch(trainer, pl_module)
        cfg = pl_module.cfg
        run_cp_plots(
            trainer,
            pl_module,
            pl_module.model,
            batch["knot_vector"],
            batch["norm_ics"],
            batch["ics"],
            batch["trajectories"],
            cfg.data.time_horizon,
            cfg.data.spline_degree,
            self.num_samples,
            self.dims_to_plot,
        )


class JointVisualizationCallback(Callback):
    def __init__(
        self, every_n_epochs=5, num_samples=4, dims_to_plot=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    ):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.num_samples = num_samples
        self.dims_to_plot = dims_to_plot

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        batch = _val_batch(trainer, pl_module)
        cfg = pl_module.cfg
        joint = pl_module.model

        def knot_predictor(norm_ics):
            with torch.no_grad():
                pa, kp, _ = joint.knot_model.predict_knots(norm_ics)
            return pa, kp

        run_knot_plots(
            trainer,
            pl_module,
            knot_predictor,
            batch["norm_ics"],
            batch["jerk_activity_max"],
            self.num_samples,
        )

        run_traj_plots(
            trainer,
            pl_module,
            joint.cp_model.traj_model,
            batch["norm_ics"],
            batch["norm_trajectories"],
            cfg.data.time_horizon,
            self.num_samples,
            self.dims_to_plot,
            skip_overlay=True,
        )

        # CP plots use the knot vector produced by the joint forward.
        with torch.no_grad():
            _, knot_vector, _, _ = pl_module(batch["norm_ics"])
        run_cp_plots(
            trainer,
            pl_module,
            joint.cp_model,
            knot_vector,
            batch["norm_ics"],
            batch["ics"],
            batch["trajectories"],
            cfg.data.time_horizon,
            cfg.data.spline_degree,
            self.num_samples,
            self.dims_to_plot,
        )


# ---------- Misc ----------


class MetricsSyncCallback(Callback):
    def __init__(self, path):
        self.path = Path(path)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            metrics = {
                k: v.item() if hasattr(v, "item") else v
                for k, v in trainer.callback_metrics.items()
            }
            self.path.write_text(json.dumps(metrics))


def get_visualization_callback(stage, every_n_epochs=5):
    cls = {
        "knots": KnotVisualizationCallback,
        "traj": TrajVisualizationCallback,
        "cp": CPVisualizationCallback,
        "joint": JointVisualizationCallback,
    }.get(stage)
    if cls is None:
        raise ValueError(f"Unknown stage: {stage}")
    return cls(every_n_epochs=every_n_epochs)
