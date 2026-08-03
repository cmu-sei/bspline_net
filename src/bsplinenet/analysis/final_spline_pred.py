"""
bspline_net (Github)

Copyright 2026 Carnegie Mellon University.

NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

Licensed under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.

[DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.

This Software includes and/or makes use of Third-Party Software each subject to its own license.

DM26-0750
"""
"""
final_spline_pred.py
---------------------
Generate control-point-level predictions and visualisations for a trained
trajectory-quantile model.

This script is an entry point for the *final spline* analysis step. It
* loads a saved quantile trajectory model and a knot-placement model,
* samples a single initial-condition (IC) from the test split,
* computes the model's predictive bands,
* widens those bands with the conformal correction defined in the
  ``final_spline.yaml`` configuration,
* projects the trajectory bands onto the spline's control-point basis,
* and finally produces a multi-panel plot showing:
  - the ground-truth trajectory,
  - the mean prediction,
  - the conformal prediction band,
  - the control-point estimates together with per-point confidence
    intervals.

The script is executed via Hydra, e.g.:

```bash
python -m bsplinenet.analysis.final_spline_pred \
    +seed=42 \
    checkpoint_path=path/to/trajectory.ckpt \
    knot_model_path=path/to/knot.ckpt
```
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import hydra
from omegaconf import DictConfig

from .traj_conformal_analysis import (
    band_residual,
    calibrate_joint_per_dim,
    collect_predictions,
    estimate_dim_weights,
    load_model,
)

from ..data import KnotDataModule, compute_derived_data_config, get_datamodule
from ..knot_model import ActivityModel, LitKnotModel
from ..traj_model import LitTrajModel
from ..utils import bspline_basis


def load_trajectory_model(checkpoint_path, device):
    """Load the quantile trajectory model. Returns (model, train_cfg, qlevels)."""
    model, cfg, q_levels = load_model(checkpoint_path)
    model = model.to(device)
    return model, cfg, q_levels


def load_knot_model(checkpoint_path, device):
    """Load the activity/knot model. Returns the model in eval mode."""
    # --- Load model from checkpoint ---
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = LitKnotModel(**checkpoint["hyper_parameters"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.freeze()

    model = model.to(device)
    return model


def sample_ic(cfg, seed, device):
    datamodule = get_datamodule("knots", cfg)
    test_loader = datamodule.test_dataloader()

    for b, batch in enumerate(test_loader):
        return batch


def predict_trajectory_bands(model, norm_ics):
    """Run the trajectory model on the IC. Returns (lo, med, hi) each (T, n_dim)
    at the model's rollout times (with the t=0 anchor dropped)."""
    with torch.no_grad():
        out = model(norm_ics)
    return out


def apply_conformal(lo, hi, Q, R, w):
    """Apply per-dim conformal widening. Returns (lo_cal, hi_cal) each (T, n_dim).
    Widening per dim is +/- Q * R[d] * w[d]."""

    return (lo - Q * R * w, hi + Q * R * w)

# unused
def control_point_intervals(B_dag, lo, hi):
    """Compute (c_lo, c_hi) for each control point and dim via the pointwise bound.
    B_dag shape (K, T); lo, hi shape (T, n_dim).
    Returns (c_lo, c_hi) each shape (K, n_dim)."""
    mid  = 0.5 * (hi + lo)
    half = 0.5 * (hi - lo)
    cp_med  = B_dag @ mid
    cp_diff = torch.abs(B_dag) @ half
    return cp_med - cp_diff, cp_med + cp_diff



def plot_per_dim(
    t_phys, true_traj, mean, lo_cal, hi_cal, knot_vector, cp_mean, cp_lo, cp_hi, names, path, degree
):
    """3-column grid per dimension. Each subplot:
    - truth (dimgray)
    - mean prediction (red dashed)
    - conformal trajectory band (blue fill, dotted edges)
    - control points (goldenrod) with vertical error bars (c_lo to c_hi) at Greville abscissae."""

    def _np(x):
        return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

    t_phys = _np(t_phys)
    true_traj = _np(true_traj)
    mean = _np(mean)
    lo_cal = _np(lo_cal)
    hi_cal = _np(hi_cal)
    knot_vector = _np(knot_vector).ravel()
    cp_mean = _np(cp_mean)
    cp_lo = _np(cp_lo)
    cp_hi = _np(cp_hi)
    n_dim = true_traj.shape[-1]
    K = cp_mean.shape[0]
    t0, t1 = float(t_phys[0]), float(t_phys[-1])
    knot_phys = t0 + knot_vector * (t1 - t0)
    if len(knot_phys) >= K + degree + 1:
        cp_phys = np.array([knot_phys[i + 1 : i + 1 + degree].mean() for i in range(K)])
    else:
        cp_phys = np.linspace(t0, t1, K)
    dims = list(range(n_dim))
    ncols = 3
    nrows = (len(dims) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1)
    for ax, d in zip(axes, dims):
        ax.fill_between(
            t_phys, lo_cal[:, d], hi_cal[:, d], color="blue", alpha=0.6, label="conformal band"
        )
        ax.plot(t_phys, lo_cal[:, d], color="blue", alpha=0.6, lw=0.9, ls=":")
        ax.plot(t_phys, hi_cal[:, d], color="blue", alpha=0.6, lw=0.9, ls=":")
        ax.plot(t_phys, true_traj[:, d], color="dimgray", lw=2, label="truth")
        ax.plot(t_phys, mean[:, d], color="red", lw=1.2, ls="--", label="mean")
        yerr_lo = cp_mean[:, d] - cp_lo[:, d]
        yerr_hi = cp_hi[:, d] - cp_mean[:, d]
        ax.errorbar(
            cp_phys,
            cp_mean[:, d],
            yerr=np.vstack([yerr_lo, yerr_hi]),
            fmt="o",
            color="black",
            ecolor="black",
            elinewidth=1.0,
            capsize=1.5,
            markersize=3,
            label="control points",
        )
        ax.set_ylabel(names[d], fontsize=11)
        ax.set_xlabel("Time [sec]", fontsize=11)
        ax.grid(alpha=0.3)
        ax.tick_params(labelbottom=True, labelsize=10)
        ax.legend(fontsize=10, loc="best")
        ax.tick_params(labelbottom=True)
    for ax in axes[len(dims) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)

@hydra.main(version_base=None, config_path="../../configs",config_name="final_spline")
def main(cfg: DictConfig):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    uniform = cfg["uniform"]

    # 1. IC and truth
    device = "cpu"
    model, train_cfg, qlevels = load_trajectory_model(cfg["checkpoint_path"], device)
    knot_model = load_knot_model(cfg["knot_model_path"], device)
    batch = sample_ic(train_cfg, cfg["seed"], device)

    # 2. Get knot vector
    norm_ics = batch["norm_ics"].to(device)
    ics = batch["ics"].to(device)
    jerk_predicted, knot_positions, knot_vector = knot_model.predict_knots(norm_ics)

    # 2.a - get just first value (or pick one)
    knot_vector = knot_vector.cpu()[0:1]
    ic = batch["ics"][0:1]
    norm_ic = batch["norm_ics"][0:1]

    # 3. Trajectory bands
    out = predict_trajectory_bands(model, norm_ic)
    lo, mean, hi = out["lo"], out["med"], out["hi"]
    R = torch.tensor(list(train_cfg.data.R), dtype=mean.dtype)
    w = torch.tensor(cfg["conformal"]["w"], dtype=mean.dtype)
    Q = float(cfg["conformal"]["Q"])

    # 4. Conformal widening
    lo_cal, hi_cal = apply_conformal(lo, hi, Q, R, w)
    deg = train_cfg["data"]["spline_degree"]
    n_cp_max = train_cfg["data"]["n_cp_max"]
    Ns = mean.shape[1]
    t = torch.linspace(0, 1, Ns, device=knot_vector.device, dtype=mean.dtype)

    # variable knot vector - get true end point
    end_knots = torch.where(knot_vector[0] == 1)[0]
    end_point = len(knot_vector[0]) if len(end_knots) <= deg + 1 else end_knots[deg + 1]
    kv = knot_vector[0, :end_point].to(mean.dtype)
    if uniform:
        # returns the knots (total number the same) uniformally spaced
        # 0,0,0,0, 0.1 ,1,1,1,1 ( for degree 3) get linspace [0,0.5,1]
        num_knots = end_knots[0] - deg  
        print(kv)
        print(deg, end_knots[0], num_knots)
        kv[deg:end_knots[0]] = torch.linspace(0,1, num_knots)
        print(kv)
    Bit = bspline_basis(t.to(mean.dtype), kv, deg)
    B_dag = torch.linalg.pinv(Bit)

    cp_mean = B_dag @ mean
    cp_med = B_dag @ (0.5 * (hi_cal + lo_cal))
    cp_diff = torch.abs(B_dag) @ (0.5 * (hi_cal - lo_cal))

    cp_lo = cp_med - cp_diff
    cp_hi = cp_med + cp_diff

    # Truth subsampled to the model's grid; physical-unit scaling for plotting.
    true_full = batch["norm_trajectories"][0:1]
    idx = torch.linspace(0, true_full.shape[1] - 1, Ns).round().long()
    true_traj = true_full[0, idx, :]
    R_b = R.view(1, -1)
    true_phys = true_traj * R_b
    mean_phys = mean[0] * R_b
    lo_phys = lo_cal[0] * R_b
    hi_phys = hi_cal[0] * R_b
    cp_mean_phys = cp_mean[0] * R_b
    cp_lo_phys = cp_lo[0] * R_b
    cp_hi_phys = cp_hi[0] * R_b
    t_phys = torch.linspace(0, float(train_cfg.data.time_horizon), Ns)

    # 9. Plot
    names = cfg.get("channel_names", [f"dim {d}" for d in range(mean.shape[-1])])
    plot_per_dim(
        t_phys,
        true_phys,
        mean_phys,
        lo_phys,
        hi_phys,
        knot_vector[0, :end_point],
        cp_mean_phys,
        cp_lo_phys,
        cp_hi_phys,
        names,
        out_dir / f"control_points_with_bounds{'_uniform' if uniform else ''}.png",
        degree=deg,
    )


if __name__ == "__main__":
    main()
