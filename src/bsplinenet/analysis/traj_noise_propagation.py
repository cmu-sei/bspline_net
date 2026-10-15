"""
bspline_net (Github)

Copyright 2026 Carnegie Mellon University.

NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

Licensed under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.

[DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.

This Software includes and/or makes use of Third-Party Software each subject to its own license.

DM26-0750
"""
# PAPER: TMLR
"""Input-noise robustness figure on CONFORMALLY-CALIBRATED bands (Option B).

Pipeline:
  1. Load the quantile model + datamodule.
  2. Calibrate the per-dim conformal scalar Q on a held-out set (reuses
     conformal_analysis: estimate_dim_weights / calibrate_joint_per_dim), so
     the bands achieve the target joint coverage. Widening is band +/- Q*R*w.
  3. Pick a random nominal IC, perturb it in a ball, push the whole set through
     in ONE batched forward pass.
  4. Apply the SAME Q*R*w widening to every perturbed band, then marginalize:
     the combined band is the mixture-quantile over the perturbation set.
     p(y|x0) = int p(y|x') P(x'|x0) dx'; sampling x'~P(x'|x0) does the weighting.

Illustrative demonstration of a capability the architecture enables (cheap,
batched, calibrated), not a validated empirical robustness claim.

Usage:
    python -m bsplinenet.analysis.traj_noise_propagation.py --config conformal_config.yaml
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from .traj_conformal_analysis import (
    band_residual,
    calibrate_joint_per_dim,
    collect_predictions,
    estimate_dim_weights,
    load_model,
)

from ..data import get_datamodule

# --------------------------- config (edit me) ------------------------------
N_PERTURB = 64  # perturbed ICs in the ball (batched -> ~free)
# Per-channel state-estimation error in PHYSICAL units (std for gaussian).
# >>> set these to your estimator's real accuracy <<<  order matches R:
#   [u(m/s), x(m), v(m/s), y(m), w(m/s), z(m),
#    p(rad/s), phi(rad), q(rad/s), theta(rad), r(rad/s), psi(rad)]
PHYS_STD = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
NOISE = "gaussian"  # "gaussian" (std=PHYS_STD/R) or "ball" (radius in norm units)
DIMS_TO_PLOT = list(range(12))  # all 12 state dims in a grid
MARGINAL_Q = 0.975  # mixture-quantile level for the combined band
SEED = 0

TIME_HORIZON = 2.5  # cfg.data.time_horizon  -> x-axis in seconds
R = np.array([3.0, 5.0, 3.0, 5.0, 3.0, 5.0, 1.5708, 0.7854, 1.5708, 0.7854, 1.5708, 0.7854])
CH_LABEL = [
    "u [m/s]",
    "x [m]",
    "v [m/s]",
    "y [m]",
    "w [m/s]",
    "z [m]",
    "p [rad/sec]",
    "$\\phi$ [rad]",
    "q [rad/s]",
    "$\\theta$ [rad]",
    "r [rad/sec]",
    "$\\psi$ [rad]",
]

PLOT_PHYSICAL = True  # multiply normalized outputs by R for physical units
SHOW_TRUE = True  # overlay the true trajectory for the nominal IC


def perturb(x0, n, phys_std, R, kind, rng):
    """Return (n, n_dim) perturbations of x0 (1, n_dim), in NORMALIZED units.

    phys_std: per-channel physical std (state-estimation error). Converted to
    the model's normalized space via /R, so each channel is perturbed by a
    physically-meaningful, appropriately-scaled amount (not a flat normalized
    epsilon that would mean different physical errors per channel)."""
    std_norm = phys_std / R  # per-channel normalized std
    d = x0.shape[-1]
    if kind == "gaussian":
        noise = rng.normal(0, 1, size=(n, d)) * std_norm[None, :]
    else:  # uniform within a per-channel-scaled ellipsoid
        dirs = rng.normal(0, 1, size=(n, d))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        radii = rng.uniform(0, 1, size=(n, 1)) ** (1.0 / d)
        noise = dirs * radii * std_norm[None, :]
    return x0 + noise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="conformal_config.yaml")
    ap.add_argument("--out", default="robustness_figure.png")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    # ---- load model + calibrate per-dim Q on a held-out set ----
    model, train_cfg, qlevels = load_model(cfg["checkpoint_path"])
    model.model.set_alpha(1.0)
    n_dim = model.model.n_dim
    nominal = qlevels[1] - qlevels[0]
    alpha = cfg.get("alpha", 1.0 - nominal)

    dm = get_datamodule("traj", train_cfg)
    lo_all, med_all, hi_all, y_all = collect_predictions(model, dm, cfg["calibration_size"])

    # SAME calibration split as conformal_analysis (first `frac` -> calibration).
    frac = cfg.get("calibration_fraction", 0.8)
    k = int(round(frac * lo_all.shape[0]))
    lo_c, med_c, hi_c, y_c = lo_all[:k], med_all[:k], hi_all[:k], y_all[:k]
    # drop the t=0 anchor (zero-width band by construction), as in conformal_analysis
    lo_c, med_c, hi_c, y_c = lo_c[:, 1:], med_c[:, 1:], hi_c[:, 1:], y_c[:, 1:]
    Rt = torch.tensor(list(train_cfg.data.R), dtype=lo_c.dtype)
    # Sub-split: first half estimates per-dim weights, second half computes Q.
    half = lo_c.shape[0] // 2
    weight_q = cfg.get("weight_quantile", 0.95)
    w = estimate_dim_weights(med_c[:half], y_c[:half], Rt, q=weight_q)
    Q, _ = calibrate_joint_per_dim(lo_c[half:], hi_c[half:], y_c[half:], alpha, Rt, w)
    print(f"Calibrated per-dim Q = {Q:.4e}  (widen by Q*R*w per channel)")

    # ---- pick the in-distribution IC with the WIDEST raw band, so the blue
    #      raw quantile band is visibly nonzero in the figure ----
    dm.setup("test")
    batch = next(iter(dm.test_dataloader()))
    ics_b = batch["norm_ics"]  # (B, n_dim)
    with torch.no_grad():
        ob = model.model(ics_b)
    raw_w = ob["hi"] - ob["lo"]  # (B, T, n_dim)
    # TARGET_DIM=None -> widest overall; else widest on that dimension
    TARGET_DIM = None
    if TARGET_DIM is None:
        score = raw_w.mean(dim=(1, 2))  # (B,)
    else:
        score = raw_w[:, :, TARGET_DIM].mean(dim=1)
    bi = int(score.argmax())
    print(f"selected batch IC {bi} (widest raw band; mean width {score[bi]:.4e})")

    x0 = ics_b[bi : bi + 1].numpy().astype(np.float32)  # (1, n_dim)
    y_true_full = batch["norm_trajectories"][bi].numpy()  # (n_samples, n_dim)

    x_pert = perturb(x0, N_PERTURB, PHYS_STD, R, NOISE, rng).astype(np.float32)
    x_all = np.concatenate([x0, x_pert], axis=0)

    with torch.no_grad():
        out = model.model(torch.from_numpy(x_all))
    lo_raw = out["lo"].numpy()
    med = out["med"].numpy()
    hi_raw = out["hi"].numpy()

    # ---- conformal widening (band +/- Q * R * w) -> calibrated bands ----
    Rwrow = (R * w.numpy()).reshape(1, 1, -1)
    lo = lo_raw - Q * Rwrow  # calibrated
    hi = hi_raw + Q * Rwrow

    T = med.shape[1]
    t_sec = np.linspace(0, TIME_HORIZON, T)

    # true trajectory: subsample the batch truth to the rollout grid (normalized)
    true_traj = None
    if SHOW_TRUE:
        idx = np.linspace(0, y_true_full.shape[0] - 1, T).round().astype(int)
        true_traj = y_true_full[idx] * (R if PLOT_PHYSICAL else 1.0)  # match plot units

    # nominal (row 0): raw (pre-conformal) and calibrated
    lo0_raw, hi0_raw = lo_raw[0], hi_raw[0]
    lo0, med0, hi0 = lo[0], med[0], hi[0]

    # ---- MARGINAL combined band (mixture-quantile, not a max-envelope) ----
    # p(y|x0) = int p(y|x') P(x'|x0) dx'. Sampling x'~P(x'|x0) IS the weighting,
    # so the mixture over samples approximates the marginal; we take percentiles
    # of the pooled band endpoints. Stable as N grows; far samples down-weighted.
    pert_lo = lo[1:]  # (N, T, dim)
    pert_hi = hi[1:]
    lo_marg = np.quantile(pert_lo, 1 - MARGINAL_Q, axis=0)
    hi_marg = np.quantile(pert_hi, MARGINAL_Q, axis=0)

    # physical-unit scaling (per channel)
    scale = R if PLOT_PHYSICAL else np.ones_like(R)

    # report band-width expansion (mean over time) per dim
    print("Band-width expansion (marginal vs nominal), mean over time:")
    for d in DIMS_TO_PLOT:
        w_nom = (hi0[:, d] - lo0[:, d]).mean()
        w_marg = (hi_marg[:, d] - lo_marg[:, d]).mean()
        print(f"  {CH_LABEL[d]:>14}: +{100*(w_marg/w_nom - 1):.1f}%")

    # plot
    nplots = len(DIMS_TO_PLOT)
    ncols = 3
    nrows = (nplots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1)
    for ax, d in zip(axes, DIMS_TO_PLOT):
        s = scale[d]
        # perturbed medians, faint (the input-noise spread)
        for j in range(1, med.shape[0]):
            ax.plot(t_sec, med[j, :, d] * s, color="lightgray", lw=0.8, alpha=0.5, zorder=1)
        # calibrated (conformal) band -- steelblue, matches conformal_analysis
        ax.fill_between(
            t_sec,
            lo0[:, d] * s,
            hi0[:, d] * s,
            color="blue",
            alpha=1.0,
            zorder=2,
            label="post CQR band",
        )
        ax.plot(t_sec, lo0[:, d] * s, color="blue", lw=0.9, ls=":", zorder=2)
        ax.plot(t_sec, hi0[:, d] * s, color="blue", lw=0.9, ls=":", zorder=2)
        # raw pre-conformal quantile band -- gold, matches conformal_analysis
        ax.fill_between(
            t_sec,
            lo0_raw[:, d] * s,
            hi0_raw[:, d] * s,
            color="goldenrod",
            alpha=1.0,
            zorder=3,
            label="raw quantile band",
        )
        # marginal combined band (input noise) -- green dashed outline
        ax.plot(
            t_sec,
            lo_marg[:, d] * s,
            color="darkgreen",
            lw=1.0,
            ls="--",
            zorder=4,
            label="marginal band (noise)",
        )
        ax.plot(t_sec, hi_marg[:, d] * s, color="darkgreen", lw=1.0, ls="--", zorder=4)
        if true_traj is not None:
            ax.plot(t_sec, true_traj[:, d], color="dimgray", lw=1.6, zorder=5, label="truth")
        # nominal mean
        ax.plot(t_sec, med0[:, d] * s, color="red", lw=1.0, ls="--", zorder=6, label="mean")
        # true trajectory
        ax.set_ylabel(CH_LABEL[d] if PLOT_PHYSICAL else f"{CH_LABEL[d]} (norm)", fontsize=11)
        ax.grid(alpha=0.3)
    for ax in axes[nplots:]:
        ax.axis("off")
    for ax in axes[:nplots]:
        ax.set_xlabel("Time [sec]", fontsize=11)
        ax.legend(loc="best", fontsize=10)
        ax.tick_params(
            labelbottom=True, labelsize=11
        )  # force tick labels (sharex hides them otherwise)
    fig.suptitle(
        f"Input-noise robustness on calibrated bands "
        f"(per-channel state-est. error, N={N_PERTURB}, "
        f"marginal {int(MARGINAL_Q*100)}%) - illustrative",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=600, bbox_inches="tight")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
