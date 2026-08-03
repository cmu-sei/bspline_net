"""
bspline_net (Github)

Copyright 2026 Carnegie Mellon University.

NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

Licensed under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.

[DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.

This Software includes and/or makes use of Third-Party Software each subject to its own license.

DM26-0750
"""
"""Per-dim-weighted conformal prediction for the quantile-mode trajectory model.

For each calibration trajectory, score = max over (t, d) of band_residual / (R[d]*w[d]).
Conformal Q = (1-alpha)(1+1/n) quantile of those scores. Widen by Q*R[d]*w[d] per dim.
Joint guarantee: P(truth in band at every (t,d)) >= 1-alpha. Per-dim weights w are
estimated on a held-out portion of the calibration set, as the q-th percentile
of |y - med|/R[d] within each dim (normalized to geometric mean 1), so that the
score function is fixed before computing Q (preserving exchangeability).

Usage :
    # Default config (conformal_config.yaml in the same folder)
    python -m bsplinenet.analysis.traj_conformal_analysis

"""

import matplotlib.pyplot as plt
import numpy as np
import torch
#import yaml
from omegaconf import DictConfig

import hydra
from pathlib import Path

from ..data import get_datamodule
from ..traj_model import LitTrajModel

CHANNEL_NAMES = [
    "u [m/sec]",
    "x [m]",
    "v [m/sec]",
    "y [m]",
    "w [m/sec]",
    "z [m]",
    "p [rad/sec]",
    "phi [rad]",
    "q [rad/sec]",
    "theta [rad]",
    "r [rad/sec]",
    "psi [rad]",
]

# ----------------------------------------------------------------------
# Helper functions (unchanged)
# ----------------------------------------------------------------------
def load_model(checkpoint_path):
    """ Load the model 
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["hyper_parameters"]["cfg"]
    model = LitTrajModel(cfg)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    qlevels = model.model.quantile_levels
    if qlevels is None or len(qlevels) != 2:
        raise RuntimeError(f"Need quantile-mode model with 2 levels; got {qlevels}.")
    return model, cfg, list(qlevels)


def collect_predictions(model, datamodule, max_samples):
    """ Collect predictions from the model 
    """
    datamodule.setup("test") #in test mode
    los, meds, his, ys = [], [], [], []
    seen, passes = 0, 0
    while seen < max_samples:
        for batch in datamodule.test_dataloader():
            with torch.no_grad():
                out = model.model(batch["norm_ics"])
            traj = batch["norm_trajectories"]
            idx = torch.linspace(0, traj.shape[1] - 1, out["med"].shape[1]).round().long()
            los.append(out["lo"])
            meds.append(out["med"])
            his.append(out["hi"])
            ys.append(traj[:, idx, :])
            seen += out["med"].shape[0]
            if seen >= max_samples:
                break
        passes += 1
        if passes > 1000:
            print(f"WARNING: stopped after {passes} passes with {seen} samples")
            break
    return tuple(torch.cat(xs, dim=0)[:max_samples] for xs in (los, meds, his, ys))


def band_residual(lo, hi, y):
    """CQR signed score: max(lo-y, y-hi). Positive iff y outside [lo, hi]."""
    return torch.maximum(lo - y, y - hi)


def conformal_quantile(scores, alpha, dim=0):
    """Finite-sample-adjusted (1-alpha) quantile of `scores` along `dim`."""
    n = scores.shape[dim]
    return torch.quantile(scores, min(1.0, (1.0 - alpha) * (1.0 + 1.0 / n)), dim=dim)


def estimate_dim_weights(med, y, R, q=0.95):
    """w[d] = q-th percentile of |y - med|/R[d] per dim, geom-mean normalized to 1.

    Uses |residual|, NOT band_residual: signed band_residual goes negative for
    well-covered cells and confounds difficulty with band slack.
    (numpy quantile, not torch -- torch.quantile has a ~16M element limit.)
    """
    err = (y - med).abs() / R.view(1, 1, -1)
    err_per_dim = err.permute(2, 0, 1).reshape(err.shape[-1], -1).cpu().numpy()
    w = torch.tensor(np.quantile(err_per_dim, q, axis=1), dtype=err.dtype).clamp_min(1e-12)
    return w / torch.exp(torch.log(w).mean())


def calibrate_joint_per_dim(lo, hi, y, alpha, R, w):
    """Scalar Q for joint coverage with per-dim scaling. Widening: +/- Q*R[d]*w[d]."""
    err = band_residual(lo, hi, y) / (R * w).view(1, 1, -1)
    per_traj = err.flatten(1).amax(1)
    return float(conformal_quantile(per_traj, alpha)), per_traj


def coverage(lo, hi, y):
    """ helper function for coverage of y between low and hi (marginal)
    """
    inside = (y >= lo) & (y <= hi)
    return inside.float().mean(dim=0), inside.float().mean().item()


def trajectory_coverage(lo, hi, y):
    """ helper function for coverage of y between low and hi (trajectory)
    """
    return ((y >= lo) & (y <= hi)).flatten(1).all(dim=1).float().mean().item()


def plot_score_distribution(scores, Q, alpha, path):
    """Plot score distribution
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores.cpu().numpy(), bins=120, log=True, color="C0", alpha=0.7)
    ax.axvline(Q, color="C3", ls="--", label=f"Q = {Q:.3g}  ({(1-alpha)*100:.1f}% quantile)")
    ax.set(
        xlabel="Per-trajectory score (R*w-normalized worst cell)",
        ylabel="Count (log)",
        title="Per-dim weighted conformity scores",
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_example(i, lo, med, hi, lo_w, hi_w, y, t, dims, names, path):
    ncols = 3
    nrows = (len(dims) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1)
    for ax, d in zip(axes, dims):
        ax.fill_between(
            t,
            lo_w[i, :, d].cpu(),
            hi_w[i, :, d].cpu(),
            color="blue",
            alpha=1.0,
            label="per-dim band",
        )
        ax.plot(t, lo_w[i, :, d].cpu(), color="blue", lw=0.9, ls=":")
        ax.plot(t, hi_w[i, :, d].cpu(), color="blue", lw=0.9, ls=":")
        ax.fill_between(
            t,
            lo[i, :, d].cpu(),
            hi[i, :, d].cpu(),
            color="goldenrod",
            alpha=1.0,
            label="raw quantile band",
        )
        ax.plot(t, y[i, :, d].cpu(), color="dimgray", lw=2, label="truth")
        ax.plot(t, med[i, :, d].cpu(), color="red", lw=1.2, ls="--", label="mean")

        ax.grid(alpha=0.3)
        ax.set_ylabel(names[d], fontsize=11)
        ax.set_xlabel("Time [sec]", fontsize=11)
        ax.tick_params(labelbottom=True, labelsize=10)
        ax.legend(fontsize=10, loc="best")
        ax.tick_params(labelbottom=True)
    for ax in axes[len(dims) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)


@hydra.main(version_base=None, config_path="../../configs", config_name="conformal_config")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.get("analysis_seed", 0))
    np.random.seed(cfg.get("analysis_seed", 0))
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    model, train_cfg, qlevels = load_model(cfg["checkpoint_path"])
    nominal = qlevels[1] - qlevels[0]
    alpha = cfg.get("alpha", 1.0 - nominal)
    print(f"Quantile levels {qlevels}  ->  nominal coverage {nominal*100:.1f}%")
    print(f"alpha = {alpha}  (target >= {(1-alpha)*100:.1f}%)")

    dm = get_datamodule("traj", train_cfg)
    lo, med, hi, y = collect_predictions(model, dm, cfg["calibration_size"])
    print(f"Collected {lo.shape[0]} samples, (N, T, n_dim) = {tuple(lo.shape)}")

    # Cal/test split, drop the t=0 anchor (zero band width by construction).
    k = int(round(cfg.get("calibration_fraction", 0.8) * lo.shape[0]))
    lo_c, med_c, hi_c, y_c = (a[:k, 1:] for a in (lo, med, hi, y))
    lo_t, med_t, hi_t, y_t = (a[k:, 1:] for a in (lo, med, hi, y))
    print(f"Split: {k} calibration / {lo_t.shape[0]} test")

    T = lo_t.shape[1]
    R = torch.tensor(list(train_cfg.data.R), dtype=med.dtype)
    n_samples = int(train_cfg.data.n_samples)
    tt = torch.linspace(0.0, float(train_cfg.data.time_horizon), n_samples)
    t_phys = tt[torch.linspace(0, n_samples - 1, T + 1).round().long()][1:]

    # Sub-split: first half estimates per-dim weights, second half calibrates Q.
    half = lo_c.shape[0] // 2
    weight_q = cfg.get("weight_quantile", 0.95)
    w = estimate_dim_weights(med_c[:half], y_c[:half], R, q=weight_q)
    Q, scores = calibrate_joint_per_dim(lo_c[half:], hi_c[half:], y_c[half:], alpha, R, w)
    print(f"\nCalibration sub-split: {half} weight estimation / {lo_c.shape[0]-half} Q calibration")

    print(f"\nPer-dim weights w (q={weight_q*100:.0f}, geom-mean=1):")
    names = cfg.get("channel_names", CHANNEL_NAMES)
    for d, (name, wd) in enumerate(zip(names, w.tolist())):
        print(f"  {name:>14s}  w[{d}] = {wd:.3f}")
    print(f"\nQ per-dim       (scalar):   {Q:.3e}")

    Rwb = (R * w).view(1, 1, -1)
    bands = {"raw": (lo_t, hi_t), "per-dim": (lo_t - Q * Rwb, hi_t + Q * Rwb)}

    print("\n--- Coverage (test) ---")
    print(f"  {'method':<10} cell-level   trajectory-level   mean width")
    for name, (l, h) in bands.items():
        cov_td, _ = coverage(l, h, y_t)
        print(
            f"  {name:<10} {cov_td.mean()*100:7.2f}%      "
            f"{trajectory_coverage(l, h, y_t)*100:7.2f}%        {(h-l).mean():.3e}"
        )

    print("\n--- Mean band width per dimension (test) ---")
    print(f"  {'channel':>14s}   {'raw':>10s}   {'per-dim':>10s}")
    for d, name in enumerate(names):
        w_raw = (hi_t[:, :, d] - lo_t[:, :, d]).mean().item()
        w_pd = (bands["per-dim"][1][:, :, d] - bands["per-dim"][0][:, :, d]).mean().item()
        print(f"  {name:>14s}   {w_raw:10.3e}   {w_pd:10.3e}")

    plot_score_distribution(scores, Q, alpha, out_dir / "joint_scores.png")
    lo_w, hi_w = bands["per-dim"]
    worst = band_residual(lo_t, hi_t, y_t).amax(dim=(1, 2)).argmax().item()
    plot_example(
        worst,
        lo_t,
        med_t,
        hi_t,
        lo_w,
        hi_w,
        y_t,
        t_phys.cpu().numpy(),
        cfg.get("plot_dims", [0, 4]),
        names,
        out_dir / "example.png",
    )
    print(f"\nSaved plots to {out_dir}/  (worst case example: sample {worst})")


if __name__ == "__main__":
    main()
    