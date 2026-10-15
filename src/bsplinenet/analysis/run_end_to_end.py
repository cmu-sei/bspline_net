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
"""End-to-end control-interval margin assessment.

(1) Loads a trajectory model + knot model
(2) Generates test trajectories, 
(3) applies precomputed conformal Q*R*w (as in final_spline_pred.py), and 
(4)projects those bands onto each trajectory's own
B-spline control-point basis. Reports:
  - a per-channel histogram of the signed margin between the true
    trajectory's own (least-squares-fit) control points and the calibrated
    control-point interval edge (cp_margin_histograms.png);
  - the Eq 20 approximation residual (true trajectory vs. its own spline
    fit) against the calibrated half-width, at both the trajectory level
    and the control-point level.

Q, R, w are not recalibrated here -- they come from a prior
traj_conformal_analysis.py run on the given checkpoint (see
src/configs/cp_interval_analysis.yaml).

Usage:
    python -m bsplinenet.analysis.run_end_to_end
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from .final_spline_pred import apply_conformal, control_point_intervals, load_knot_model
from .traj_conformal_analysis import CHANNEL_NAMES, band_residual, load_model
from ..data import get_datamodule
from ..utils import bspline_basis, compute_controlpoints_from_trajectory


def collect_eval_data(traj_model, knot_model, datamodule):
    """Run the trajectory model and the knot model on one batch of test
    trajectories.

    lo/hi: (N, T, n_dim) on the model's rollout grid (t=0 included).
    traj_full: (N, n_samples, n_dim), the un-subsampled trajectory.
    knot_vector: (N, n_knots_total).
    """
    datamodule.setup("test")
    batch = next(iter(datamodule.test_dataloader()))
    norm_ics = batch["norm_ics"]
    with torch.no_grad():
        out = traj_model.model(norm_ics)
        _, _, knot_vector = knot_model.predict_knots(norm_ics)
    return out["lo"], out["hi"], batch["norm_trajectories"], knot_vector


def cp_interval_margin(control_points, lo_cal, hi_cal, knot_vector, degree):
    """Per-trajectory, per-dimension: how far the true trajectory's own
    least-squares-fit control points sit from the nearest edge of the
    calibrated control-point interval -- positive means outside, negative
    means inside.

    Each trajectory has its own knot vector (own basis, own pinv), so the
    interval is built per-trajectory via control_point_intervals; the true
    control points come in already fit (control_points, from
    compute_controlpoints_from_trajectory) and are sliced to that
    trajectory's own active count.

    Returns:
        margin: (N, n_dim), worst-case (max over that trajectory's control
            points) signed distance to the interval edge.
        cp_half_width: (N, n_dim), that trajectory's own mean (over its
            control points) calibrated control-point interval half-width --
            for comparing against the trajectory-space half-width directly.
    """
    N, T = control_points.shape[0], lo_cal.shape[1] # number of trajectories, rollout time steps
    t_grid = torch.linspace(0, 1, T, dtype=lo_cal.dtype)

    #outputs (empty/zero arrays)
    margin = torch.zeros(N, control_points.shape[-1], dtype=lo_cal.dtype)
    cp_half_width = torch.zeros_like(margin)

    for b in range(N): # for each tarjectory

        #build spline basis
        end_knots = torch.where(knot_vector[b] == 1)[0]
        end_point = len(knot_vector[b]) if len(end_knots) <= degree + 1 else end_knots[degree + 1]
        kv = knot_vector[b, :end_point]
        n_cp_b = end_point - degree - 1

        Bit = bspline_basis(t_grid, kv, degree)
        B_dag = torch.linalg.pinv(Bit)

        # calibrated interval
        cp_lo_cal, cp_hi_cal = control_point_intervals(B_dag, lo_cal[b], hi_cal[b])

        # control points of true trajectory's least squares fit (best)
        true_cp = control_points[b, :n_cp_b]

        #signed difference
        margin[b] = band_residual(cp_lo_cal, cp_hi_cal, true_cp).amax(dim=0)

        #width
        cp_half_width[b] = (0.5 * (cp_hi_cal - cp_lo_cal)).mean(dim=0)

    return margin, cp_half_width


def plot_cp_margin_histograms(names, margin, path):
    """Grid of per-channel histograms of the signed distance from the true
    control points to the nearest calibrated-interval edge (positive =
    outside, negative = inside), with a vertical line at 0."""
    n = len(names)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows))
    axes = np.array(axes).reshape(-1)
    for d, (ax, name) in enumerate(zip(axes, names)):
        ax.hist(margin[:, d], bins=41, color="C1", alpha=0.7)
        ax.axvline(0.0, color="C3", ls="--", lw=1.2)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("outside <-- margin --> inside", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=350, bbox_inches="tight")
    plt.close(fig)


@hydra.main(config_path="../../configs", config_name="cp_interval_analysis")#, version_base=None)
def main(cfg: DictConfig) -> None:

    torch.manual_seed(cfg.get("analysis_seed", 0))
    np.random.seed(cfg.get("analysis_seed", 0))
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    traj_model, train_cfg, qlevels = load_model(cfg["checkpoint_path"])
    knot_model = load_knot_model(cfg["knot_model_path"], device="cpu")
    nominal = qlevels[1] - qlevels[0]
    print(f"Quantile levels {qlevels}  ->  nominal coverage {nominal * 100:.1f}%")

    R = torch.tensor(list(cfg["conformal"]["R"]), dtype=torch.float32)
    w = torch.tensor(list(cfg["conformal"]["w"]), dtype=torch.float32)
    Q = float(cfg["conformal"]["Q"])
    degree = train_cfg.data.spline_degree
    names = cfg.get("channel_names", CHANNEL_NAMES)

    ## generate data
    # One batch of exactly eval_size trajectories, rather than accumulating
    # over several dataloader batches.
    OmegaConf.update(train_cfg, "data.batch_size", cfg["eval_size"], force_add=True)
    OmegaConf.update(train_cfg, "data.val_batches", 1, force_add=True)
    dm = get_datamodule("traj", train_cfg)

    ## run the neural network on all generated data
    lo, hi, traj_full, knot_vector = collect_eval_data(traj_model, knot_model, dm)
    print(f"Collected {lo.shape[0]} test trajectories, (N, T, n_dim) = {tuple(lo.shape)}")

    ## build control intervals for each trajectory
    lo_cal, hi_cal = apply_conformal(lo, hi, Q, R, w)

    # True control points: proper least-squares spline fit of the full-res
    # trajectory against its own (predicted) knot vector -- reused for both
    # the control-point margin and the Eq 20 residual.
    n_cp_max = train_cfg.data.n_cp_max
    control_points, fits = compute_controlpoints_from_trajectory(
        traj_full, knot_vector, degree, n_cp_max, reconstruct_fit=True
    )
    cp_margin, cp_half_width = cp_interval_margin(
        control_points, lo_cal, hi_cal, knot_vector, degree
    )  # each (N, n_dim)
    cp_margin_phys = cp_margin * R  # physical units, for plotting

    # Eq 20: sup-norm residual between the true trajectory and its own
    # least-squares spline fit, vs. the calibrated half-width.

    eps_knot = (traj_full - fits).abs().amax(dim=1)  # (N, n_dim), sup over time
    half_width_traj = (0.5 * (hi_cal - lo_cal)).mean(dim=1)  # (N, n_dim), per trajectory
    eps_knot_p95 = torch.quantile(eps_knot, 0.95, dim=0)  # (n_dim,), across trajectories
    half_width_traj_mean = half_width_traj.mean(dim=0)  # (n_dim,)
    p95_ratio = eps_knot_p95 / half_width_traj_mean  # (n_dim,)

    ## print statistics
    print("\n--- residual vs. calibrated half-width (test, mean) ---")
    print(
        f"  {'dimension':>14s}   {'eps_knot':>10s}   {'eps_knot p95':>12s}   "
        f"{'traj half-width':>16s}   {'cp half-width':>14s}   {'p95/traj hw':>11s}"
    )
    for d, name in enumerate(names):
        print(
            f"  {name:>14s}   {(eps_knot.mean(dim=0)[d] * R[d]):10.3e}  "
            f"{(eps_knot_p95[d] * R[d]):12.3e}  "
            f"{(half_width_traj_mean[d] * R[d]):16.3e}  "
            f"{(cp_half_width.mean(dim=0)[d] * R[d]):14.3e}   "
            f"{p95_ratio[d] * 100:10.1f}%"
        )


    ## make a histogram of the control-point margin, per channel
    plot_cp_margin_histograms(names, cp_margin_phys.numpy(), out_dir / "cp_margin_histograms.png")

    print(f"Done. Saved plot to {out_dir}/")


if __name__ == "__main__":
    main()
