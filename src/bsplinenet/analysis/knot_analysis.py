"""
bspline_net (Github)

Copyright 2026 Carnegie Mellon University.

NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

Licensed under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.

[DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.

This Software includes and/or makes use of Third-Party Software each subject to its own license.

DM26-0750
"""
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from bsplinenet.data import KnotDataModule, compute_derived_data_config, get_datamodule
from bsplinenet.knot_model import ActivityModel, LitKnotModel
from bsplinenet.utils import compute_controlpoints_from_trajectory

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 18,
    }
)

C_TRUTH = "black"
C_PRED = "blue"
C_EVEN = "darkorange"


def greville_abscissae(knots, degree):
    n_cps = len(knots) - degree - 1
    greville = np.array([knots[i + 1 : i + degree + 1].mean() for i in range(n_cps)])
    return greville


def plot_example(
    idx,
    trajectories_all,
    jerk_fits,
    even_fits,
    jerk_cps,
    even_cps,
    knot_vectors,
    knot_vectors_even,
    time_horizon,
    spline_degree,
):
    traj = trajectories_all[idx]
    fit_pred = jerk_fits[idx]
    fit_even = even_fits[idx]
    cps_pred = jerk_cps[idx]
    cps_even = even_cps[idx]
    kv_pred = knot_vectors[idx]
    kv_even = knot_vectors_even[idx]

    if traj.ndim == 1:
        traj = traj[:, None]
        fit_pred = fit_pred[:, None]
        fit_even = fit_even[:, None]

    n_dims = traj.shape[1]
    t_traj = np.linspace(0, time_horizon, traj.shape[0])

    # Greville abscissae for control point x-positions (scaled to time horizon)
    greville_pred = greville_abscissae(kv_pred, spline_degree) * time_horizon
    greville_even = greville_abscissae(kv_even, spline_degree) * time_horizon

    n_cp_pred = len(greville_pred)
    n_cp_even = len(greville_even)

    mask_pred = np.any(cps_pred != 0, axis=-1) if cps_pred.ndim > 1 else cps_pred != 0
    mask_even = np.any(cps_even != 0, axis=-1) if cps_even.ndim > 1 else cps_even != 0
    # --- Top rows: trajectory + fits + control points per dimension ---
    # --- Bottom row: residuals ---
    fig, axes = plt.subplots(n_dims, 2, figsize=(16, 5 * n_dims))
    if n_dims == 1:
        axes = axes[None, :]  # make 2D

    dim_name = [
        "x velocity [m/s]",
        "x position [m]",
        "y velocity [m/s]",
        "y position [m]",
        "z velocity [m/s]",
        "z position [m]",
        "roll rate [rad/s]",
        "roll [radians]",
        "pitch rate [rads/s]",
        "pitch [radians]",
        "yaw rate [rad/s]",
        "yaw [radians]",
    ]

    for d in range(n_dims):
        ax_fit = axes[d, 0]
        ax_res = axes[d, 1]

        # Left: trajectory + fits + control points
        ax_fit.plot(t_traj, traj[:, d], "-", linewidth=2, label="Truth", alpha=0.7, color=C_TRUTH)
        ax_fit.plot(
            t_traj, fit_pred[:, d], "--", linewidth=1.5, label="Predicted Knots", color=C_PRED
        )
        ax_fit.plot(t_traj, fit_even[:, d], ":", linewidth=1.5, label="Even-Spaced", color=C_EVEN)

        cp_dim_pred = cps_pred[:, d] if cps_pred.ndim > 1 else cps_pred
        cp_dim_even = cps_even[:, d] if cps_even.ndim > 1 else cps_even
        ax_fit.scatter(
            greville_pred[mask_pred],
            cp_dim_pred[mask_pred],
            color=C_PRED,
            marker="^",
            s=50,
            zorder=5,
            label="Pred CPs",
        )
        ax_fit.scatter(
            greville_even[mask_even],
            cp_dim_even[mask_even],
            color=C_EVEN,
            marker="v",
            s=50,
            zorder=5,
            label="Even CPs",
        )
        # Mark actual knot positions on x-axis
        valid_kv_pred = kv_pred[(kv_pred > 0) & (kv_pred < 1)] * time_horizon
        valid_kv_even = kv_even[(kv_even > 0) & (kv_even < 1)] * time_horizon
        ax_fit.scatter(
            valid_kv_pred,
            np.full_like(valid_kv_pred, ax_fit.get_ylim()[0]),
            color=C_PRED,
            marker="|",
            s=100,
            zorder=5,
            alpha=0.5,
        )
        ax_fit.scatter(
            valid_kv_even,
            np.full_like(valid_kv_even, ax_fit.get_ylim()[0]),
            color=C_EVEN,
            marker="|",
            s=100,
            zorder=5,
            alpha=0.5,
        )

        ax_fit.set_ylabel(f"{dim_name[d]}")
        ax_fit.set_xlabel("Time [s]")
        ax_fit.legend(fontsize=10)
        ax_fit.set_title(f"{dim_name[d]}")

        # Right: residuals (this is where you'll sta the difference)
        res_pred = traj[:, d] - fit_pred[:, d]
        res_even = traj[:, d] - fit_even[:, d]
        ax_res.plot(t_traj, res_pred, "-", linewidth=1, label="Pred Residual", color=C_PRED)
        ax_res.plot(t_traj, res_even, "-", linewidth=1, label="Even Residual", color=C_EVEN)
        ax_res.axhline(0, color="k", linewidth=0.5, linestyle="--")
        ax_res.set_ylabel("Residual")

        ax_res.set_xlabel("Time [s]")
        ax_res.set_title(f"{dim_name[d]}: Residuals")
        ax_res.legend(fontsize=10)

    rmse_p = np.sqrt(((traj - fit_pred) ** 2).mean())
    rmse_e = np.sqrt(((traj - fit_even) ** 2).mean())
    r2_p = 1 - ((traj - fit_pred) ** 2).sum() / ((traj - traj.mean(axis=0)) ** 2).sum()
    r2_e = 1 - ((traj - fit_even) ** 2).sum() / ((traj - traj.mean(axis=0)) ** 2).sum()

    fig.suptitle(
        f"Sample {idx}\n"
        f"Predicted: RMSE={rmse_p:.6f}, R^2={r2_p:.4f}  |  "
        f"Even-Spaced: RMSE={rmse_e:.6f}, R^2={r2_e:.4f}",
        fontsize=14,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"analyses/knots/example_sample_{idx}.png", dpi=150)
    plt.show()


def plot_cp_distance_vs_jerk(
    jerk_cps,
    even_cps,
    trajectories_all,
    jerk_fits,
    knot_vectors,
    knot_vectors_even,
    time_horizon,
    spline_degree,
    jerk_preds,
):

    groups = {
        "Position [m]": [1, 3, 5],
        "Velocity [m/s]": [0, 2, 4],
        "Angle [rad]": [7, 9, 11],
        "Angular Rate [rad/s]": [6, 8, 10],
    }

    data_pred = {g: {"jerk": [], "dist": []} for g in groups}
    data_even = {g: {"jerk": [], "dist": []} for g in groups}

    n_jerk_samples = jerk_preds.shape[1]

    for i in range(len(trajectories_all)):
        traj = trajectories_all[i]
        cps_pred = jerk_cps[i]
        cps_even = even_cps[i]
        kv_pred = knot_vectors[i]
        kv_even = knot_vectors_even[i]
        jerk_i = jerk_preds[i]

        greville_pred = greville_abscissae(kv_pred, spline_degree)
        greville_even = greville_abscissae(kv_even, spline_degree)

        mask_pred = np.any(cps_pred != 0, axis=-1) if cps_pred.ndim > 1 else cps_pred != 0
        mask_even = np.any(cps_even != 0, axis=-1) if cps_even.ndim > 1 else cps_even != 0

        n_t = traj.shape[0]

        for g_name, dims in groups.items():
            for g, cp in zip(greville_pred[mask_pred], cps_pred[mask_pred]):
                t_idx = int(np.clip(g * (n_t - 1), 0, n_t - 1))
                j_idx = int(np.clip(g * (n_jerk_samples - 1), 0, n_jerk_samples - 1))
                dist = np.sqrt(((cp[dims] - traj[t_idx, dims]) ** 2).sum())
                data_pred[g_name]["jerk"].append(jerk_i[j_idx])
                data_pred[g_name]["dist"].append(dist)

            for g, cp in zip(greville_even[mask_even], cps_even[mask_even]):
                t_idx = int(np.clip(g * (n_t - 1), 0, n_t - 1))
                j_idx = int(np.clip(g * (n_jerk_samples - 1), 0, n_jerk_samples - 1))
                dist = np.sqrt(((cp[dims] - traj[t_idx, dims]) ** 2).sum())
                data_even[g_name]["jerk"].append(jerk_i[j_idx])
                data_even[g_name]["dist"].append(dist)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, g_name in zip(axes.flat, groups.keys()):
        j_pred = np.array(data_pred[g_name]["jerk"])
        d_pred = np.array(data_pred[g_name]["dist"])
        j_even = np.array(data_even[g_name]["jerk"])
        d_even = np.array(data_even[g_name]["dist"])

        ax.scatter(j_even, d_even, alpha=0.2, s=0.3, color="bisque")  # , label="Even-Spaced")
        ax.scatter(j_pred, d_pred, alpha=0.2, s=0.3, color="lightsteelblue")  # , label="Predicted")

        # Binned mean line
        n_bins = 20
        all_jerk = np.concatenate([j_pred, j_even])
        bin_edges = np.linspace(np.percentile(all_jerk, 1), np.percentile(all_jerk, 99), n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        mean_pred = np.array(
            [
                (
                    d_pred[(j_pred >= bin_edges[j]) & (j_pred < bin_edges[j + 1])].mean()
                    if ((j_pred >= bin_edges[j]) & (j_pred < bin_edges[j + 1])).sum() > 0
                    else np.nan
                )
                for j in range(n_bins)
            ]
        )
        mean_even = np.array(
            [
                (
                    d_even[(j_even >= bin_edges[j]) & (j_even < bin_edges[j + 1])].mean()
                    if ((j_even >= bin_edges[j]) & (j_even < bin_edges[j + 1])).sum() > 0
                    else np.nan
                )
                for j in range(n_bins)
            ]
        )

        ax.plot(
            bin_centers, mean_pred, "-o", color=C_PRED, linewidth=2, markersize=4, label="Pred Mean"
        )
        ax.plot(
            bin_centers, mean_even, "-o", color=C_EVEN, linewidth=2, markersize=4, label="Even Mean"
        )

        ax.set_xlabel("Predicted Jerk Activity")
        ax.set_ylabel(f"CP Distance - {g_name}")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, mean_even.max() * 1.1)
        ax.set_title(g_name)
        ax.legend(fontsize=10, frameon=False)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Control Point Distance to Trajectory vs Predicted Jerk Activity")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("analyses/knots/cp_distance_vs_jerk.png", dpi=150)
    plt.show()


@hydra.main(config_path="../../configs", config_name="knot_analysis", version_base=None)
def main(cfg: DictConfig):

    cfg = compute_derived_data_config(cfg)
    # Override data config for analysis
    OmegaConf.update(cfg, "data.batch_size", 1000, force_add=True)
    OmegaConf.update(cfg, "data.test_batches", 10, force_add=True)
    OmegaConf.update(cfg, "data.num_workers", 1, force_add=True)
    OmegaConf.update(cfg, "model.knots.max_error", 0.001, force_add=True)

    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # --- Load model from checkpoint ---
    checkpoint = torch.load(cfg.ckpt_path, map_location=device, weights_only=False)
    model = LitKnotModel(**checkpoint["hyper_parameters"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.freeze()

    model = model.to(device)

    # --- Set up data ---
    # Get datamodule
    datamodule = get_datamodule(cfg.stage, cfg)
    test_loader = datamodule.test_dataloader()

    # --- Run inference and collect results ---
    all_ics = []
    all_jerk_preds = []
    all_jerk_truth = []
    all_knot_positions = []
    all_knot_vector = []
    all_knot_vector_even = []
    all_jerk_fits = []
    all_even_fits = []
    all_jerk_cps = []
    all_even_cps = []
    all_n_knots = []
    with torch.no_grad():
        for b, batch in enumerate(test_loader):
            print(f"\rbatch {b}", end="")
            norm_ics = batch["norm_ics"].to(device)
            jerk_truth = batch["jerk_activity_max"]
            jerk_predicted, knot_positions, knot_vector = model.predict_knots(norm_ics)
            jerk_predicted = jerk_predicted.cpu()
            knot_positions = knot_positions.cpu()
            knot_vector = knot_vector.cpu()
            jerk_predicted_cps, jerk_predicted_fits = compute_controlpoints_from_trajectory(
                batch["trajectories"].to(knot_vector.device),
                knot_vector,
                cfg.data.spline_degree,
                cfg.data.n_cp_max,
                True,
            )
            jerk_predicted_cps = jerk_predicted_cps.cpu()
            jerk_predicted_fits = jerk_predicted_fits.cpu()

            start = cfg.data.spline_degree  # +1 no minus 1 b/c first val of linspace is zero
            n_knots = torch.argmax(knot_positions.cpu(), dim=1)
            # breakpoint()

            edit_rows = torch.repeat_interleave(torch.arange(len(n_knots)), n_knots + 1)
            edit_cols = torch.cat([torch.arange(start, n + start + 1, 1) for n in n_knots])
            # edit_vals = torch.cat([torch.arange(0, 1, 1/(n+1) ) for n in n_knots])
            edit_vals = torch.cat([torch.arange(n + 1).float() / (n + 1) for n in n_knots])
            knot_vector_even = knot_vector.clone().cpu()
            knot_vector_even[edit_rows, edit_cols] = edit_vals
            # knot_vector_even = knot_vector_even**1.5

            even_space_cps, even_space_fits = compute_controlpoints_from_trajectory(
                batch["trajectories"].cpu(),
                knot_vector_even.cpu(),
                cfg.data.spline_degree,
                cfg.data.n_cp_max,
                True,
            )

            # print('-------')
            # print('ics', batch['ics'][0:3,0])
            # print('jerk cps ', jerk_predicted_cps[0:3,:5,0])
            # print('even cps ', even_space_cps[0:3,:5,0])
            # print('jerk fit ', jerk_predicted_fits[0:3,:5,0])
            # print('even fit ', even_space_fits[0:3,:5,0])
            # exit()
            all_ics.append(batch["ics"].cpu().numpy())
            all_jerk_preds.append(jerk_predicted.cpu().numpy())
            all_jerk_truth.append(jerk_truth.cpu().numpy())
            all_knot_positions.append(knot_positions.cpu().numpy())
            all_knot_vector.append(knot_vector.cpu().numpy())
            all_knot_vector_even.append(knot_vector_even.cpu().numpy())
            all_jerk_fits.append(jerk_predicted_fits.cpu().numpy())
            all_jerk_cps.append(jerk_predicted_cps.cpu().numpy())
            all_even_fits.append(even_space_fits.cpu().numpy())
            all_even_cps.append(even_space_cps.cpu().numpy())
            all_n_knots.append(n_knots.cpu().numpy())

    ics = np.concatenate(all_ics, axis=0)
    jerk_preds = np.concatenate(all_jerk_preds, axis=0)
    jerk_truth = np.concatenate(all_jerk_truth, axis=0)
    knot_vectors = np.concatenate(all_knot_vector, axis=0)
    knot_vectors_even = np.concatenate(all_knot_vector_even, axis=0)
    jerk_fits = np.concatenate(all_jerk_fits, axis=0)
    jerk_cps = np.concatenate(all_jerk_cps, axis=0)
    even_fits = np.concatenate(all_even_fits, axis=0)
    even_cps = np.concatenate(all_even_cps, axis=0)
    trajectories_all = np.concatenate(
        [b["trajectories"].numpy() for b in test_loader.dataset], axis=0
    )
    n_knots_all = np.concatenate(all_n_knots, axis=0)

    ic_names = [
        "dx",
        "x",
        "dy",
        "y",
        "dz",
        "z",
        "d_roll",
        "roll",
        "d_pitch",
        "pitch",
        "d_yaw",
        "yaw",
    ]

    def per_sample_rmse(fit, truth):
        residuals = fit - truth
        if residuals.ndim == 3:
            return np.sqrt((residuals**2).mean(axis=(1, 2)))
        else:
            return np.sqrt((residuals**2).mean(axis=1))

    rmse_predicted = per_sample_rmse(jerk_fits, trajectories_all)
    rmse_even = per_sample_rmse(even_fits, trajectories_all)

    print(f"\n--- Fit Quality Summary ---")
    print(
        f"Predicted knots RMSE:    mean={rmse_predicted.mean():.6f}, median={np.median(rmse_predicted):.6f}, std={rmse_predicted.std():.6f}"
    )
    print(
        f"Even-spaced knots RMSE:  mean={rmse_even.mean():.6f}, median={np.median(rmse_even):.6f}, std={rmse_even.std():.6f}"
    )
    print(f"Improvement (even - pred): mean={np.mean(rmse_even - rmse_predicted):.6f}")
    print(f"% samples where predicted is better: {100 * np.mean(rmse_predicted < rmse_even):.1f}%")

    # --- Histogram: RMSE comparison ---
    fig, ax = plt.subplots(figsize=(7, 7))
    bins = np.linspace(0, max(rmse_predicted.max(), rmse_even.max()) * 1.05, 60)
    ax.hist(
        rmse_predicted,
        bins=bins,
        alpha=0.6,
        label=f"Predicted Knots (mean={rmse_predicted.mean():.4f})",
        edgecolor="k",
        color=C_PRED,
    )
    ax.hist(
        rmse_even,
        bins=bins,
        alpha=0.6,
        label=f"Even-Spaced Knots (mean={rmse_even.mean():.4f})",
        edgecolor="k",
        color=C_EVEN,
    )
    ax.set_xlabel("RMSE", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.set_title("Fit Quality: Predicted vs Even-Spaced Knot Placement")
    ax.legend()
    fig.tight_layout()
    fig.savefig("analyses/knots/rmse_histogram.png", dpi=150)
    plt.show()

    # --- Knot quantity histogram ---
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.hist(
        n_knots_all.reshape(-1),
        bins=np.arange(n_knots_all.min(), n_knots_all.max() + 2, 2),
        color="gray",
        alpha=0.5,
        edgecolor="k",
    )
    ax.set_xlabel("Number of Knots", fontsize=14)
    ax.set_ylabel("Count of Trajectories", fontsize=14)
    print(
        f"Knots- mean: {np.mean(n_knots_all)}, min: {np.min(n_knots_all)} max: {np.max(n_knots_all)}, median {np.median(n_knots_all)}"
    )
    plt.savefig("analyses/knots/n_knots_histogram.png")
    plt.show()

    # --- RMSE vs number of knots: per-bin histograms ---
    n_knots_np = n_knots_all.reshape(-1)
    knot_bins = np.arange(n_knots_np.min(), n_knots_np.max() + 2, 2)

    n_bins = len(knot_bins) - 1
    ncols = 4
    nrows = (n_bins + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flat

    for i in range(len(knot_bins) - 1):
        ax = axes[i]
        lo, hi = knot_bins[i], knot_bins[i + 1]
        mask = (n_knots_np >= lo) & (n_knots_np < hi)
        count = mask.sum()
        if count == 0:
            ax.set_visible(False)
            continue

        bins_rmse = np.linspace(0, max(rmse_predicted.max(), rmse_even.max()) * 0.5, 30)
        ax.hist(
            rmse_predicted[mask],
            bins=bins_rmse,
            alpha=0.6,
            label="Predicted",
            edgecolor="k",
            color=C_PRED,
        )
        ax.hist(
            rmse_even[mask],
            bins=bins_rmse,
            alpha=0.6,
            label="Even-Spaced",
            edgecolor="k",
            color=C_EVEN,
        )
        ax.set_title(f"Knots [{lo}, {hi})  (n={count})")
        ax.set_xlabel("RMSE")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("RMSE Distribution by Number of Knots", fontsize=14)
    fig.tight_layout()
    fig.savefig("analyses/knots/rmse_per_knot_bin.png", dpi=150)
    plt.show()

    # --- Mean RMSE std vs n_knots ---
    bin_centers = []
    mean_pred = []
    std_pred = []
    mean_even = []
    std_even = []

    for i in range(len(knot_bins) - 1):
        lo, hi = knot_bins[i], knot_bins[i + 1]
        mask = (n_knots_np >= lo) & (n_knots_np < hi)
        if mask.sum() == 0:
            continue
        bin_centers.append((lo + hi) / 2)
        mean_pred.append(rmse_predicted[mask].mean())
        std_pred.append(rmse_predicted[mask].std())
        mean_even.append(rmse_even[mask].mean())
        std_even.append(rmse_even[mask].std())

    bin_centers = np.array(bin_centers)
    mean_pred = np.array(mean_pred)
    std_pred = np.array(std_pred)
    mean_even = np.array(mean_even)
    std_even = np.array(std_even)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(bin_centers, mean_pred, "-o", color=C_PRED, label="Predicted Knots")
    ax.fill_between(
        bin_centers, mean_pred - std_pred, mean_pred + std_pred, alpha=0.2, color=C_PRED
    )
    ax.plot(bin_centers, mean_even, "-o", color=C_EVEN, label="Even-Spaced Knots")
    ax.fill_between(
        bin_centers, mean_even - std_even, mean_even + std_even, alpha=0.2, color=C_EVEN
    )
    ax.set_xlabel("Number of Knots", fontsize=14)
    ax.set_ylabel("RMSE", fontsize=14)
    ax.set_title("Mean RMSE +/- Std vs Number of Knots", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("analyses/knots/rmse_vs_nknots.png", dpi=150)
    plt.show()

    # --- RMSE diff vs input features ---
    rmse_diff = rmse_even - rmse_predicted

    n_ics = min(len(ic_names), ics.shape[1])
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    for i, ax in enumerate(axes.flat):
        if i >= n_ics:
            ax.set_visible(False)
            continue
        ax.scatter(ics[:, i], rmse_diff, alpha=0.5, s=1, color=C_PRED)
        ax.axhline(0, color=C_TRUTH, linewidth=0.5, linestyle="--")
        ax.set_xlabel(ic_names[i])
        ax.set_ylabel("RMSE(even) - RMSE(pred)")
        ax.set_title(ic_names[i])
    fig.suptitle(
        "RMSE Improvement (Predicted over Even-Spaced) vs Input Features\n(positive = predicted is better)",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig("analyses/knots/rmse_diff_vs_features.png", dpi=150)
    plt.show()

    # --- Example trajectories ---
    best_improvement_idx = np.argmax(rmse_diff)
    median_idx = np.argsort(rmse_diff)[len(rmse_diff) // 2]

    print(
        f"\nPlotting best improvement sample (idx={best_improvement_idx}, diff={rmse_diff[best_improvement_idx]:.5f})"
    )
    plot_example(
        best_improvement_idx,
        trajectories_all,
        jerk_fits,
        even_fits,
        jerk_cps,
        even_cps,
        knot_vectors,
        knot_vectors_even,
        cfg.data.time_horizon,
        cfg.data.spline_degree,
    )
    print(f"Plotting median sample (idx={median_idx}, diff={rmse_diff[median_idx]:.5f})")
    plot_example(
        median_idx,
        trajectories_all,
        jerk_fits,
        even_fits,
        jerk_cps,
        even_cps,
        knot_vectors,
        knot_vectors_even,
        cfg.data.time_horizon,
        cfg.data.spline_degree,
    )

    plot_cp_distance_vs_jerk(
        jerk_cps,
        even_cps,
        trajectories_all,
        jerk_fits,
        knot_vectors,
        knot_vectors_even,
        cfg.data.time_horizon,
        cfg.data.spline_degree,
        jerk_preds,
    )
    print(f"Collected {len(ics)} samples")
    print(jerk_truth.shape, jerk_preds.shape)
    errors = np.abs(jerk_preds - jerk_truth)
    if errors.ndim > 1:
        errors = errors.mean(axis=tuple(range(1, errors.ndim)))

    # --- Error distribution ---
    fig, ax = plt.subplots()
    ax.hist(errors, bins=50, edgecolor="k", alpha=0.7, color=C_PRED)
    ax.set_xlabel("Absolute Error")
    ax.set_ylabel("Count")
    ax.set_title("Prediction Error Distribution")
    fig.savefig("analyses/knots/error_distribution.png", dpi=150)
    plt.show()

    # --- Input feature vs error ---
    n_features = min(ics.shape[1] if ics.ndim > 1 else 1, 8)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, ax in enumerate(axes.flat):
        if i >= n_features:
            ax.set_visible(False)
            continue
        feat = ics[:, i] if ics.ndim > 1 else ics
        ax.scatter(feat, errors, alpha=0.05, s=1, color=C_PRED)
        ax.set_xlabel(f"Feature {i}")
        ax.set_ylabel("Error")
    fig.suptitle("Input Features vs Prediction Error")
    fig.tight_layout()
    fig.savefig("analyses/knots/feature_vs_error.png", dpi=150)
    plt.show()

    # --- Good vs bad predictions ---
    good_mask = errors < np.percentile(errors, 10)
    bad_mask = errors > np.percentile(errors, 90)

    if ics.ndim > 1:
        print("\n--- Feature means: good (p10) vs bad (p90) predictions ---")
        for i in range(min(ics.shape[1], 20)):
            good_mean = ics[good_mask, i].mean()
            bad_mean = ics[bad_mask, i].mean()
            print(
                f"  Feature {i}: good={good_mean:.4f}, bad={bad_mean:.4f}, diff={good_mean - bad_mean:.4f}"
            )

    plt.savefig("analyses/knots/feature_analysis.png")
    plt.show()

    # --- Single sample jerk truth vs prediction ---
    idx = 21
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    axes[0].plot(jerk_truth[idx], label="Truth", color=C_TRUTH)
    axes[0].plot(jerk_preds[idx], label="Prediction", color=C_PRED, linestyle="--")
    axes[0].set_ylabel("Jerk")
    axes[0].legend()
    axes[0].set_title(f"Sample {idx}")

    axes[1].plot(np.abs(jerk_truth[idx] - jerk_preds[idx]), color=C_PRED, label="Abs Error")
    axes[1].set_ylabel("Error")
    axes[1].set_xlabel("Time Step")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig("analyses/knots/single_sample_with_error.png", dpi=150)
    plt.show()

    # --- Jerk activity scatter ---# --- Jerk activity scatter + percent error histogram ---
    fig, ax1 = plt.subplots(1, 1, figsize=(7, 7))

    # Left: scatter
    ax1.scatter(
        jerk_truth.reshape(-1),
        jerk_preds.reshape(-1),
        color="steelblue",
        alpha=0.1,
        s=0.5,
        label="per-sample",
    )
    ax1.scatter(
        jerk_truth.mean(axis=1).reshape(-1),
        jerk_preds.mean(axis=1).reshape(-1),
        color="violet",
        alpha=0.2,
        s=1,
        label="per-trajectory",
    )
    ax1.set_yscale("log")
    ax1.set_xscale("log")
    ax1.set_xlim(1, 201)
    ax1.set_ylim(1, 201)
    ax1.set_xlabel("True Jerk Activity")
    ax1.set_ylabel("Predicted Jerk Activity")
    ax1.set_title("Jerk Activity Prediction")
    ax1.axline((0, 0), (1, 1), linewidth=1, alpha=0.3, linestyle="--", color=C_TRUTH)
    legend = ax1.legend(markerscale=10)

    # Inset: percent error histogram
    ax2 = ax1.inset_axes([0.58, 0.12, 0.4, 0.3])  # [x, y, width, height] in axes coords
    pct_error = (jerk_preds - jerk_truth) / np.clip(np.abs(jerk_truth), 1e-6, None)
    pct_error = pct_error.reshape(-1)

    ax2.hist(
        pct_error,
        bins=np.linspace(-0.5, 0.5, 61),
        color=C_PRED,
        edgecolor="k",
        alpha=0.7,
        linewidth=0.5,
    )
    ax2.axvline(0, color=C_TRUTH, linewidth=1, linestyle="--")
    ax2.axvline(
        np.median(pct_error),
        color="gray",
        linewidth=1.5,
        linestyle="-",
        label=f"Median: {np.median(pct_error):.1f}%",
    )
    ax2.set_xlabel("Relative Error", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.tick_params(labelsize=12)
    ax2.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig("analyses/knots/jerk_activity_truth_v_prediction.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
