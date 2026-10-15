"""
bspline_net (Github)

Copyright 2026 Carnegie Mellon University.

NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

Licensed under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.

[DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.

This Software includes and/or makes use of Third-Party Software each subject to its own license.

DM26-0750
"""
import numpy as np
import torch
from matplotlib import pyplot as plt


def plot(
    name,
    epoch,
    trajectories,
    spacing_pred,
    spacing_comp,
    knot_positions,
    d,
    n_dim,
    batch_size,
    t_eval,
    spacing_loss,
    knot_vector=None,
    predicted_controlpoints=None,
    pred_trajectories=None,
    has_controlpoints=False,
):
    """Plot training diagnostics."""

    ics = trajectories[:, 0, :]
    idx = np.random.randint(batch_size)
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    kp = knot_positions[idx].detach().numpy()

    for i in range(n_dim):
        ax = axes.flatten()[i]
        ax.plot(t_eval, trajectories[idx, :, i].numpy(), "k-", linewidth=2, label="True")

        if has_controlpoints:
            knots_np = knot_vector[idx].detach().numpy()
            n_cp_eff = (knot_positions[idx] < 0.99).sum().item() + d + 1
            cp_positions = np.array([knots_np[j + 1 : j + d + 1].mean() for j in range(n_cp_eff)])
            cps_pred = predicted_controlpoints[idx, :n_cp_eff, i].detach().numpy()

            ax.scatter(
                cp_positions,
                cps_pred,
                c="orange",
                s=40,
                zorder=5,
                marker="x",
                label="Pred CPs",
            )
            ax.plot(
                t_eval,
                pred_trajectories[idx, :, i].detach().numpy(),
                "orange",
                linestyle="--",
                linewidth=1.5,
                label="Pred Traj",
            )

        for k in kp:
            if k < 0.99:
                ax.axvline(k, color="g", alpha=0.3, linestyle=":")

        ax.set_title(f"Dim {i}")
        if i == 0:
            ax.legend(fontsize=6)

    ax = axes.flatten()[12]
    ax.plot(t_eval, spacing_pred[idx].detach().numpy(), "b-", label="Predicted Spacing")
    ax.plot(t_eval, spacing_comp[idx].detach().numpy(), "r--", label="Computed Spacing")
    for k in kp:
        if k < 0.99:
            ax.axvline(k, color="g", alpha=0.5, linestyle=":")
    ax.set_title("Spacing")
    ax.legend(fontsize=6)

    ax = axes.flatten()[13]
    for i in range(min(batch_size, 50)):
        kp_i = knot_positions[i].detach().numpy()
        active = kp_i[kp_i < 0.99]
        ax.scatter(active, np.ones_like(active) * i, s=5)
    ax.set_xlim(-0.05, 1.05)
    ax.set_title("Knot positions (batch)")

    ax = axes.flatten()[14]
    ax.imshow(spacing_pred.detach().numpy(), aspect="auto", cmap="viridis")
    ax.set_title("Signal (batch)")

    ax = axes.flatten()[15]
    info_text = f"Epoch {epoch}\nSpacingLoss: {spacing_loss:.5f}\nSample idx: {idx}"

    # Add IC loss info if control points and ICs are available
    if has_controlpoints and ics is not None:
        pred_ic = predicted_controlpoints[:, 0, :].detach()
        true_ic = ics.detach()

        # Relative IC loss for selected sample: ((pred - true) / true)^2
        # Avoid division by zero
        rel_ic_sample = (((pred_ic[idx] - true_ic[idx])) ** 2).mean().item()

        # Relative IC loss for batch
        rel_ic_batch = (((pred_ic - true_ic)) ** 2).mean().item()

        info_text += f"\nRel IC (sample): {rel_ic_sample:.5f}"
        info_text += f"\nRel IC (batch): {rel_ic_batch:.5f}"
        # breakpoint()

    ax.text(0.5, 0.5, info_text, ha="center", va="center", fontsize=11)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"scripts/{name}_e_{epoch:04d}.png")
    plt.close()
