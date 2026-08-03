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
Knot-Placement Analysis for B-Spline Trajectories
=================================================

This script evaluates the quality of the knot-placement predictions made by a
trained ``LitKnotModel``.  It loads a checkpoint, runs inference on a test
datamodule, reconstructs the corresponding B-splines, and produces a suite of
diagnostic visualisations:

* RMSE and R^2 comparison between *predicted* and *even-spaced* knot placements.
* Histograms of RMSE, number of knots, and per-bin RMSE distributions.
* Scatter plots of input initial conditions vs. RMSE improvement.
* Control-point distance vs. predicted jerk-activity analysis.
* Example trajectory visualisations (truth, predicted fit, even-spaced fit,
  control points, residuals).
* Jerk-activity prediction vs. ground-truth scatter with relative-error inset.

All figures are saved under ``analyses/knots/`` (or the path you specify in the
Hydra configuration) and are also displayed interactively via ``matplotlib``.

Typical usage
-------------
Run the script with Hydra configuration overrides, e.g.:

```bash
python src/bsplinenet/analysis/knot_analysis.py \
    ckpt_path=path/to/checkpoint.ckpt \
    data.batch_size=1000 \
    data.test_batches=10 \
    model.knots.max_error=0.001
```
"""

#!/usr/bin/env python3
import hydra
import numpy as np
import torch
from matplotlib import pyplot as plt
from omegaconf import DictConfig

from .data import BaseTrajectoryDataset, compute_derived_data_config
from .utils import build_knot_vector, compute_activity, density_to_knots


@hydra.main(version_base=None, config_path="../configs/data", config_name="default")
def main(cfg: DictConfig, max_error=0.01):
    """Debug via visualiztion function.
    """
    # Compute derived config values
    cfg = compute_derived_data_config(cfg, isdatacfg=True)
    dataset = BaseTrajectoryDataset(cfg, batches_per_epoch=1)
    ## data = dataset._generate_epoch_data()
    data = dataset._generate_epoch_data(dataset.epoch_size)

    initial_conditions = data["ics"][0:1]  # shape ( 1, ndim)

    trajectory = data["trajectories"][0:1]  # shape ( 1, n_samples, ndim)
    time = np.linspace(0, 1, cfg.n_samples)
    activity, deriv2, trajectory_with_padding = compute_activity(
        trajectory,
        time_horizon=cfg.time_horizon,
        return_2nd_deriv=True,
        max_translational_rate=cfg.max_translational_rate,
        max_roll_pitch_rate=cfg.max_roll_pitch_rate,
        max_yaw_rate=cfg.max_yaw_rate,
    )  # shape (1, n_samples)
    theory_density = torch.sqrt(activity / 8 / max_error)

    # knot positions are on normalized scale
    knot_positions, _ = density_to_knots(
        theory_density, cfg.max_density, cfg.min_density, cfg.time_horizon
    )
    #knot_vector = build_knot_vector(knot_positions, cfg.spline_degree)

    _, ax = plt.subplots(4, 6, figsize=(50, 30))
    ax = ax.reshape(-1)
    for i in range(cfg.n_dim // 2):
        # plot initial condition, trajectory for this axis.
        ax[i].plot(time[0:1], initial_conditions[0, 2 * i], "o", label="IC")
        ax[i].plot(time, trajectory[0, :, 2 * i], label="Trajectory")
        # draw lines where the knots would go
        ax[i].vlines(
            knot_positions, ymin=trajectory[0, :, 2 * i].min(), ymax=trajectory[0, :, 2 * i].max()
        )
        ax[i].legend()

    for j in range(deriv2.shape[2]):

        ax[j + i + 1].plot(trajectory_with_padding[0, j, :].diff().diff())

    for k in range(deriv2.shape[2]):

        ax[k + j + i + 2].scatter(time, deriv2[0, :, k])
        ax[k + j + i + 2].plot(time, deriv2[0, :, k])

    ax[k + j + i + 3].plot(time, activity[0])
    ax[k + j + i + 3].plot(time, theory_density[0])

    plt.savefig("logs/knot_placement.png")

    # gen_data(batch_size = 1, min_radius=0.5, n_dim=cfg.data.n_dim, A=A, tt=tt)


if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
