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
traj_cp_analysis.py
-------------------
Visualise and inspect the learned control-point (CP) embeddings produced by a
trained CP-DeepONet model.

The script is an analysis entry point that:
* loads a checkpoint of a ``LitCPModel`` (the CP-DeepONet),
* adapts the checkpoint weights so that the model can be used with the
  current PyTorch-Lightning implementation (e.g., reshaping linear -> conv
  weights),
* creates a data module for the requested stage (train/val/test) using the
  Hydra-provided configuration,
* extracts the IC- and CP-embeddings for a small batch of trajectories,
* and finally plots a few representative embedding maps (saved as
  ``logs/cp_embedding.png``).

Running the script
------------------
Because the file uses Hydra for configuration, invoke it as a module:

```bash
python -m bsplinenet.analysis.traj_cp_analysis \
    config_path=../../configs \
    config_name=cp_analysis \
    +ckpt_path=path/to/checkpoint.ckpt \
    +stage=test
```
"""
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from ..cp_model import CPDeepONetModel, LitCPModel
from ..data import CPDataModule, compute_derived_data_config, get_datamodule
from ..utils import compute_controlpoints_from_trajectory

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


@hydra.main(config_path="../../configs", config_name="cp_analysis", version_base=None)
def main(cfg: DictConfig):
    print(cfg)
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

    state_dict = checkpoint["state_dict"]
    # Layer 0: Linear(5, 64) -> Conv1d(1, 64, kernel_size=5)
    # Weight: (64, 5) -> (64, 1, 5)  just add the input channel dim
    w = state_dict["model.cp_local.0.weight"]
    state_dict["model.cp_local.0.weight"] = w.unsqueeze(1)

    # Layers 2, 4: Linear -> Conv1d(kernel_size=1)
    # Weight: (out, in) -> (out, in, 1)
    for key in [
        "model.cp_local.2.weight",
        "model.cp_local.4.weight",
        "model.cp_local.6.weight",
        "model.cp_local.8.weight",
        "model.cp_local.10.weight",
        "model.cp_local.12.weight",
    ]:
        state_dict[key] = state_dict[key].unsqueeze(-1)

    model = LitCPModel(**checkpoint["hyper_parameters"])
    model.load_state_dict(state_dict)
    model.eval()
    model.freeze()

    model = model.to(device)

    # --- Set up data ---
    # Get datamodule
    datamodule = get_datamodule(cfg.stage, cfg)
    test_loader = datamodule.test_dataloader()

    with torch.no_grad():
        for b, batch in enumerate(test_loader):
            print(f"\rbatch {b} ", end="")
            trajectories = batch["trajectories"]
            norm_ics = batch["norm_ics"]
            ics = batch["ics"]
            knot_vector = batch["knot_vector"]
            knot_positions = batch["knot_positions"]

            ics_embeddings = model.model.to("cpu")._forward_ic(norm_ics)
            cp_embeddings, _ = model.model.to("cpu")._forward_cp(knot_vector)

            print(ics_embeddings.shape)
            print(cp_embeddings.shape)
            print(knot_positions[0])
            break

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    g, idx0 = torch.sort(cp_embeddings[0, 0, :].detach())
    g[-1] = 0
    g = g.reshape(64, 32)
    g = g.numpy()

    h = cp_embeddings[0, 1, :].detach()
    h[-1] = 0
    h = h[idx0]
    h = h.reshape(64, 32)
    h = h.numpy()

    ax[0].imshow(ics_embeddings[0, idx0, 0].detach().numpy().reshape(64, 32))
    ax[1].imshow(g)
    ax[2].imshow(h)
    plt.show()
    plt.savefig("logs/cp_embedding.png") #kludge


if __name__ == "__main__":
    main()
