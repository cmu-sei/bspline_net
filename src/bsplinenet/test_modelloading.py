"""
bspline_net (Github)

Copyright 2026 Carnegie Mellon University.

NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

Licensed under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.

[DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.

This Software includes and/or makes use of Third-Party Software each subject to its own license.

DM26-0750
"""
#!/usr/bin/env python3
"""Test model loading and forward passes for each training stage.

Usage:
    python scripts/test_model_loading.py [stage]

    stage: knots, cp, joint, or all (default: all)
"""

import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf

from .cp_model import CPDeepONetModel, LitCPModel
from .data import compute_derived_data_config
from .joint_model import LitJointModel
from .knot_model import ActivityModel, LitKnotModel


def get_test_config():
    """Create test configuration."""
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "data": {
                "batch_size": 8,
                "batches_per_epoch": 3,
                "val_batches": 2,
                "min_radius": 0.001,
                "time_horizon": 2.5,
                "sample_interval": 0.005,
                "spline_degree": 3,
                "max_density": 40,  # knots per second
                "min_density": 2,  # knots per second
                "n_dim": 12,
                "max_x": 2.0,
                "max_dx": 2.0,
                "max_ag": 0.7854,
                "max_da": 1.5708,
                "uniform_knot_prob": 0.3,
                "num_workers": 0,
                "max_translational_rate": 5.0,
                "max_roll_pitch_rate": 12,
                "max_yaw_rate": 6,
            },
            "model": {
                "knots": {
                    "hidden": 64,
                    "kernel_1": 5,
                    "kernel_2": 5,
                    "lr": 0.01,
                    "max_error": 0.01,
                },
                "cp": {
                    "widths_ic": [64, 128],
                    "widths_kcp": [128, 64],
                    "rank": 32,
                    "metapos": 8,
                    "lr": 0.0001,
                    "trunk_recon_weight": 0.1,
                    "trunk_smooth_weight": 0.1,
                    "trunk_ortho_weight": 0.01,
                    "branch_smooth_weight": 0.1,
                    "ic_loss_weight": 0.1,
                },
                "joint": {
                    "knot_lr": 0.005,
                    "cp_lr": 0.0001,
                    "activity_loss_weight": 1.0,
                },
            },
        }
    )
    return compute_derived_data_config(cfg)


def create_sample_batch(cfg):
    """Create a sample batch for testing."""
    batch_size = cfg.data.batch_size
    n_samples = cfg.data.n_samples
    n_dim = len(cfg.data.R)
    n_interior_knots = cfg.data.n_interior_knots
    spline_degree = cfg.data.spline_degree

    ics = torch.randn(batch_size, n_dim)
    trajectories = torch.randn(batch_size, n_samples, n_dim)
    R_tensor = torch.tensor(cfg.data.R, dtype=torch.float32)

    # Create valid knot vector
    knot_positions = torch.rand(batch_size, n_interior_knots).sort(dim=1).values
    zeros = torch.zeros(batch_size, spline_degree + 1)
    ones = torch.ones(batch_size, spline_degree + 1)
    knot_vector = torch.cat([zeros, knot_positions, ones], dim=1)

    return {
        "ics": ics,
        "trajectories": trajectories,
        "norm_ics": ics / R_tensor,
        "norm_trajectories": trajectories / R_tensor,
        "jerk_activity_max": torch.rand(batch_size, n_samples),
        "weighting_vector": torch.ones(batch_size, n_samples),
        "knot_positions": knot_positions,
        "knot_vector": knot_vector,
    }


def print_tensor_info(name: str, tensor: torch.Tensor, indent: int = 2):
    """Print tensor information."""
    prefix = " " * indent
    print(f"{prefix}{name}:")
    print(f"{prefix}  shape: {tuple(tensor.shape)}")
    print(f"{prefix}  dtype: {tensor.dtype}")
    print(f"{prefix}  min:   {tensor.min().item():.6f}")
    print(f"{prefix}  max:   {tensor.max().item():.6f}")
    print(f"{prefix}  mean:  {tensor.mean().item():.6f}")
    if tensor.requires_grad:
        print(f"{prefix}  requires_grad: True")


def test_knot_model(cfg, batch):
    """Test knot model loading and forward pass."""
    print("\n" + "=" * 60)
    print("Testing: Knot Model (ActivityModel + LitKnotModel)")
    print("=" * 60)

    # Test raw model
    print("\n--- ActivityModel (raw) ---")
    model = ActivityModel(
        n_dim=len(cfg.data.R),
        n_samples=cfg.data.n_samples,
        hidden=cfg.model.knots.hidden,
        kernel_1=cfg.model.knots.kernel_1,
        kernel_2=cfg.model.knots.kernel_2,
    )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    output = model(batch["norm_ics"])
    print_tensor_info("Output (activity)", output)

    # Test Lightning wrapper
    print("\n--- LitKnotModel (Lightning) ---")
    lit_model = LitKnotModel(cfg)

    # Forward
    activity = lit_model(batch["norm_ics"])
    print_tensor_info("Activity Prediction", activity)

    # Predict knots
    jerk_activity, knot_positions, knot_vector = lit_model.predict_knots(batch["norm_ics"])
    print_tensor_info("Knot positions", knot_positions)
    print_tensor_info("Knot vector", knot_vector)

    # Training step
    loss = lit_model.training_step(batch, batch_idx=0)
    print(f"\n  Training loss: {loss.item():.6f}")

    # Backward
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(lit_model.parameters(), float("inf"))
    print(f"  Gradient norm: {grad_norm.item():.6f}")

    print("\n Knot model tests passed!")
    return True


def test_cp_model(cfg, batch):
    """Test control point model loading and forward pass."""
    print("\n" + "=" * 60)
    print("Testing: CP Model (CPDeepONetModel + LitCPModel)")
    print("=" * 60)

    # Test raw model
    print("\n--- CPDeepONetModel (raw) ---")
    model = CPDeepONetModel(
        n_dim=len(cfg.data.R),
        n_cp_max=cfg.data.n_cp_max,
        spline_degree=cfg.data.spline_degree,
        widths_ic=cfg.model.cp.widths_ic,
        widths_kcp=cfg.model.cp.widths_kcp,
        rank=cfg.model.cp.rank,
        metapos=cfg.model.cp.metapos,
    )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    R = torch.tensor(cfg.data.R, dtype=torch.float32)
    pred_cp, aux_losses = model(batch["knot_vector"], batch["norm_ics"], R)
    print_tensor_info("Output (control points)", pred_cp)

    print("\n  Auxiliary losses:")
    for name, value in aux_losses.items():
        print(f"    {name}: {value.item():.6f}")

    # Test Lightning wrapper
    print("\n--- LitCPModel (Lightning) ---")
    lit_model = LitCPModel(cfg)

    # Forward
    pred_cp, aux_losses = lit_model(batch["knot_vector"], batch["norm_ics"])
    print_tensor_info("Control points", pred_cp)

    # Training step
    loss = lit_model.training_step(batch, batch_idx=0)
    print(f"\n  Training loss: {loss.item():.6f}")

    # Backward
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(lit_model.parameters(), float("inf"))
    print(f"  Gradient norm: {grad_norm.item():.6f}")

    print("\n CP model tests passed!")
    return True


def test_joint_model(cfg, batch):
    """Test joint model loading and forward pass."""
    print("\n" + "=" * 60)
    print("Testing: Joint Model (LitJointModel)")
    print("=" * 60)

    print("\n--- LitJointModel ---")
    model = LitJointModel(cfg)

    knot_params = sum(p.numel() for p in model.knot_model.parameters())
    cp_params = sum(p.numel() for p in model.cp_model.parameters())
    print(f"  Knot model parameters: {knot_params:,}")
    print(f"  CP model parameters: {cp_params:,}")
    print(f"  Total parameters: {knot_params + cp_params:,}")

    # Forward
    activity, knot_vector, pred_cp, aux_losses = model(batch["norm_ics"])
    print_tensor_info("activity", activity)
    print_tensor_info("Knot vector", knot_vector)
    print_tensor_info("Control points", pred_cp)

    print("\n  Auxiliary losses:")
    for name, value in aux_losses.items():
        print(f"    {name}: {value.item():.6f}")

    # Training step
    loss = model.training_step(batch, batch_idx=0)
    print(f"\n  Training loss: {loss.item():.6f}")

    # Backward and check both models get gradients
    loss.backward()

    knot_grad_norm = torch.nn.utils.clip_grad_norm_(model.knot_model.parameters(), float("inf"))
    cp_grad_norm = torch.nn.utils.clip_grad_norm_(model.cp_model.parameters(), float("inf"))
    print(f"  Knot model gradient norm: {knot_grad_norm.item():.6f}")
    print(f"  CP model gradient norm: {cp_grad_norm.item():.6f}")

    # Check optimizer has separate param groups
    optimizer = model.configure_optimizers()
    print(f"\n  Optimizer param groups: {len(optimizer.param_groups)}")
    print(f"    Group 0 (knot) LR: {optimizer.param_groups[0]['lr']}")
    print(f"    Group 1 (cp) LR: {optimizer.param_groups[1]['lr']}")

    print("\n Joint model tests passed!")
    
    return True


def main():
    stages = ["knots", "cp", "joint"]

    # Parse command line arg
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "all":
            pass
        elif arg in stages:
            stages = [arg]
        else:
            print(f"Unknown stage: {arg}")
            print(f"Valid options: {stages + ['all']}")
            sys.exit(1)

    # Setup
    cfg = get_test_config()
    batch = create_sample_batch(cfg)

    print("=" * 60)
    print("Model Loading Test")
    print("=" * 60)
    print(f"\nConfig:")
    print(f"  batch_size: {cfg.data.batch_size}")
    print(f"  n_samples: {cfg.data.n_samples}")
    print(f"  n_dim: {len(cfg.data.R)}")
    print(f"  n_interior_knots: {cfg.data.n_interior_knots}")
    print(f"  spline_degree: {cfg.data.spline_degree}")

    # Test each stage
    results = {}
    test_funcs = {
        "knots": test_knot_model,
        "cp": test_cp_model,
        "joint": test_joint_model,
    }

    for stage in stages:
        try:
            # Create fresh batch for each test (gradients get cleared)
            batch = create_sample_batch(cfg)
            results[stage] = test_funcs[stage](cfg, batch)
        except Exception as e:
            print(f"\n {stage} failed: {e}")
            import traceback

            traceback.print_exc()
            results[stage] = False

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for stage, passed in results.items():
        status = " PASS" if passed else " FAIL"
        print(f"  {stage}: {status}")

    if all(results.values()):
        print("\nAll tests passed!")
        sys.exit(0)
    else:
        print("\nSome tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
