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
"""Test data loading for each training stage.

Usage:
    python scripts/test_dataloading.py [stage]

    stage: knots, cp, joint, or all (default: all)
"""

import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf

from bsplinenet.data import compute_derived_data_config, get_datamodule


def mock_gen_data(batch_size: int, min_radius: float):
    """Mock data generation for testing."""
    n_dim = 12
    n_samples = 501

    # Generate random ICs in a shell
    directions = torch.randn(batch_size, n_dim)
    directions = directions / directions.norm(dim=1, keepdim=True)
    radii = min_radius + (1 - min_radius) * torch.rand(batch_size, 1)
    ics = directions * radii

    # Generate fake trajectories (random walk from IC)
    noise = torch.randn(batch_size, n_samples, n_dim) * 0.01
    trajectories = ics.unsqueeze(1) + noise.cumsum(dim=1)

    return ics, trajectories


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
                "max_density": 40,  # knots/second
                "min_density": 2,  # knots/second
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
        }
    )
    return compute_derived_data_config(cfg)


def print_batch_info(batch: dict, stage: str):
    """Print information about a batch."""
    print(f"\n{'='*60}")
    print(f"Stage: {stage}")
    print(f"{'='*60}")
    print(f"Keys: {list(batch.keys())}")
    print()

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}:")
            print(f"    shape: {tuple(value.shape)}")
            print(f"    dtype: {value.dtype}")
            print(f"    min:   {value.min().item():.4f}")
            print(f"    max:   {value.max().item():.4f}")
            print(f"    mean:  {value.mean().item():.4f}")
        else:
            print(f"  {key}: {value}")

    print()


def test_stage(stage: str, cfg):
    """Test a single stage."""
    print(f"\nLoading datamodule for stage: {stage}")

    try:
        dm = get_datamodule(stage, cfg)

        # Test train dataloader
        print(f"\n--- Train DataLoader ---")
        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))
        print_batch_info(batch, f"{stage} (train)")

        # Test val dataloader
        print(f"\n--- Val DataLoader ---")
        val_loader = dm.val_dataloader()
        batch = next(iter(val_loader))
        print_batch_info(batch, f"{stage} (val)")

        # Stage-specific checks
        if stage == "knots":
            assert "jerk_activity_max" in batch, "Missing jerk_activity_max for knots stage"
            print(" jerk_activity_max present")

        elif stage == "cp":
            assert "knot_vector" in batch, "Missing knot_vector for cp stage"
            assert "knot_positions" in batch, "Missing knot_positions for cp stage"

            # Check knot vector structure
            kv = batch["knot_vector"]
            d = cfg.data.spline_degree
            assert (kv[:, : d + 1] == 0).all(), "Knot vector should start with d+1 zeros"
            assert (kv[:, -(d + 1) :] == 1).all(), "Knot vector should end with d+1 ones"
            print(" knot_vector present and valid")
            print(" knot_positions present")

        elif stage == "joint":
            assert "jerk_activity_max" in batch, "Missing jerk_activity_max for joint stage"
            print(" jerk_activity_max present")

        print(f"\n Stage '{stage}' passed all checks!")
        return True

    except Exception as e:
        print(f"\n Stage '{stage}' failed: {e}")
        import traceback

        traceback.print_exc()
        return False


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

    print("=" * 60)
    print("Data Loading Test")
    print("=" * 60)
    print(f"\nConfig:")
    print(f"  batch_size: {cfg.data.batch_size}")
    print(f"  n_samples: {cfg.data.n_samples}")
    print(f"  n_interior_knots: {cfg.data.n_interior_knots}")
    print(f"  spline_degree: {cfg.data.spline_degree}")

    # Test each stage
    results = {}
    for stage in stages:
        results[stage] = test_stage(stage, cfg)

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
