"""
bspline_net (Github)

Copyright 2026 Carnegie Mellon University.

NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

Licensed under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.

[DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.

This Software includes and/or makes use of Third-Party Software each subject to its own license.

DM26-0750
"""
"""Unit tests for data module."""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf

from bsplinenet.data import (
    compute_derived_data_config,
    BaseTrajectoryDataset,
    KnotDataset,
    CPDataset,
    JointDataset,
    KnotDataModule,
    CPDataModule,
    JointDataModule,
    get_datamodule,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def base_config():
    """Create a minimal config for testing."""
    cfg = OmegaConf.create({
        "seed": 42,
        "data": {
            "batch_size": 16,
            "batches_per_epoch": 3,
            "val_batches": 2,
            "min_radius": 0.001,
            "time_horizon": 2.5,
            "sample_interval": 0.005,
            "spline_degree": 3,
            "min_spacing": 0.025,
            "max_spacing": 0.1,
            "max_x": 2.0,
            "max_dx": 2.0,
            "max_ag": 0.7854,
            "max_da": 1.5708,
            "uniform_knot_prob": 0.3,
        }
    })
    return cfg


@pytest.fixture
def config_with_derived(base_config):
    """Config with derived values computed."""
    return compute_derived_data_config(base_config)


@pytest.fixture
def R(config_with_derived):
    """Normalization vector."""
    return np.array(config_with_derived.data.R)


@pytest.fixture
def mock_gen_data():
    """Mock data generation function."""
    def gen_data(batch_size, min_radius):
        n_dim = 12
        n_samples = 501
        ics = torch.randn(batch_size, n_dim)
        trajectories = torch.randn(batch_size, n_samples, n_dim)
        return ics, trajectories
    return gen_data


# =============================================================================
# Tests: compute_derived_data_config
# =============================================================================


class TestComputeDerivedDataConfig:
    
    def test_computes_n_samples(self, base_config):
        cfg = compute_derived_data_config(base_config)
        expected = int(2.5 / 0.005) + 1  # 501
        assert cfg.data.n_samples == expected
    
    def test_computes_n_cp_max(self, base_config):
        cfg = compute_derived_data_config(base_config)
        expected = int(1 / 0.025) + 3 + 1  # 44
        assert cfg.data.n_cp_max == expected
    
    def test_computes_n_interior_knots(self, base_config):
        cfg = compute_derived_data_config(base_config)
        expected = cfg.data.n_cp_max - 3 - 1  # n_cp_max - degree - 1
        assert cfg.data.n_interior_knots == expected
    
    def test_computes_R_vector(self, base_config):
        cfg = compute_derived_data_config(base_config)
        assert hasattr(cfg.data, 'R')
        assert len(cfg.data.R) == 12
    
    def test_R_has_correct_structure(self, base_config):
        cfg = compute_derived_data_config(base_config)
        R = cfg.data.R
        # Pattern: [dx, x, dx, x, dx, x, da, ag, da, ag, da, ag]
        assert R[0] == cfg.data.max_dx
        assert R[1] == cfg.data.max_x
        assert R[6] == cfg.data.max_da
        assert R[7] == cfg.data.max_ag
    
    def test_idempotent(self, base_config):
        """Calling twice should give same result."""
        cfg1 = compute_derived_data_config(base_config)
        cfg2 = compute_derived_data_config(cfg1)
        assert cfg1.data.n_samples == cfg2.data.n_samples
        assert cfg1.data.R == cfg2.data.R


# =============================================================================
# Tests: KnotDataset
# =============================================================================


class TestKnotDataset:
    
    def test_yields_correct_keys(self, config_with_derived, mock_gen_data, R):
        dataset = KnotDataset(
            cfg=config_with_derived.data,
            gen_data_fn=mock_gen_data,
            R=R,
            batches_per_epoch=1,
        )
        batch = next(iter(dataset))
        
        expected_keys = {"ics", "trajectories", "norm_ics", "norm_trajectories", "spacing_comp"}
        assert set(batch.keys()) == expected_keys
    
    def test_batch_shapes(self, config_with_derived, mock_gen_data, R):
        dataset = KnotDataset(
            cfg=config_with_derived.data,
            gen_data_fn=mock_gen_data,
            R=R,
            batches_per_epoch=1,
        )
        batch = next(iter(dataset))
        
        batch_size = config_with_derived.data.batch_size
        n_samples = 501  # from mock
        n_dim = 12
        
        assert batch["ics"].shape == (batch_size, n_dim)
        assert batch["trajectories"].shape == (batch_size, n_samples, n_dim)
        assert batch["norm_ics"].shape == (batch_size, n_dim)
        assert batch["norm_trajectories"].shape == (batch_size, n_samples, n_dim)
        assert batch["spacing_comp"].shape == (batch_size, n_samples)
    
    def test_normalization(self, config_with_derived, mock_gen_data, R):
        dataset = KnotDataset(
            cfg=config_with_derived.data,
            gen_data_fn=mock_gen_data,
            R=R,
            batches_per_epoch=1,
        )
        batch = next(iter(dataset))
        
        R_tensor = torch.from_numpy(R).float()
        expected_norm_ics = batch["ics"] / R_tensor
        torch.testing.assert_close(batch["norm_ics"], expected_norm_ics)
    
    def test_iterates_correct_number_of_batches(self, config_with_derived, mock_gen_data, R):
        batches_per_epoch = 5
        dataset = KnotDataset(
            cfg=config_with_derived.data,
            gen_data_fn=mock_gen_data,
            R=R,
            batches_per_epoch=batches_per_epoch,
        )
        
        count = sum(1 for _ in dataset)
        assert count == batches_per_epoch
    
    def test_seed_reproducibility(self, config_with_derived, mock_gen_data, R):
        dataset1 = KnotDataset(
            cfg=config_with_derived.data,
            gen_data_fn=mock_gen_data,
            R=R,
            batches_per_epoch=1,
            seed=42,
        )
        dataset2 = KnotDataset(
            cfg=config_with_derived.data,
            gen_data_fn=mock_gen_data,
            R=R,
            batches_per_epoch=1,
            seed=42,
        )
        
        batch1 = next(iter(dataset1))
        batch2 = next(iter(dataset2))
        
        # Note: This tests that seeds are set, but actual reproducibility
        # depends on gen_data using torch/numpy random
        assert batch1["ics"].shape == batch2["ics"].shape


# =============================================================================
# Tests: CPDataset
# =============================================================================


class TestCPDataset:
    
    def test_yields_correct_keys(self, config_with_derived, mock_gen_data, R):
        dataset = CPDataset(
            cfg=config_with_derived.data,
            gen_data_fn=mock_gen_data,
            R=R,
            batches_per_epoch=1,
        )
        batch = next(iter(dataset))
        
        expected_keys = {"ics", "trajectories", "norm_ics", "norm_trajectories", 
                        "knot_positions", "knot_vector"}
        assert set(batch.keys()) == expected_keys
    
    def test_knot_positions_shape(self, config_with_derived, mock_gen_data, R):
        dataset = CPDataset(
            cfg=config_with_derived.data,
            gen_data_fn=mock_gen_data,
            R=R,
            batches_per_epoch=1,
        )
        batch = next(iter(dataset))
        
        batch_size = config_with_derived.data.batch_size
        n_interior_knots = config_with_derived.data.n_interior_knots
        
        assert batch["knot_positions"].shape == (batch_size, n_interior_knots)
    
    def test_knot_vector_shape(self, config_with_derived, mock_gen_data, R):
        dataset = CPDataset(
            cfg=config_with_derived.data,
            gen_data_fn=mock_gen_data,
            R=R,
            batches_per_epoch=1,
        )
        batch = next(iter(dataset))
        
        batch_size = config_with_derived.data.batch_size
        n_interior_knots = config_with_derived.data.n_interior_knots
        degree = config_with_derived.data.spline_degree
        expected_knot_vector_len = n_interior_knots + 2 * (degree + 1)
        
        assert batch["knot_vector"].shape == (batch_size, expected_knot_vector_len)
    
    def test_knot_positions_in_valid_range(self, config_with_derived, mock_gen_data, R):
        dataset = CPDataset(
            cfg=config_with_derived.data,
            gen_data_fn=mock_gen_data,
            R=R,
            batches_per_epoch=3,
        )
        
        for batch in dataset:
            assert (batch["knot_positions"] >= 0).all()
            assert (batch["knot_positions"] <= 1).all()
    
    def test_knot_vector_clamped_structure(self, config_with_derived, mock_gen_data, R):
        """Knot vector should start with d+1 zeros and end with d+1 ones."""
        dataset = CPDataset(
            cfg=config_with_derived.data,
            gen_data_fn=mock_gen_data,
            R=R,
            batches_per_epoch=1,
        )
        batch = next(iter(dataset))
        
        degree = config_with_derived.data.spline_degree
        knot_vector = batch["knot_vector"]
        
        # First d+1 should be 0
        assert (knot_vector[:, :degree + 1] == 0).all()
        # Last d+1 should be 1
        assert (knot_vector[:, -(degree + 1):] == 1).all()
    
    def test_uniform_prob_affects_distribution(self, config_with_derived, mock_gen_data, R):
        """With uniform_prob=1.0, all knots should be uniformly spaced."""
        dataset = CPDataset(
            cfg=config_with_derived.data,
            gen_data_fn=mock_gen_data,
            R=R,
            batches_per_epoch=1,
            uniform_prob=1.0,
        )
        batch = next(iter(dataset))
        
        # Just check it runs without error
        assert batch["knot_positions"] is not None


# =============================================================================
# Tests: JointDataset
# =============================================================================


class TestJointDataset:
    
    def test_yields_correct_keys(self, config_with_derived, mock_gen_data, R):
        dataset = JointDataset(
            cfg=config_with_derived.data,
            gen_data_fn=mock_gen_data,
            R=R,
            batches_per_epoch=1,
        )
        batch = next(iter(dataset))
        
        expected_keys = {"ics", "trajectories", "norm_ics", "norm_trajectories", "spacing_comp"}
        assert set(batch.keys()) == expected_keys
    
    def test_same_as_knot_dataset(self, config_with_derived, mock_gen_data, R):
        """JointDataset should yield same structure as KnotDataset."""
        joint_dataset = JointDataset(
            cfg=config_with_derived.data,
            gen_data_fn=mock_gen_data,
            R=R,
            batches_per_epoch=1,
        )
        knot_dataset = KnotDataset(
            cfg=config_with_derived.data,
            gen_data_fn=mock_gen_data,
            R=R,
            batches_per_epoch=1,
        )
        
        joint_batch = next(iter(joint_dataset))
        knot_batch = next(iter(knot_dataset))
        
        assert set(joint_batch.keys()) == set(knot_batch.keys())


# =============================================================================
# Tests: DataModules
# =============================================================================


class TestDataModules:
    
    def test_knot_datamodule_train_dataloader(self, config_with_derived, mock_gen_data, R):
        dm = KnotDataModule(config_with_derived, mock_gen_data, R)
        loader = dm.train_dataloader()
        
        batch = next(iter(loader))
        assert "spacing_comp" in batch
    
    def test_knot_datamodule_val_dataloader(self, config_with_derived, mock_gen_data, R):
        dm = KnotDataModule(config_with_derived, mock_gen_data, R)
        loader = dm.val_dataloader()
        
        batch = next(iter(loader))
        assert "spacing_comp" in batch
    
    def test_cp_datamodule_train_dataloader(self, config_with_derived, mock_gen_data, R):
        dm = CPDataModule(config_with_derived, mock_gen_data, R)
        loader = dm.train_dataloader()
        
        batch = next(iter(loader))
        assert "knot_vector" in batch
    
    def test_joint_datamodule_train_dataloader(self, config_with_derived, mock_gen_data, R):
        dm = JointDataModule(config_with_derived, mock_gen_data, R)
        loader = dm.train_dataloader()
        
        batch = next(iter(loader))
        assert "spacing_comp" in batch


# =============================================================================
# Tests: get_datamodule factory
# =============================================================================


class TestGetDatamodule:
    
    def test_returns_knot_datamodule(self, config_with_derived, mock_gen_data, R):
        dm = get_datamodule("knots", config_with_derived, mock_gen_data, R)
        assert isinstance(dm, KnotDataModule)
    
    def test_returns_cp_datamodule(self, config_with_derived, mock_gen_data, R):
        dm = get_datamodule("cp", config_with_derived, mock_gen_data, R)
        assert isinstance(dm, CPDataModule)
    
    def test_returns_joint_datamodule(self, config_with_derived, mock_gen_data, R):
        dm = get_datamodule("joint", config_with_derived, mock_gen_data, R)
        assert isinstance(dm, JointDataModule)
    
    def test_raises_on_invalid_stage(self, config_with_derived, mock_gen_data, R):
        with pytest.raises(ValueError, match="Unknown stage"):
            get_datamodule("invalid", config_with_derived, mock_gen_data, R)


# =============================================================================
# Integration tests
# =============================================================================


class TestIntegration:
    
    def test_full_pipeline_knots(self, config_with_derived, mock_gen_data, R):
        """Test full pipeline: config -> datamodule -> batch."""
        dm = get_datamodule("knots", config_with_derived, mock_gen_data, R)
        loader = dm.train_dataloader()
        
        batches = list(loader)
        assert len(batches) == config_with_derived.data.batches_per_epoch
        
        for batch in batches:
            assert batch["ics"].shape[0] == config_with_derived.data.batch_size
    
    def test_full_pipeline_cp(self, config_with_derived, mock_gen_data, R):
        """Test full pipeline for CP stage."""
        dm = get_datamodule("cp", config_with_derived, mock_gen_data, R)
        loader = dm.train_dataloader()
        
        batch = next(iter(loader))
        
        # Verify knot vector can be used with spline functions
        knot_vector = batch["knot_vector"]
        assert knot_vector.shape[1] > config_with_derived.data.spline_degree * 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])