"""
bspline_net (Github)

Copyright 2026 Carnegie Mellon University.

NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

Licensed under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.

[DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.

This Software includes and/or makes use of Third-Party Software each subject to its own license.

DM26-0750
"""
"""Unit tests for model modules."""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf

from src.bsplinenet.data import compute_derived_data_config
from src.bsplinenet.knot_model import ActivityModel, LitKnotModel
from src.bsplinenet.cp_model import CPDeepONetModel, LitCPModel
from src.bsplinenet.joint_model import LitJointModel


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def base_config():
    """Create a minimal config for testing."""
    cfg = OmegaConf.create({
        "seed": 42,
        "data": {
            "batch_size": 4,
            "batches_per_epoch": 2,
            "val_batches": 1,
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
        },
        "model": {
            "knots": {
                "hidden": 64,
                "kernel_size": 5,
                "lr": 0.01,
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
                "spacing_loss_weight": 1.0,
            },
        },
    })
    return compute_derived_data_config(cfg)


@pytest.fixture
def R(base_config):
    """Normalization vector."""
    return np.array(base_config.data.R)


@pytest.fixture
def sample_batch(base_config):
    """Create a sample batch for testing."""
    batch_size = base_config.data.batch_size
    n_samples = base_config.data.n_samples
    n_dim = len(base_config.data.R)
    n_interior_knots = base_config.data.n_interior_knots
    spline_degree = base_config.data.spline_degree

    ics = torch.randn(batch_size, n_dim)
    trajectories = torch.randn(batch_size, n_samples, n_dim)
    R_tensor = torch.tensor(base_config.data.R, dtype=torch.float32)
    
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
        "spacing_comp": torch.rand(batch_size, n_samples),
        "weighting_vector": torch.ones(batch_size, n_samples),
        "knot_positions": knot_positions,
        "knot_vector": knot_vector,
    }


# =============================================================================
# Tests: ActivityModel
# =============================================================================


class TestActivityModel:

    def test_init(self, base_config):
        model = ActivityModel(
            n_dim=len(base_config.data.R),
            n_samples=base_config.data.n_samples,
            hidden=base_config.model.knots.hidden,
            kernel_size=base_config.model.knots.kernel_size,
            min_spacing=base_config.data.min_spacing,
            max_spacing=base_config.data.max_spacing,
        )
        assert model is not None

    def test_forward_shape(self, base_config, sample_batch):
        model = ActivityModel(
            n_dim=len(base_config.data.R),
            n_samples=base_config.data.n_samples,
            hidden=base_config.model.knots.hidden,
            kernel_size=base_config.model.knots.kernel_size,
            min_spacing=base_config.data.min_spacing,
            max_spacing=base_config.data.max_spacing,
        )
        
        output = model(sample_batch["norm_ics"])
        
        assert output.shape == (base_config.data.batch_size, base_config.data.n_samples)

    def test_output_positive(self, base_config, sample_batch):
        """Spacing values should be positive."""
        model = ActivityModel(
            n_dim=len(base_config.data.R),
            n_samples=base_config.data.n_samples,
            hidden=base_config.model.knots.hidden,
            kernel_size=base_config.model.knots.kernel_size,
            min_spacing=base_config.data.min_spacing,
            max_spacing=base_config.data.max_spacing,
        )
        
        output = model(sample_batch["norm_ics"])
        
        assert (output > 0).all()


# =============================================================================
# Tests: LitKnotModel
# =============================================================================


class TestLitKnotModel:

    def test_init(self, base_config):
        model = LitKnotModel(base_config)
        assert model is not None
        assert hasattr(model, "model")
        assert hasattr(model, "R")

    def test_forward(self, base_config, sample_batch):
        model = LitKnotModel(base_config)
        output = model(sample_batch["norm_ics"])
        
        assert output.shape == (base_config.data.batch_size, base_config.data.n_samples)

    def test_predict_knots(self, base_config, sample_batch):
        model = LitKnotModel(base_config)
        knot_positions, knot_vector = model.predict_knots(sample_batch["norm_ics"])
        
        assert knot_positions.shape == (base_config.data.batch_size, base_config.data.n_interior_knots)
        assert (knot_positions >= 0).all() and (knot_positions <= 1).all()
        
        # Check knot vector structure
        d = base_config.data.spline_degree
        assert (knot_vector[:, :d+1] == 0).all()
        assert (knot_vector[:, -(d+1):] == 1).all()

    def test_training_step(self, base_config, sample_batch):
        model = LitKnotModel(base_config)
        loss = model.training_step(sample_batch, batch_idx=0)
        
        assert loss is not None
        assert loss.ndim == 0  # Scalar
        assert not torch.isnan(loss)

    def test_configure_optimizers(self, base_config):
        model = LitKnotModel(base_config)
        optimizer = model.configure_optimizers()
        
        assert optimizer is not None


# =============================================================================
# Tests: CPDeepONetModel
# =============================================================================


class TestCPDeepONetModel:

    def test_init(self, base_config):
        model = CPDeepONetModel(
            n_dim=len(base_config.data.R),
            n_cp_max=base_config.data.n_cp_max,
            spline_degree=base_config.data.spline_degree,
            widths_ic=base_config.model.cp.widths_ic,
            widths_kcp=base_config.model.cp.widths_kcp,
            rank=base_config.model.cp.rank,
            metapos=base_config.model.cp.metapos,
        )
        assert model is not None

    def test_forward_shape(self, base_config, sample_batch):
        model = CPDeepONetModel(
            n_dim=len(base_config.data.R),
            n_cp_max=base_config.data.n_cp_max,
            spline_degree=base_config.data.spline_degree,
            widths_ic=base_config.model.cp.widths_ic,
            widths_kcp=base_config.model.cp.widths_kcp,
            rank=base_config.model.cp.rank,
            metapos=base_config.model.cp.metapos,
        )
        
        R = torch.tensor(base_config.data.R, dtype=torch.float32)
        pred_cp, aux_losses = model(
            sample_batch["knot_vector"],
            sample_batch["norm_ics"],
            R,
        )
        
        assert pred_cp.shape == (
            base_config.data.batch_size,
            base_config.data.n_cp_max,
            len(base_config.data.R),
        )

    def test_aux_losses_present(self, base_config, sample_batch):
        model = CPDeepONetModel(
            n_dim=len(base_config.data.R),
            n_cp_max=base_config.data.n_cp_max,
            spline_degree=base_config.data.spline_degree,
            widths_ic=base_config.model.cp.widths_ic,
            widths_kcp=base_config.model.cp.widths_kcp,
            rank=base_config.model.cp.rank,
            metapos=base_config.model.cp.metapos,
        )
        
        R = torch.tensor(base_config.data.R, dtype=torch.float32)
        _, aux_losses = model(
            sample_batch["knot_vector"],
            sample_batch["norm_ics"],
            R,
        )
        
        assert "recon" in aux_losses
        assert "smooth" in aux_losses
        assert "ortho" in aux_losses
        assert "branch_smooth" in aux_losses


# =============================================================================
# Tests: LitCPModel
# =============================================================================


class TestLitCPModel:

    def test_init(self, base_config):
        model = LitCPModel(base_config)
        assert model is not None
        assert hasattr(model, "model")
        assert hasattr(model, "R")

    def test_forward(self, base_config, sample_batch):
        model = LitCPModel(base_config)
        pred_cp, aux_losses = model(
            sample_batch["knot_vector"],
            sample_batch["norm_ics"],
        )
        
        assert pred_cp.shape == (
            base_config.data.batch_size,
            base_config.data.n_cp_max,
            len(base_config.data.R),
        )

    def test_training_step(self, base_config, sample_batch):
        model = LitCPModel(base_config)
        loss = model.training_step(sample_batch, batch_idx=0)
        
        assert loss is not None
        assert loss.ndim == 0
        assert not torch.isnan(loss)

    def test_configure_optimizers(self, base_config):
        model = LitCPModel(base_config)
        optimizer = model.configure_optimizers()
        
        assert optimizer is not None


# =============================================================================
# Tests: LitJointModel
# =============================================================================


class TestLitJointModel:

    def test_init(self, base_config):
        model = LitJointModel(base_config)
        assert model is not None
        assert hasattr(model, "knot_model")
        assert hasattr(model, "cp_model")
        assert hasattr(model, "R")

    def test_forward(self, base_config, sample_batch):
        model = LitJointModel(base_config)
        spacing, knot_vector, pred_cp, aux_losses = model(sample_batch["norm_ics"])
        
        assert spacing.shape == (base_config.data.batch_size, base_config.data.n_samples)
        assert pred_cp.shape == (
            base_config.data.batch_size,
            base_config.data.n_cp_max,
            len(base_config.data.R),
        )

    def test_training_step(self, base_config, sample_batch):
        model = LitJointModel(base_config)
        loss = model.training_step(sample_batch, batch_idx=0)
        
        assert loss is not None
        assert loss.ndim == 0
        assert not torch.isnan(loss)

    def test_configure_optimizers_separate_lrs(self, base_config):
        model = LitJointModel(base_config)
        optimizer = model.configure_optimizers()
        
        # Should have 2 param groups with different LRs
        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]["lr"] == base_config.model.joint.knot_lr
        assert optimizer.param_groups[1]["lr"] == base_config.model.joint.cp_lr


# =============================================================================
# Tests: Gradient flow
# =============================================================================


class TestGradientFlow:

    def test_knot_model_gradients(self, base_config, sample_batch):
        model = LitKnotModel(base_config)
        loss = model.training_step(sample_batch, batch_idx=0)
        loss.backward()
        
        # Check that gradients exist
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                break
        assert has_grad

    def test_cp_model_gradients(self, base_config, sample_batch):
        model = LitCPModel(base_config)
        loss = model.training_step(sample_batch, batch_idx=0)
        loss.backward()
        
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                break
        assert has_grad

    def test_joint_model_gradients_both_models(self, base_config, sample_batch):
        model = LitJointModel(base_config)
        loss = model.training_step(sample_batch, batch_idx=0)
        loss.backward()
        
        # Check knot model has gradients
        knot_has_grad = False
        for param in model.knot_model.parameters():
            if param.grad is not None:
                knot_has_grad = True
                break
        
        # Check cp model has gradients
        cp_has_grad = False
        for param in model.cp_model.parameters():
            if param.grad is not None:
                cp_has_grad = True
                break
        
        assert knot_has_grad, "Knot model should have gradients"
        assert cp_has_grad, "CP model should have gradients"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])