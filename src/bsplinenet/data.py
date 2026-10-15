"""
bspline_net (Github)

Copyright 2026 Carnegie Mellon University.

NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

Licensed under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.

[DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.

This Software includes and/or makes use of Third-Party Software each subject to its own license.

DM26-0750
"""
# PAPER: shared
"""Data modules and datasets for B-spline trajectory learning.

Three training stages:
- knots: Train knot placement model (needs jerk_activity_max)
- cp: Train control point model with random knot vectors
- joint: Joint fine-tuning of both models
"""

import time
from typing import Callable, Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, open_dict
from torch.utils.data import DataLoader, IterableDataset

from .utils import ODESYS, build_knot_vector, compute_activity, density_to_knots, gen_ic_ellips

ode_sys = ODESYS.QUADROTOR_FAST
ode_func = ode_sys.get_ode()
batch_integrate = ode_sys.get_batch_integrator()  # None if unsupported


def compute_derived_data_config(cfg: DictConfig, isdatacfg=False) -> DictConfig:
    """Compute derived values from base config.

    Adds:
        - n_samples: from time_horizon and sample_interval
        - n_interior_knots: from time horizon and min_spacing (maximum knots)
        - R: normalization vector

    Args:
        cfg: Config with data section

    Returns:
        Config with derived values added to data section
    """
    with open_dict(cfg):

        data = cfg if isdatacfg else cfg.data

        # Time discretization
        data.n_samples = int(data.time_horizon / data.sample_interval) + 1
        # B-spline structure
        data.n_interior_knots = int(data.time_horizon * data.max_density)
        data.n_cp_max = data.n_interior_knots + data.spline_degree + 1
        data.n_minimum_knots = int(data.time_horizon * data.min_density)

        # Normalization vector
        data.R = [
            data.max_dx,
            data.max_x,
            data.max_dx,
            data.max_x,
            data.max_dx,
            data.max_x,
            data.max_da,
            data.max_ag,
            data.max_da,
            data.max_ag,
            data.max_da,
            data.max_ag,
        ]

    return data if isdatacfg else cfg


def gen_data(
    batch_size: int, min_radius: float, n_dim: int, A: np.ndarray, tt: np.ndarray, chunk: int = 2048
):
    if batch_size == 0:
        return None, None
    directions = np.random.randn(n_dim, batch_size)
    directions /= np.linalg.norm(directions, axis=0, keepdims=True)
    u = np.random.rand(batch_size)
    radii = (min_radius**n_dim + u * (1 - min_radius**n_dim)) ** (1 / n_dim)
    ics_np = (A @ (directions * radii)).T

    if batch_integrate is not None:
        T = tt.shape[0]
        tt64 = tt.astype(np.float64)
        traj_f32 = np.empty((batch_size, T, n_dim), dtype=np.float32)  # ONE float32 pool
        for s in range(0, batch_size, chunk):
            e = min(s + chunk, batch_size)
            block = batch_integrate(ics_np[s:e], tt64)  # float64, only chunk-sized
            traj_f32[s:e] = block.astype(np.float32, copy=False)
            del block  # free chunk's float64 immediately
        trajectories = torch.from_numpy(traj_f32)
    else:
        trajectories = torch.stack(
            [
                torch.from_numpy(ode_func(ics_np[i], tt, (np.zeros(n_dim),)).astype(np.float32))
                for i in range(batch_size)
            ]
        )
    ics = torch.from_numpy(ics_np.astype(np.float32))
    return ics, trajectories


class BaseTrajectoryDataset(IterableDataset):
    """Base dataset that generates trajectories from initial conditions.

    All stage-specific datasets inherit from this and add their own fields.
    """

    def __init__(
        self,
        cfg: DictConfig,
        batches_per_epoch: int,
        seed: Optional[int] = None,
    ):
        self.cfg = cfg
        self.gen_data_fn = gen_data
        self.R = torch.tensor(cfg.R).float()
        self.batches_per_epoch = batches_per_epoch
        self.seed = seed
        self.batch_size = cfg.batch_size
        self.epoch_size = batches_per_epoch * cfg.batch_size
        self.n_dim = cfg.n_dim
        self.n_samples = cfg.n_samples
        self.time_horizon = cfg.time_horizon
        self.A = np.eye(self.n_dim) * np.array(self.R)
        self.tt = np.linspace(0, self.time_horizon, self.n_samples)
        self.regen_every = int(self.cfg.get("regen_every_n_epochs", 1))
        self._pool = None
        self._pool_age = 0

    def _generate_epoch_data(self, esize) -> Dict[str, torch.Tensor]:
        """Generate all data for one epoch at once."""

        # worker_info = torch.utils.data.get_worker_info()
        # worker_id = worker_info.id if worker_info else "main"
        # print(f"Creating epoch data - worker {worker_id}")
        ics, trajectories = self.gen_data_fn(
            batch_size=esize,  # generate entire epoch at once
            min_radius=self.cfg.min_radius,
            n_dim=self.n_dim,
            A=self.A,
            tt=self.tt,
        )
        return {
            "ics": ics,
            "trajectories": trajectories,
            "norm_ics": ics / self.R,
            "norm_trajectories": trajectories / self.R,
        }

    def _slice_batch(
        self, epoch_data: Dict[str, torch.Tensor], start: int, end: int
    ) -> Dict[str, torch.Tensor]:
        """Slice a batch from epoch data."""
        return {k: v[start:end] for k, v in epoch_data.items()}

    def _compute_activity(self, trajectories: torch.Tensor) -> torch.Tensor:
        """Compute adaptive spacing from unnormalized trajectories."""
        return compute_activity(
            trajectories,
            time_horizon=self.cfg.time_horizon,
            max_translational_rate=self.cfg.max_translational_rate,
            max_roll_pitch_rate=self.cfg.max_roll_pitch_rate,
            max_yaw_rate=self.cfg.max_yaw_rate,
        )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            esize = self.epoch_size
        else:
            esize = self.epoch_size // worker_info.num_workers

        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        # Regenerate the pool only when stale (or first time); otherwise reuse.
        if self._pool is None or self._pool_age >= self.regen_every:
            epoch_data = self._generate_epoch_data(esize)
            epoch_data = self._add_stage_specific_fields(epoch_data)
            self._pool = epoch_data
            self._pool_age = 0
        self._pool_age += 1

        # Reshuffle the (possibly reused) pool so batch groupings vary each epoch.
        # For a fixed-seed dataset (val), skip the shuffle to stay reproducible.
        epoch_data = self._pool
        if self.seed is None:
            perm = torch.randperm(esize)
            epoch_data = {k: v[perm] for k, v in epoch_data.items()}

        n_batches = esize // self.batch_size
        for i in range(n_batches):
            start = i * self.batch_size
            end = start + self.batch_size
            yield self._slice_batch(epoch_data, start, end)

    def _add_stage_specific_fields(
        self, epoch_data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Override in subclasses to add stage-specific fields to full epoch data."""
        raise NotImplementedError


class KnotDataset(BaseTrajectoryDataset):
    """Dataset for knot model training.

    Yields:
        ics: Initial conditions (batch, n_dim)
        trajectories: Full trajectories (batch, n_samples, n_dim)
        norm_ics: Normalized initial conditions
        norm_trajectories: Normalized trajectories
        jerk_activity_max: Computed adaptive spacing (batch, n_samples)
    """

    def _add_stage_specific_fields(
        self, epoch_data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        epoch_data["jerk_activity_max"] = self._compute_activity(epoch_data["trajectories"])
        return epoch_data


class TrajDataset(BaseTrajectoryDataset):
    """Dataset for trajectory modeling.

    Yields:
        ics: Initial conditions (batch, n_dim)
        trajectories: Full trajectories (batch, n_samples, n_dim)
        norm_ics: Normalized initial conditions
        norm_trajectories: Normalized trajectories
    """

    def _add_stage_specific_fields(
        self, epoch_data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return epoch_data


class CPDataset(BaseTrajectoryDataset):
    """Dataset for control point model training with random knot vectors.

    Generates random knot vectors that mimic the distribution of adaptive spacing,
    allowing CP model to train independently of the knot model.

    Yields:
        ics: Initial conditions (batch, n_dim)
        trajectories: Full trajectories (batch, n_samples, n_dim)
        norm_ics: Normalized initial conditions
        norm_trajectories: Normalized trajectories
        knot_positions: Interior knot positions (batch, n_interior_knots)
        knot_vector: Full clamped knot vector (batch, n_knots_total)
    """

    def __init__(
        self,
        cfg: DictConfig,
        batches_per_epoch: int,
        seed: Optional[int] = None,
        uniform_prob: float = 0.3,
        random_prob: float = 0.3,
    ):
        super().__init__(cfg, batches_per_epoch, seed)
        self.uniform_prob = uniform_prob
        self.random_prob = random_prob
        self.n_interior_knots = cfg.n_interior_knots
        self.spline_degree = cfg.spline_degree
        self.min_spacing = 1.0 / (self.cfg.max_density * self.cfg.time_horizon)

    def _generate_knot_positions(
        self, epoch_size: int, reference_activity: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate interior knot positions.
        Args:
            epoch_size: Number of knot vectors to generate
            reference_activity: Optional spacing to mimic (epoch, n_samples)

        Returns:
            knot_positions: (epoch, n_interior_knots) with values in [0, 1],
                           padded with 1.0 for unused knots
        """
        knot_positions = torch.ones(epoch_size, self.n_interior_knots)
        # print(f"  _generate_knot_positions 1 {time.time()} {epoch_size} {self.uniform_prob} {reference_activity is None}")
        for b in range(epoch_size):
            type_rand = torch.rand(1).item()
            use_uniform = type_rand < self.uniform_prob
            use_random = type_rand > 1.0 - self.random_prob

            if use_uniform:
                # Uniform spacing with random number of knots
                n_knots = torch.randint(3, self.n_interior_knots + 1, (1,)).item()
                positions = torch.linspace(0, 1, n_knots + 2)[1:-1]  # Exclude 0 and 1

            elif use_random or (reference_activity is None):
                # Random with variable density
                n_knots = torch.randint(3, self.n_interior_knots + 1, (1,)).item()
                gaps = torch.rand(n_knots + 1)
                gaps[1:] += self.min_spacing
                positions = torch.cumsum(gaps, dim=0)[:-1]
                positions = self.min_spacing + (positions / positions[-1]) * (
                    1.0 - 2 * self.min_spacing
                )

            else:
                positions, _ = density_to_knots(
                    reference_activity[b : b + 1],
                    max_density=self.cfg.max_density,
                    min_density=self.cfg.min_density,
                    t_max=self.time_horizon,
                )

                knot_positions[b] = positions[0]
                continue

            knot_positions[b, :n_knots] = positions

        # print(f"  _generate_knot_positions 2 {time.time()}")
        return knot_positions

    def _add_stage_specific_fields(
        self, epoch_data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        epoch_size = epoch_data["ics"].shape[0]
        n_samples = epoch_data["norm_trajectories"].shape[1]

        # reference_activity = None
        reference_activity = self._compute_activity(epoch_data["trajectories"])
        knot_positions = self._generate_knot_positions(epoch_size, reference_activity)
        knot_vector = build_knot_vector(knot_positions, self.spline_degree)

        epoch_data["knot_positions"] = knot_positions
        epoch_data["knot_vector"] = knot_vector
        epoch_data["weighting_vector"] = torch.ones(epoch_size, n_samples)
        return epoch_data


class JointDataset(BaseTrajectoryDataset):
    """Dataset for joint training of knot and CP models.

    Similar to KnotDataset - knot vectors are computed by the knot model
    during training, not precomputed here.

    Yields:
        ics: Initial conditions (batch, n_dim)
        trajectories: Full trajectories (batch, n_samples, n_dim)
        norm_ics: Normalized initial conditions
        norm_trajectories: Normalized trajectories
        jerk_activity_max: Computed adaptive spacing (batch, n_samples)
    """

    def _add_stage_specific_fields(
        self, epoch_data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        epoch_data["jerk_activity_max"] = self._compute_activity(epoch_data["trajectories"])
        return epoch_data


# =============================================================================
# DataModules
# =============================================================================


class KnotDataModule(pl.LightningDataModule):
    """DataModule for knot model training."""

    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.gen_data_fn = gen_data
        self.R = torch.tensor(cfg.data.R).float()

    def train_dataloader(self) -> DataLoader:
        regen = int(self.cfg.data.get("regen_every_n_epochs", 1))
        num_workers = self.cfg.data.num_workers
        persistent = self.cfg.data.persistent_workers
        if regen > 1 and num_workers > 0 and not persistent:
            raise ValueError(
                f"regen_every_n_epochs={regen} requires persistent_workers=True "
                f"when num_workers={num_workers} (>0): otherwise the cached data "
                f"pool is discarded each epoch and data is regenerated every "
                f"epoch, defeating the cache. Set data.persistent_workers: true "
                f"(or num_workers: 0)."
            )

        dataset = KnotDataset(
            cfg=self.cfg.data,
            batches_per_epoch=self.cfg.data.batches_per_epoch,
            seed=None,
        )
        return DataLoader(
            dataset,
            batch_size=None,
            pin_memory=self.cfg.data.pin_memory,
            num_workers=num_workers,
            persistent_workers=persistent,
            prefetch_factor=2 if num_workers > 0 else None,
        )

    def val_dataloader(self) -> DataLoader:
        dataset = KnotDataset(
            cfg=self.cfg.data,
            batches_per_epoch=self.cfg.data.val_batches,
            seed=self.cfg.seed,  # Fixed for reproducibility
        )
        return DataLoader(
            dataset,
            batch_size=None,
            pin_memory=self.cfg.data.pin_memory,
            persistent_workers=self.cfg.data.persistent_workers,
            num_workers=1,
        )

    def test_dataloader(self) -> DataLoader:
        dataset = KnotDataset(
            cfg=self.cfg.data,
            batches_per_epoch=self.cfg.data.test_batches,
            seed=None,  # self.cfg.seed,  # Fixed for reproducibility
        )
        return DataLoader(
            dataset,
            batch_size=None,
            pin_memory=self.cfg.data.pin_memory,
            persistent_workers=self.cfg.data.persistent_workers,
            num_workers=1,
        )


class TrajDataModule(pl.LightningDataModule):
    """DataModule for control point model training."""

    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.gen_data_fn = gen_data
        self.R = torch.tensor(cfg.data.R).float()

    def train_dataloader(self) -> DataLoader:
        regen = int(self.cfg.data.get("regen_every_n_epochs", 1))
        num_workers = self.cfg.data.num_workers
        persistent = self.cfg.data.persistent_workers
        if regen > 1 and num_workers > 0 and not persistent:
            raise ValueError(
                f"regen_every_n_epochs={regen} requires persistent_workers=True "
                f"when num_workers={num_workers} (>0): otherwise the cached data "
                f"pool is discarded each epoch and data is regenerated every "
                f"epoch, defeating the cache. Set data.persistent_workers: true "
                f"(or num_workers: 0)."
            )

        dataset = TrajDataset(
            cfg=self.cfg.data,
            batches_per_epoch=self.cfg.data.batches_per_epoch,
            seed=None,
        )
        return DataLoader(
            dataset,
            batch_size=None,
            pin_memory=self.cfg.data.pin_memory,
            num_workers=num_workers,
            persistent_workers=persistent,
            prefetch_factor=2 if num_workers > 0 else None,
        )

    def val_dataloader(self) -> DataLoader:
        dataset = TrajDataset(
            cfg=self.cfg.data,
            batches_per_epoch=self.cfg.data.val_batches,
            seed=self.cfg.seed,
        )
        return DataLoader(
            dataset,
            batch_size=None,
            pin_memory=self.cfg.data.pin_memory,
            persistent_workers=self.cfg.data.persistent_workers,
            num_workers=1,
        )

    def test_dataloader(self) -> DataLoader:
        dataset = TrajDataset(
            cfg=self.cfg.data,
            batches_per_epoch=self.cfg.data.val_batches,
            seed=None,  # self.cfg.seed,
        )
        return DataLoader(
            dataset,
            batch_size=None,
            pin_memory=self.cfg.data.pin_memory,
            persistent_workers=self.cfg.data.persistent_workers,
            num_workers=1,
        )


class CPDataModule(pl.LightningDataModule):
    """DataModule for control point model training."""

    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.gen_data_fn = gen_data
        self.R = torch.tensor(cfg.data.R).float()

    def train_dataloader(self) -> DataLoader:
        regen = int(self.cfg.data.get("regen_every_n_epochs", 1))
        num_workers = self.cfg.data.num_workers
        persistent = self.cfg.data.persistent_workers
        if regen > 1 and num_workers > 0 and not persistent:
            raise ValueError(
                f"regen_every_n_epochs={regen} requires persistent_workers=True "
                f"when num_workers={num_workers} (>0): otherwise the cached data "
                f"pool is discarded each epoch and data is regenerated every "
                f"epoch, defeating the cache. Set data.persistent_workers: true "
                f"(or num_workers: 0)."
            )
        dataset = CPDataset(
            cfg=self.cfg.data,
            batches_per_epoch=self.cfg.data.batches_per_epoch,
            seed=None,
            uniform_prob=self.cfg.data.get("uniform_knot_prob", 0.3),
            random_prob=self.cfg.data.get("random_knot_prob", 0.3),
        )
        return DataLoader(
            dataset,
            batch_size=None,
            pin_memory=self.cfg.data.pin_memory,
            persistent_workers=self.cfg.data.persistent_workers,
            num_workers=self.cfg.data.num_workers,
            prefetch_factor=self.cfg.data.prefetch_factor,
        )

    def val_dataloader(self) -> DataLoader:
        dataset = CPDataset(
            cfg=self.cfg.data,
            batches_per_epoch=self.cfg.data.val_batches,
            seed=self.cfg.seed,
            uniform_prob=self.cfg.data.get("uniform_knot_prob", 0.3),
        )
        return DataLoader(
            dataset,
            batch_size=None,
            pin_memory=self.cfg.data.pin_memory,
            persistent_workers=self.cfg.data.persistent_workers,
            num_workers=1,
        )

    def test_dataloader(self) -> DataLoader:
        dataset = CPDataset(
            cfg=self.cfg.data,
            batches_per_epoch=self.cfg.data.val_batches,
            seed=None,  # self.cfg.seed,
            uniform_prob=self.cfg.data.get("uniform_knot_prob", 0.3),
        )
        return DataLoader(
            dataset,
            batch_size=None,
            pin_memory=self.cfg.data.pin_memory,
            persistent_workers=self.cfg.data.persistent_workers,
            num_workers=1,
        )


class JointDataModule(pl.LightningDataModule):
    """DataModule for joint training of knot and CP models."""

    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.gen_data_fn = gen_data
        self.R = torch.tensor(cfg.data.R).float()

    def train_dataloader(self) -> DataLoader:
        regen = int(self.cfg.data.get("regen_every_n_epochs", 1))
        num_workers = self.cfg.data.num_workers
        persistent = self.cfg.data.persistent_workers
        if regen > 1 and num_workers > 0 and not persistent:
            raise ValueError(
                f"regen_every_n_epochs={regen} requires persistent_workers=True "
                f"when num_workers={num_workers} (>0): otherwise the cached data "
                f"pool is discarded each epoch and data is regenerated every "
                f"epoch, defeating the cache. Set data.persistent_workers: true "
                f"(or num_workers: 0)."
            )
        dataset = JointDataset(
            cfg=self.cfg.data,
            batches_per_epoch=self.cfg.data.batches_per_epoch,
            seed=None,
        )
        return DataLoader(
            dataset,
            batch_size=None,
            pin_memory=self.cfg.data.pin_memory,
            persistent_workers=self.cfg.data.persistent_workers,
            num_workers=self.cfg.data.num_workers,
            prefetch_factor=self.cfg.data.prefetch_factor,
        )

    def val_dataloader(self) -> DataLoader:
        dataset = JointDataset(
            cfg=self.cfg.data,
            batches_per_epoch=1,
            seed=self.cfg.seed,
        )
        return DataLoader(
            dataset,
            batch_size=None,
            pin_memory=self.cfg.data.pin_memory,
            persistent_workers=self.cfg.data.persistent_workers,
            num_workers=1,
        )


# =============================================================================
# Factory function
# =============================================================================


def get_datamodule(
    stage: str,
    cfg: DictConfig,
) -> pl.LightningDataModule:
    """Factory function to get the appropriate DataModule for a training stage.

    Args:
        stage: One of 'knots', 'cp', 'joint'
        cfg: Full config
        R: Normalization array

    Returns:
        Appropriate DataModule for the stage
    """
    datamodules = {
        "knots": KnotDataModule,
        "traj": TrajDataModule,
        "cp": CPDataModule,
        "joint": JointDataModule,
    }

    if stage not in datamodules:
        raise ValueError(f"Unknown stage: {stage}. Must be one of {list(datamodules.keys())}")

    return datamodules[stage](cfg)
