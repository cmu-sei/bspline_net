"""
bspline_net (Github)

Copyright 2026 Carnegie Mellon University.

NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

Licensed under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.

[DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.

This Software includes and/or makes use of Third-Party Software each subject to its own license.

DM26-0750
"""
# input: initial condition, number of control points
# output: suggested location of control points (n_cp-length vector)
# optimize: mse fit + distribution of points from uniform.

from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import numpy as np
import torch
from bsplinenet.ff_net import gen_data_np, ODESYS
from quadrotor.ellipsoid import gen_ic_ellips
from pathlib import Path
from functools import partial

# ==============================================================================
# CONFIGURATION PARAMETERS
# ==============================================================================

# --- State Space Configuration ---
N_DIM = 12  # Number of state space dimensions (fixed for quadrotor)

# --- Physical Limits ---
MAX_X = 2  # m - maximum position
MAX_DX = 2  # m/s - maximum velocity
MAX_AG = np.pi / 4  # radians - maximum angle
MAX_DA = np.pi / 2  # rad/sec - maximum angular velocity

# --- B-Spline Configuration ---
SPLINE_DEGREE = 3
MIN_SPACING = 0.025
MAX_SPACING = 0.1
NOMINAL_SPACING = 0.1

# --- Simulation Configuration ---
TIME_HORIZON = 2.5  # seconds
SAMPLE_INTERVAL = 0.005  # seconds
N_SAMPLES = int(TIME_HORIZON / SAMPLE_INTERVAL) + 1

# --- Training Configuration ---
EPOCH_SIZE = 5000
BATCH_SIZE = 500
MIN_RADIUS = 1e-3

# Knot model training
KNOT_MODEL_EPOCHS = 0 #20
KNOT_MODEL_BATCHES = 25
KNOT_MODEL_LR = 1e-2
KNOT_MODEL_HIDDEN = 128
KNOT_MODEL_KERNEL_SIZE = 5

KNOT_MODEL_LR_JOINT = 5e-3
# Control point model training
CP_MODEL_EPOCHS = 100
CP_MODEL_BATCHES = 20
CP_MODEL_LR =1e-4 # was 2e-4

# CP model architecture
CP_WIDTHS_IC = [512, 1024, 512, 1024, 512]
CP_WIDTHS_KCP = [1024, 256, 256]
CP_RANK = 1024
CP_CHANNELS = 32

# Trunk encoder-decoder losses
TRUNK_RECON_WEIGHT = 0.1
TRUNK_SMOOTH_WEIGHT = 0.1
TRUNK_ORTHO_WEIGHT = 0.01

# Branch smoothness (contrastive) loss
BRANCH_SMOOTH_WEIGHT = 0.1
BRANCH_SMOOTH_TEMP = 0.1

# Trunk pretraining
TRUNK_PRETRAIN_EPOCHS = 0 #20
TRUNK_PRETRAIN_BATCHES = 25
TRUNK_PRETRAIN_LR = 1e-3

# --- Checkpoint Paths ---
KNOT_CHECKPOINT_PATH = Path("knot_checkpoint.pth.tar")
CP_CHECKPOINT_PATH = Path("cp_checkpoint.pth.tar")

# ==============================================================================
# DERIVED PARAMETERS (computed from configuration)
# ==============================================================================

R_IC = np.array(
    [
        MAX_DX,
        MAX_X,
        MAX_DX,
        MAX_X,
        MAX_DX,
        MAX_X,
        MAX_DA,
        MAX_AG,
        MAX_DA,
        MAX_AG,
        MAX_DA,
        MAX_AG,
    ]
)

R = np.array(
    [
        MAX_DX,
        MAX_X,
        MAX_DX,
        MAX_X,
        MAX_DX,
        MAX_X,
        2 * MAX_DA,
        np.pi,
        2 * MAX_DA,
        np.pi,
        MAX_DA,
        np.pi / 4,
    ]
)

A = np.eye(N_DIM, N_DIM) * R_IC
N_CP_MAX = int(1 / MIN_SPACING) + SPLINE_DEGREE + 1
N_INTERIOR_KNOTS = N_CP_MAX - SPLINE_DEGREE - 1
TT = np.linspace(0, TIME_HORIZON, N_SAMPLES)

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================


def trajectory_generator(batch_size, n_dim, A, R, tt, ode_func, min_radius=0.1):
    """Generate a batch of initial conditions and trajectories."""
    ics_np = gen_ic_shell(n_dim, A, batch_size, min_radius=min_radius).T
    ics = torch.from_numpy(ics_np).to(torch.float32)

    trajectories = torch.stack(
        [
            torch.from_numpy(
                ode_func(ics[i, :], tt, (np.zeros(n_dim),)).astype(np.float32)
            )
            for i in range(batch_size)
        ]
    )

    return ics, trajectories


def gen_ic_shell(n_dim, A, n_samples, min_radius=0.0):
    """Generate initial conditions uniformly distributed in a shell."""
    directions = np.random.randn(n_dim, n_samples)
    directions /= np.linalg.norm(directions, axis=0, keepdims=True)

    u = np.random.rand(n_samples)
    radii = (min_radius**n_dim + u * (1 - min_radius**n_dim)) ** (1 / n_dim)

    points = directions * radii
    return A @ points


def bspline_basis(t, knots, d):
    """Compute B-spline basis functions using Cox-de Boor recursion."""
    Ns = t.shape[0]
    n_knots = knots.shape[0]

    B = torch.zeros(Ns, n_knots - 1, device=knots.device)

    for i in range(n_knots - 1):
        left = t >= knots[i]
        right = t < knots[i + 1]
        B[:, i] = (left & right).to(t.dtype)

    B[:, -1] = torch.where(t == knots[-1], torch.ones_like(t), B[:, -1])

    for k in range(1, d + 1):
        B_new = torch.zeros(Ns, n_knots - 1 - k, device=knots.device)

        for i in range(n_knots - 1 - k):
            denom_left = knots[i + k] - knots[i]
            if denom_left > 1e-10:
                left = (t - knots[i]) / denom_left * B[:, i]
            else:
                left = torch.zeros(Ns, device=knots.device)

            denom_right = knots[i + k + 1] - knots[i + 1]
            if denom_right > 1e-10:
                right = (knots[i + k + 1] - t) / denom_right * B[:, i + 1]
            else:
                right = torch.zeros(Ns, device=knots.device)

            B_new[:, i] = left + right

        B = B_new

    endpoint_mask = (t == knots[-1]).unsqueeze(1)
    endpoint_basis = torch.zeros_like(B)
    endpoint_basis[:, -1] = 1.0
    B = torch.where(endpoint_mask, endpoint_basis, B)

    return B


def compute_spacing(
    trajectories,
    min_spacing=0.01,
    max_spacing=0.2,
    n_samples=100,
    window_sizes=[3, 5, 7, 9],
):
    """Compute adaptive spacing based on trajectory activity."""
    max_pad = max(window_sizes) // 2
    max_step = 1 / min_spacing / (n_samples - 1)
    min_step = 1 / max_spacing / (n_samples - 1)
    sensitivity = 2

    Ns = trajectories.shape[1]
    traj_padded = torch.nn.functional.pad(
        trajectories, (0, 0, max_pad, max_pad), mode="replicate"
    )

    activity = (
        torch.stack(
            [
                (
                    traj_padded[:, max_pad + w // 2 : max_pad + w // 2 + Ns]
                    - traj_padded[:, max_pad - w // 2 : max_pad - w // 2 + Ns]
                )
                .abs()
                .max(dim=-1)
                .values
                for w in window_sizes
            ],
            dim=-1,
        ).mean(dim=-1)
        * n_samples
        * min_spacing
    )

    activity_bounded = 1 - torch.exp(-activity * sensitivity)
    spacing = min_step + (max_step - min_step) * activity_bounded

    return spacing


def spacing_to_knots(spacing, n_knots_max):
    """Place a knot every time cumulative spacing crosses an integer."""
    batch = spacing.shape[0]
    t_eval = torch.linspace(0, 1, spacing.shape[1], device=spacing.device)

    knot_positions = torch.ones(batch, n_knots_max, device=spacing.device)
    n_knots_actual = torch.zeros(batch, dtype=torch.int, device=spacing.device)

    for b in range(batch):
        cumsum = spacing[b].cumsum(dim=0)

        knot_idx = 0
        threshold = 1
        while threshold <= cumsum[-1] and knot_idx < n_knots_max:
            idx = (cumsum >= threshold).float().argmax()
            knot_positions[b, knot_idx] = t_eval[idx]
            knot_idx += 1
            threshold += 1

        n_knots_actual[b] = knot_idx

    return knot_positions, n_knots_actual


def build_knot_vector(knot_positions, d):
    """Build clamped knot vector from interior knot positions."""
    batch = knot_positions.shape[0]

    zeros = torch.zeros(batch, d + 1, device=knot_positions.device)
    ones = torch.ones(batch, d + 1, device=knot_positions.device)

    knot_vector = torch.cat([zeros, knot_positions, ones], dim=1)
    return knot_vector


def compute_signal_loss(signal, activity, base_density=0.05, activity_scale=0.5):
    """Compute loss for signal matching activity distribution."""
    total_activity = activity.sum(dim=1)
    target_density = base_density + activity_scale * total_activity / activity.shape[1]

    coverage = ((signal - activity) ** 2).sum(dim=1).mean()

    actual_density = signal.mean(dim=1)
    sparsity = ((actual_density - target_density) ** 2).mean()

    return coverage + sparsity


def find_knot_positions(signal, n_knots, t_eval):
    """Non-differentiable peak finding for interior knots."""
    batch = signal.shape[0]
    knot_positions = torch.zeros(batch, n_knots, device=signal.device)

    for b in range(batch):
        sig = signal[b].detach().cpu().numpy()
        peak_indices, properties = find_peaks(sig, height=0)

        if len(peak_indices) == 0:
            knot_positions[b, :] = torch.linspace(0.1, 0.9, n_knots)
        else:
            heights = properties["peak_heights"]
            top_peaks = peak_indices[np.argsort(heights)[::-1][:n_knots]]
            top_peaks = np.sort(top_peaks)

            for i, idx in enumerate(top_peaks):
                knot_positions[b, i] = t_eval[idx]

            for i in range(len(top_peaks), n_knots):
                knot_positions[b, i] = 1.0

    return knot_positions

def compute_controlpoints_from_trajectory(
    trajectories, knots, d, max_n_cp
):
    
    batch, Ns, n_state = trajectories.shape
    t = torch.linspace(0, 1, Ns, device=knots.device)
    
    control_points = torch.zeros(batch, max_n_cp, n_state, device=trajectories.device)
    
    for b in range(batch):
        # Same logic as compute_trajectory_loss
        end_knots = torch.where(knots[b] == 1)[0]
        end_point = len(knots[b]) if len(end_knots) <= d + 1 else end_knots[d + 1]
        
        # Number of control points for this batch element
        n_cp_b = end_point - d - 1
        
        # Build basis matrix: (Ns, n_cp_b)
        Bit = bspline_basis(t, knots[b, :end_point], d)
        # Solve least squares: Bit @ cp = trajectories[b]
        solution = torch.linalg.lstsq(Bit, trajectories[b]).solution
        
        # Store in the first n_cp_b slots (rest remain zero)
        control_points[b, :n_cp_b, :] = solution
    
    return control_points

def compute_trajectory_loss(
    pred_controlpoints, knots, trajectories, weighting_vector, d
):
    """Compute weighted MSE loss between predicted and true trajectories."""
    batch, Ns, n_state = trajectories.shape
    t = torch.linspace(0, 1, Ns, device=knots.device)

    total_loss = 0
    pred_trajectories = torch.zeros(batch, Ns, n_state)

    for b in range(batch):
        end_knots = torch.where(knots[b] == 1)[0]
        end_point = len(knots[b]) if len(end_knots) <= d + 1 else end_knots[d + 1]

        Bit = bspline_basis(t, knots[b, :end_point], d)
        pred_trajectories[b, :, :] = Bit @ pred_controlpoints[b, : end_point - d - 1]

    norm = (trajectories.max(dim=1)[0] - trajectories.min(dim=1)[0]).view(batch, 1, n_state)
    total_loss1 = (((pred_trajectories - trajectories) ** 2) ).mean()
    total_loss2 = (((pred_trajectories - trajectories) ** 2) /(norm + 1e-3)).mean()
    return total_loss1, total_loss2, pred_trajectories

def physics_loss(pred_cps, knots, d, t_eval, T=2.5):
    """Compute physics consistency loss (velocity = d(position)/dt)."""
    batch, n_cp, _ = pred_cps.shape

    vel_cps = pred_cps[:, :, 0::2]  # velocity control points
    pos_cps = pred_cps[:, :, 1::2]  # position control points

    # Derivative control points: Q_i = d * (P_{i+1} - P_i) / (t_{i+d+1} - t_{i+1})
    pos_cps_deriv = torch.zeros(batch, n_cp - 1, 6, device=pred_cps.device)
    for i in range(n_cp - 1):
        # For control point i of the derivative spline:
        dknot = (knots[:, i + d + 1] - knots[:, i + 1]).unsqueeze(-1)
        dknot = torch.clamp(dknot, min=1e-8)
        pos_cps_deriv[:, i, :] = d * (pos_cps[:, i + 1, :] - pos_cps[:, i, :]) / dknot

    # Scale factor for unit conversion
    scale = torch.ones(6, device=pred_cps.device)
    scale[:3] = 1.0 / T
    scale[3:] = 1.0 / (T * 2 * np.pi)
    pos_cps_deriv = pos_cps_deriv * scale

    loss = 0
    for b in range(batch):
        # Evaluate velocity spline (degree d)
        Bit_vel = bspline_basis(t_eval, knots[b], d)
        vel_eval = Bit_vel @ vel_cps[b]

        # Evaluate derivative of position spline (degree d-1)
        # Derivative knot vector: remove first and last knot
        knots_deriv = knots[b, 1:-1]
        Bit_deriv = bspline_basis(t_eval, knots_deriv, d - 1)
        pos_deriv_eval = Bit_deriv @ pos_cps_deriv[b]

        loss += ((vel_eval - pos_deriv_eval) ** 2).mean()

    return loss / batch


def mask_controlpoints(effective_knots, controlpoints, order):
    """Mask invalid control points based on knot positions."""
    batch, n_cp, n_dim = controlpoints.shape

    is_one = effective_knots >= 1.0 - 1e-8
    first_one_idx = is_one.float().argmax(dim=1)
    has_one = is_one.any(dim=1)
    first_one_idx = torch.where(
        has_one,
        first_one_idx,
        torch.tensor(effective_knots.shape[1], device=effective_knots.device),
    )

    n_valid_cp = first_one_idx.clamp(min=1, max=n_cp)

    cp_indices = torch.arange(n_cp, device=effective_knots.device).unsqueeze(0)
    valid_mask = cp_indices < n_valid_cp.unsqueeze(1)

    last_valid_idx = (n_valid_cp - 1).clamp(min=0)
    last_valid_cp = controlpoints[
        torch.arange(batch, device=effective_knots.device), last_valid_idx, :
    ]

    masked_controlpoints = controlpoints.clone()
    invalid_mask = ~valid_mask
    masked_controlpoints[invalid_mask] = last_valid_cp.unsqueeze(1).expand(
        -1, n_cp, -1
    )[invalid_mask]

    return masked_controlpoints, valid_mask, n_valid_cp


# ==============================================================================
# VISUALIZATION
# ==============================================================================


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
        ax.plot(
            t_eval, trajectories[idx, :, i].numpy(), "k-", linewidth=2, label="True"
        )

        if has_controlpoints:
            knots_np = knot_vector[idx].detach().numpy()
            n_cp_eff = (knot_positions[idx] < 0.99).sum().item() + d + 1
            cp_positions = np.array(
                [knots_np[j + 1 : j + d + 1].mean() for j in range(n_cp_eff)]
            )
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
        #breakpoint()

    ax.text(0.5, 0.5, info_text, ha="center", va="center", fontsize=11)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"scripts/{name}_e_{epoch:04d}.png")
    plt.close()


# ==============================================================================
# NEURAL NETWORK MODELS
# ==============================================================================


class ActivityModel(torch.nn.Module):
    """Predicts adaptive spacing from initial conditions."""

    def __init__(
        self,
        dim,
        R,
        Ns,
        hidden=128,
        kernel_size=5,
        max_spacing=0.2,
        min_spacing=0.01,
        n_samples=501,
    ):
        super().__init__()
        self.dim = dim
        self.R = torch.from_numpy(R).float()
        self.max_spacing = max_spacing
        self.min_spacing = min_spacing
        self.n_samples = n_samples

        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, Ns),
        )

        self.conv1 = torch.nn.Conv1d(
            1, 8, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.conv2 = torch.nn.Conv1d(
            8, 8, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.conv_out = torch.nn.Conv1d(8, 1, kernel_size=1)

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, ic):
        ic = ic / self.R
        x = self.net(ic)
        x = x.unsqueeze(1)

        h = self.conv1(x)
        h = torch.nn.SiLU()(h)

        residual = h
        h = self.conv2(h)
        h = torch.nn.SiLU()(h)
        h = h + residual

        h = self.conv_out(h)
        x = x + h

        x = x.squeeze(1)

        max_step = 1 / self.min_spacing / (self.n_samples - 1)
        min_step = 1 / self.max_spacing / (self.n_samples - 1)
        spacing = min_step + (max_step - min_step) * x

        return spacing


class CPDeepONetModel(torch.nn.Module):
    """DeepONet model for predicting control points from knots and initial conditions."""

    def __init__(self, widths_ic, widths_kcp, dim, R, n_cp, order, rank=8, metapos=8):
        super().__init__()

        self.dim = dim
        self.R = torch.from_numpy(R).float()
        self.n_cp = n_cp
        self.order = order
        self.rank = rank
        self.metapos = metapos
        self.act = torch.nn.SiLU()

        # Main branch
        ics_ = [self.dim + self.dim // 2] + widths_ic + [self.metapos * self.dim]
        ic_layer_list = []

        for i, j in zip(ics_[:-1], ics_[1:]):
            ic_layer_list.append(torch.nn.Linear(i, j))
            ic_layer_list.append(torch.nn.SiLU())
        ic_layer_list.append(torch.nn.BatchNorm1d(ics_[-1]))

        self.ic_layers = torch.nn.Sequential(*ic_layer_list)

        self.ic_conv1 = torch.nn.Conv1d(
            self.dim, 4*self.dim, kernel_size=5, padding=2
        )
        self.ic_conv1_bn = torch.nn.BatchNorm1d(4*self.dim)
        self.ic_conv2 = torch.nn.Conv1d(
            4*self.dim, self.dim, kernel_size=5, padding=2, bias=False
        )
        self.ic_conv2_bn = torch.nn.BatchNorm1d(self.dim)

        self.ic_linear1 = torch.nn.Linear(
            self.metapos * self.dim, self.rank * self.dim
        )
        self.ic_linear2 = torch.nn.Linear(
            self.rank * self.dim, self.rank * self.dim // 2
        )

        self.integrate_sequence = torch.nn.Sequential(
            torch.nn.Linear(self.rank, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, 1024),
            torch.nn.SiLU(),
            #torch.nn.Dropout(0.1),
            torch.nn.Linear(1024, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, self.rank),
        )

        # Trunk network (encoder)
        self.cp_local = torch.nn.Sequential(
            torch.nn.Linear(order + 2, widths_kcp[0]),
            torch.nn.SiLU(),
            torch.nn.Linear(widths_kcp[0], self.rank - 1),
        )

        self.cp_conv1 = torch.nn.Conv1d(
            self.n_cp, 32, kernel_size=5, stride=1, padding=2
        )
        self.cp_conv2 = torch.nn.Conv1d(
            32, self.n_cp, kernel_size=5, stride=1, padding=2
        )
        self.cp_linear1 = torch.nn.Linear(self.rank - 1, 128)
        self.cp_linear2 = torch.nn.Linear(128, self.rank - 1)

        # Trunk decoder (for reconstruction loss)
        self.cp_decoder = torch.nn.Sequential(
            torch.nn.Linear(self.rank - 1, widths_kcp[0]),
            torch.nn.SiLU(),
            torch.nn.Linear(widths_kcp[0], order + 2),
        )

    def branch_smoothness_loss(self, ic_input, ic_embed, temperature=0.1):
        batch = ic_input.shape[0]
        ic_dists = torch.cdist(ic_input, ic_input)
        ic_sim = torch.exp(-ic_dists / (ic_dists.mean() * temperature + 1e-6))
        embed_flat = ic_embed.reshape(batch, -1)
        embed_norm = embed_flat / (embed_flat.norm(dim=-1, keepdim=True) + 1e-6)
        embed_sim = embed_norm @ embed_norm.T
        return ((embed_sim - ic_sim) ** 2).mean()

    def ic_encode(self, x):
        """Encode initial conditions with symmetry-aware features."""
        inv_idx = [0, 1, 4, 5, 8, 9]      # dx, x, dz, z, dq, q
        flip_idx = [2, 3, 6, 7, 10, 11]   # dy, y, dp, p, dr, r
        
        invariant = x[..., inv_idx]
        signed = x[..., flip_idx]
        magnitudes = torch.abs(signed)

        return torch.cat([invariant, magnitudes, signed], dim=-1)

    def forward_ic(self, x):
        x = self.ic_encode(x)
        
        x = self.ic_layers(x).view(-1, self.dim, self.metapos)
        residual = x
        x = self.act(self.ic_conv1_bn(self.ic_conv1(x)))
        x = self.ic_conv2_bn(self.ic_conv2(x))
        x = self.act(x + residual)
        
        x = x.reshape(-1,self.dim * self.metapos)
        
        x = self.act(self.ic_linear1(x))
        x = self.ic_linear2(x)
        return x.view(-1, self.rank, self.dim // 2)

    def forward_cp(self, knots):
        windows = knots.unfold(dimension=1, size=self.order + 2, step=1).clone()
        x = self.act(self.cp_local(windows))
        residual = x
        x = self.act(self.cp_conv1(x))
        x = self.act(self.cp_conv2(x)) + residual
        residual = x
        x = self.act(self.cp_linear1(x))
        z = self.act(self.cp_linear2(x)) + residual

        # === Auxiliary losses ===
        # Reconstruction loss
        windows_recon = self.cp_decoder(z)
        recon_loss = ((windows - windows_recon) ** 2).mean()

        # Smoothness loss (adjacent CPs should have similar latents)
        smooth_loss = ((z[:, 1:, :] - z[:, :-1, :]) ** 2).mean()

        # Orthogonality loss (encourage decorrelated latent dimensions)
        z_centered = z - z.mean(dim=1, keepdim=True)
        cov = torch.einsum("bci,bcj->bij", z_centered, z_centered) / z.shape[1]
        ortho_loss = ((cov - torch.eye(cov.shape[-1], device=cov.device)) ** 2).mean()
        # === Continue with encoder forward ===
        x = torch.nn.ReLU()(z.view(-1, self.n_cp, self.rank - 1))
        x = torch.cat([x, torch.ones(*x.shape[:2], 1, device=x.device)], dim=-1)
        return x, {"recon": recon_loss, "smooth": smooth_loss, "ortho": ortho_loss}

    def forward_in(self, x):
        x = x.permute(0, 2, 1) # (branch, dim//2, rank)
        return self.integrate_sequence(x).permute(0, 2, 1) #(branch, rank, dim//2)

    def forward(self, knots, ic, branch_smooth_temp=0.1):
        ic_input = ic / self.R.to(ic.device)
        cp_embed, aux_losses = self.forward_cp(knots)
        ic_embed = self.forward_ic(ic_input)
        aux_losses["branch_smooth"] = self.branch_smoothness_loss(
            ic_input, ic_embed, temperature=branch_smooth_temp
        )
        ic_integ = self.forward_in(ic_embed)
        ic_embed = torch.cat([ic_embed, ic_integ], dim=2)
        pred = torch.einsum("itr,ird->itd", cp_embed, ic_embed) / (self.rank**0.5)
        pred = pred[..., [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11]]
        return (
            self.R.to(pred.device) * (pred + ic_input.reshape(-1, 1, self.dim)),
            aux_losses,
        )


# ==============================================================================
# CHECKPOINT UTILITIES
# ==============================================================================

def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint with only the optimizer state for this model."""
    
    # Get parameter ids for this model
    model_param_ids = {id(p) for p in model.parameters()}
    
    # Build list of all params in optimizer (in order)
    all_params = []
    for group in optimizer.param_groups:
        all_params.extend(group['params'])
    
    # Find indices belonging to this model
    model_indices = [i for i, p in enumerate(all_params) if id(p) in model_param_ids]
    
    # Create remapped state (new indices 0, 1, 2... for this model's params)
    full_opt_state = optimizer.state_dict()
    filtered_state = {}
    for new_idx, old_idx in enumerate(model_indices):
        if old_idx in full_opt_state['state']:
            filtered_state[new_idx] = full_opt_state['state'][old_idx]
    
    # Filter param groups to only those containing this model's params
    filtered_groups = []
    for group in optimizer.param_groups:
        group_param_ids = {id(p) for p in group['params']}
        if group_param_ids & model_param_ids:
            # Copy group settings without the actual param tensors
            new_group = {k: v for k, v in group.items() if k != 'params'}
            filtered_groups.append(new_group)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {
            'state': filtered_state,
            'param_groups': filtered_groups
        },
        "loss": loss,
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path} at epoch {epoch}")
    

def load_checkpoint(model, optimizer, path, device='cpu'):
    """Load model checkpoint and restore optimizer state."""
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Build the state dict in the format optimizer.load_state_dict expects
    # It needs 'params' in each group (just placeholders for the count)
    saved_opt = checkpoint['optimizer_state_dict']
    
    # Reconstruct param_groups with 'params' included
    current_state = optimizer.state_dict()
    
    # Update the state (momentum buffers, etc.)
    current_state['state'] = saved_opt['state']
    
    # Update hyperparameters from saved groups (lr, weight_decay, etc.)
    for i, saved_group in enumerate(saved_opt['param_groups']):
        for key, value in saved_group.items():
            if key != 'params':
                current_state['param_groups'][i][key] = value
    
    optimizer.load_state_dict(current_state)
    
    return checkpoint['epoch'], checkpoint['loss']
# ==============================================================================
# TRAINING FUNCTIONS
# ==============================================================================


def train_knot_model(knot_model, optimizer, gen_data, config):
    """Train the knot placement model."""
    print("Starting knot model training")
    epoch =0
    if Path(KNOT_CHECKPOINT_PATH).is_file():
        epoch, _ = load_checkpoint(knot_model, optimizer, KNOT_CHECKPOINT_PATH)
    
    for epoch in range(epoch, epoch + config["epochs"]):
        for batch in range(config["batches"]):
            ics, trajectories = gen_data(batch_size=BATCH_SIZE, min_radius=MIN_RADIUS)

            spacing_comp = compute_spacing(
                trajectories / torch.from_numpy(R).float(),
                min_spacing=MIN_SPACING,
                max_spacing=MAX_SPACING,
                n_samples=N_SAMPLES,
            )

            spacing_pred = knot_model(ics / torch.from_numpy(R).float())
            spacing_loss = ((spacing_pred - spacing_comp) ** 2).sum()

            optimizer.zero_grad()
            spacing_loss.backward()
            optimizer.step()

            print(
                f"\rEpoch {epoch:04d}/{batch:04d} SpacingLoss: {spacing_loss:.5f}",
                end="",
            )

        print()

        knot_positions, n_knots_predicted = spacing_to_knots(
            spacing_pred.detach(), N_INTERIOR_KNOTS
        )
    if config["epochs"]>0:
        save_checkpoint(knot_model, optimizer, epoch, spacing_loss, KNOT_CHECKPOINT_PATH)

    return epoch


def pretrain_trunk(cp_model, knot_model, gen_data, epoch, config):
    """Pretrain the trunk encoder-decoder before full model training."""
    print("Starting trunk encoder-decoder pretraining")

    # Only optimize trunk encoder/decoder parameters
    trunk_params = (
        list(cp_model.cp_local.parameters())
        + list(cp_model.cp_conv1.parameters())
        + list(cp_model.cp_conv2.parameters())
        + list(cp_model.cp_linear1.parameters())
        + list(cp_model.cp_linear2.parameters())
        + list(cp_model.cp_decoder.parameters())
    )

    optimizer = torch.optim.Adam(trunk_params, lr=config["lr"])

    for epoch in range(config["epochs"]):
        for batch in range(config["batches"]):
            ics, trajectories = gen_data(batch_size=BATCH_SIZE, min_radius=MIN_RADIUS)

            norm_ics = ics / torch.from_numpy(R).float()
            norm_trajectories = trajectories / torch.from_numpy(R).float()

            spacing_comp = compute_spacing(
                norm_trajectories,
                min_spacing=MIN_SPACING,
                max_spacing=MAX_SPACING,
                n_samples=N_SAMPLES,
            )

            with torch.no_grad():
                spacing_pred = knot_model(norm_ics)

            knot_positions, _ = spacing_to_knots(spacing_pred, N_INTERIOR_KNOTS)
            knot_vector = build_knot_vector(knot_positions, SPLINE_DEGREE)

            # Only run trunk forward to get losses
            _, aux_losses = cp_model.forward_cp(knot_vector)

            pretrain_loss = (
                aux_losses["recon"]
                + TRUNK_SMOOTH_WEIGHT * aux_losses["smooth"]
                + TRUNK_ORTHO_WEIGHT * aux_losses["ortho"]
            )

            optimizer.zero_grad()
            pretrain_loss.backward()
            optimizer.step()

            print(
                f"\rPretrain {epoch:04d}/{batch:04d} "
                f"Recon: {aux_losses['recon']:.5f} "
                f"Smooth: {aux_losses['smooth']:.5f} "
                f"Ortho: {aux_losses['ortho']:.5f}",
                end="",
            )

        print()
    print("Trunk pretraining complete")


def train_cp_model(cp_model, knot_model, optimizer_cp, gen_data, epoch, config):
    """Train the control point prediction model."""
    print("Starting control point model training")
    #epoch =0
    #if Path(CP_CHECKPOINT_PATH).is_file():
    #    epoch, _ = load_checkpoint(cp_model, optimizer_cp, CP_CHECKPOINT_PATH)
    t_eval = np.linspace(0, 1, N_SAMPLES)

    for epoch in range(epoch, epoch + config["epochs"]):
        for batch in range(config["batches"]):
            ics, trajectories = gen_data(batch_size=BATCH_SIZE, min_radius=MIN_RADIUS)

            norm_ics = ics / torch.from_numpy(R).float()
            norm_trajectories = trajectories / torch.from_numpy(R).float()

            spacing_comp = compute_spacing(
                norm_trajectories,
                min_spacing=MIN_SPACING,
                max_spacing=MAX_SPACING,
                n_samples=N_SAMPLES,
            )

            spacing_pred = knot_model(norm_ics)
            spacing_loss = ((spacing_pred - spacing_comp) ** 2).sum()

            knot_positions, n_knots_predicted = spacing_to_knots(
                spacing_pred, N_INTERIOR_KNOTS
            
            )
            knot_vector = build_knot_vector(knot_positions, SPLINE_DEGREE)
            predicted_controlpoints, aux_losses = cp_model(knot_vector, ics)
            norm_pred_cp = predicted_controlpoints / torch.from_numpy(R).float()
            traj_loss, traj_loss_norm,  norm_pred_trajectories = compute_trajectory_loss(
                norm_pred_cp,
                knot_vector,
                norm_trajectories,
                weighting_vector=spacing_pred.detach(),
                d=SPLINE_DEGREE,
            )

            predicted_trajectories = (
                norm_pred_trajectories * torch.from_numpy(R).float()
            )

            #phys_loss_pred = physics_loss(
            #    predicted_controlpoints, knot_vector, SPLINE_DEGREE,
            #    torch.linspace(0, 1, N_SAMPLES), T=TIME_HORIZON
            # )
            
            #true_controlpoints = compute_controlpoints_from_trajectory(trajectories, knot_vector, SPLINE_DEGREE, N_CP_MAX)
                
            true_endpoints_diff = (
                norm_trajectories[:, [0, -1], 1::2].diff(dim=1).squeeze()
            )
            pred_integrated_derivatives = (
                norm_pred_trajectories[:, :, ::2].sum(dim=1)
                * TIME_HORIZON
                / (N_SAMPLES - 1)
            )

            int_loss = torch.relu(
            (true_endpoints_diff - pred_integrated_derivatives) ** 2 - 0.01
            ).mean()
            ic_loss = (((predicted_controlpoints[:, 0, :] - ics)) ** 2).mean()
            branch_loss = BRANCH_SMOOTH_WEIGHT * aux_losses["branch_smooth"]
            
            # Trunk auxiliary losses
            trunk_loss = (
                TRUNK_RECON_WEIGHT * aux_losses["recon"]
                + TRUNK_SMOOTH_WEIGHT * aux_losses["smooth"]
                + TRUNK_ORTHO_WEIGHT * aux_losses["ortho"]
            )
            total_cp_loss = traj_loss + traj_loss_norm + 0.1*ic_loss  + branch_loss  + trunk_loss  + int_loss + traj_loss_norm
            optimizer_cp.zero_grad()
            total_cp_loss.backward()
            #spacing_loss.backward()
            optimizer_cp.step()

            print(
                f"\rEpoch {epoch:04d}/{batch:04d} SpacingLoss: {spacing_loss:.5f} | Traj: {traj_loss:.5f}/{traj_loss_norm:.5f} Trunk: {trunk_loss:.5f} Branch: {branch_loss:.5f} IC: {ic_loss:.7f} INT_LOSS {int_loss:.7f}", 
                # INT: {int_loss:.7f}|phys_loss {phys_loss_pred}",
                end="",
            )

        print()

        plot(
            "trajectory",
            epoch,
            trajectories,
            spacing_pred,
            spacing_comp,
            knot_positions,
            SPLINE_DEGREE,
            N_DIM,
            BATCH_SIZE,
            t_eval,
            spacing_loss,
            knot_vector=knot_vector,
            predicted_controlpoints=predicted_controlpoints,
            pred_trajectories=predicted_trajectories,
            has_controlpoints=True,
        )

        #if (epoch - 10) % 50 == 0 and epoch > 1:
            #breakpoint()
    


    save_checkpoint(cp_model, optimizer_cp, epoch, total_cp_loss, CP_CHECKPOINT_PATH)
    save_checkpoint(knot_model, optimizer_cp, epoch, total_cp_loss, KNOT_CHECKPOINT_PATH)

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================


def main():
    """Main training script."""
    print("=" * 60)
    print("B-Spline Knot Placer Training")
    print("=" * 60)

    # Validate configuration
    if N_DIM != 12:
        raise ValueError("The number of state space dimensions for quadrotor is 12.")

    print(f"Configuration: N_DIM={N_DIM}, BATCH_SIZE={BATCH_SIZE}")
    print(f"Spline: degree={SPLINE_DEGREE}, n_cp_max={N_CP_MAX}")
    print(f"Time: horizon={TIME_HORIZON}s, samples={N_SAMPLES}")

    # Initialize ODE system
    ode_sys = ODESYS.QUADROTOR
    ode_func = ode_sys.get_ode()

    # Create data generator
    gen_data = partial(
        trajectory_generator,
        n_dim=N_DIM,
        A=A,
        R=R,
        tt=TT,
        ode_func=ode_func,
    )

    # Initialize knot model
    knot_model = ActivityModel(
        dim=N_DIM,
        R=R,
        Ns=N_SAMPLES,
        hidden=KNOT_MODEL_HIDDEN,
        kernel_size=KNOT_MODEL_KERNEL_SIZE,
        max_spacing=MAX_SPACING,
        min_spacing=MIN_SPACING,
        n_samples=N_SAMPLES,
    )
    optimizer_knot = torch.optim.Adam(knot_model.parameters(), lr=KNOT_MODEL_LR)

    # Train knot model
    train_knot_model(
        knot_model,
        optimizer_knot,
        gen_data,
        config={"epochs": KNOT_MODEL_EPOCHS, "batches": KNOT_MODEL_BATCHES},
    )

    # Initialize control point model
    cp_model = CPDeepONetModel(
        CP_WIDTHS_IC,
        widths_kcp=CP_WIDTHS_KCP,
        dim=N_DIM,
        R=R,
        n_cp=N_CP_MAX,
        order=SPLINE_DEGREE,
        rank=CP_RANK,
        metapos=CP_CHANNELS,
    )
    
    
    optimizer_cp = torch.optim.Adam(
        [
            {"params": cp_model.parameters(), "lr": CP_MODEL_LR},
            {"params": knot_model.parameters(), "lr": KNOT_MODEL_LR_JOINT},
        ]
    )
    epoch =0
    if Path(CP_CHECKPOINT_PATH).is_file():
        epoch, _ = load_checkpoint(cp_model, optimizer_cp, CP_CHECKPOINT_PATH)

    # Pretrain trunk encoder-decoder
    pretrain_trunk(
        cp_model,
        knot_model,
        gen_data,
        epoch,
        config={
            "epochs": TRUNK_PRETRAIN_EPOCHS,
            "batches": TRUNK_PRETRAIN_BATCHES,
            "lr": TRUNK_PRETRAIN_LR,
        },
    )


    # Train control point model
    train_cp_model(
        cp_model,
        knot_model,
        optimizer_cp,
        gen_data,
        epoch,
        config={"epochs": CP_MODEL_EPOCHS, "batches": CP_MODEL_BATCHES},
    )

    print("Training complete!")


if __name__ == "__main__":
    main()
