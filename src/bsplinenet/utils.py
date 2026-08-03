"""
bspline_net (Github)

Copyright 2026 Carnegie Mellon University.

NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

Licensed under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.

[DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.

This Software includes and/or makes use of Third-Party Software each subject to its own license.

DM26-0750
"""
from enum import Enum, auto
from functools import partial
from pathlib import Path

import numpy as np
import scipy.io as sci
import torch
from scipy.integrate import odeint

from .Bspline_functions import Bs_comp, BsKnots, duffing_sys, leastSquaresInversion

# from .quadrotor import gen_ic_ellips
# from .quadrotor import load_matrix_K#, quadrotor_dx #, jax_quadrotor_dx,
from .quadrotor import integrate_batch, quadrotor_dx_fast


class ODESYS(Enum):
    DUFFING = auto()
    QUADROTOR = auto()
    QUADROTOR_FAST = auto()
    #    QUADROTOR_JAX = auto()

    def __str__(self):
        return str(self.name).lower()

    def get_ode(self):
        if self == ODESYS["DUFFING"]:
            return partial(odeint, duffing_sys)
        # if self == ODESYS["QUADROTOR"]:
        #    K = load_matrix_K()
        #    return partial(odeint, partial(quadrotor_dx, K=K))
        if self == ODESYS["QUADROTOR_FAST"]:
            K = load_matrix_K()
            return partial(odeint, partial(quadrotor_dx_fast, K=K))

    def get_batch_integrator(self):
        if self == ODESYS["QUADROTOR_FAST"]:
            K = load_matrix_K()
            return partial(integrate_batch, K=K)
        return None  # other systems fall back to per-trajectory odeint


#        if self == ODESYS["QUADROTOR_JAX"]:
#            K = load_matrix_K()
#            return partial(odeint, partial(jax_quadrotor_dx, K=K))


def generate_knot_vector(n_cp, d, min_factor=0.2, max_factor=3.0):
    n_interior = n_cp - d - 1
    avg_spacing = 1.0 / (n_interior + 1)
    min_spacing = avg_spacing * min_factor
    max_spacing = avg_spacing * max_factor

    # Random parameters for sinusoidal density
    peakiness = np.random.uniform(0.5, 5)
    phase = np.random.uniform(0, 2 * np.pi)
    base_freq = np.random.uniform(0.5, 3)
    n_harmonics = np.random.randint(1, 5)

    # Generate density from sinusoids
    x = np.linspace(0, 1, 1000)
    density = np.ones_like(x)

    for k in range(1, n_harmonics + 1):
        freq = base_freq * k
        weight = 1.0 / k
        density += weight * (np.sin(2 * np.pi * freq * x + phase) ** 2) * peakiness

    # Sample from density via CDF inversion
    density = density / np.sum(density)
    cdf = np.cumsum(density)
    cdf = cdf / cdf[-1]

    uniform_samples = np.sort(np.random.uniform(0, 1, n_interior))
    interior = np.interp(uniform_samples, cdf, x)

    # Enforce spacing constraints
    for _ in range(100):
        spacings = np.diff(interior)

        too_large = np.where(spacings > max_spacing)[0]
        if len(too_large) > 0:
            gap_idx = too_large[0]
            new_point = (interior[gap_idx] + interior[gap_idx + 1]) / 2

            remove_idx = np.argmin(spacings) + 1
            interior = np.delete(interior, remove_idx)
            interior = np.sort(np.append(interior, new_point))
            continue

        too_small = np.where(spacings < min_spacing)[0]
        if len(too_small) > 0:
            idx = too_small[0]
            mid = (interior[idx] + interior[idx + 1]) / 2
            interior[idx] = mid - min_spacing / 2
            interior[idx + 1] = mid + min_spacing / 2
            interior = np.clip(interior, 0, 1)
            interior = np.sort(interior)
            continue

        break

    # Build full knot vector
    Ln = np.zeros(n_cp + d + 1)
    Ln[: d + 1] = 0
    Ln[-d - 1 :] = 1
    Ln[d + 1 : -d - 1] = interior

    return Ln


def gen_SVD_of_Phi(
    ode_sys: ODESYS,
    n_cp: int,
    d: int,
    tt: np.ndarray,
    x0: np.ndarray,
    randomize_knots: bool = False,
    use_integrated: bool = False,  # NEW
):
    N_ic, n = x0.shape
    Ns = tt.size

    ode_func = ode_sys.get_ode()

    if randomize_knots:
        Ln = generate_knot_vector(n_cp, d, min_factor=0.1, max_factor=4.0)
    else:
        Ln = np.zeros(n_cp + d + 1)
        Ln[-d - 1 :] = 1
        n_interior = n_cp - d - 1
        if n_interior > 0:
            Ln[d + 1 : -d - 1] = np.linspace(0, 1, n_interior + 2)[1:-1]

    tk, Ln, Bit = BsKnots(n_cp, d, Ns, Ln)
    C = scipy.linalg.block_diag(*[Bit for i in range(n)])

    Phi = C.T @ C
    U, S, V_T = np.linalg.svd(Phi, full_matrices=True)

    if use_integrated:
        n_q = n // 2

        t_normalized = np.linspace(0, 1, Ns)
        dt_norm = t_normalized[1] - t_normalized[0]

        Int_B = np.zeros_like(Bit)
        for i in range(1, Ns):
            Int_B[i, :] = Int_B[i - 1, :] + 0.5 * (Bit[i - 1, :] + Bit[i, :]) * dt_norm

        ti = tt[-1] - tt[0]
        Int_B *= ti

        # C_int only fits velocities: dq(t) = Bit @ c
        C_int = scipy.linalg.block_diag(*[Bit for _ in range(n_q)])

        Phi_int = C_int.T @ C_int
        U_int, S_int, V_T_int = np.linalg.svd(Phi_int, full_matrices=True)

        return U, S, V_T, C, Ln, C_int, U_int, S_int, V_T_int, Int_B

    return U, S, V_T, C, Ln


def gen_data_np(
    ode_sys: ODESYS,
    n_cp: int,
    d: int,
    tt: np.ndarray,
    x0: np.ndarray,
    x1: np.ndarray = np.array([]),
    randomize_knots: bool = False,  # randomize knot point vector
    n_rand_iterations: int = 2,  # number of randomized iterations.
    use_integrated: bool = False,
):
    N_ic, n = x0.shape
    Ns = tt.size

    if not randomize_knots:
        n_rand_iterations = 1

    x_tot = np.zeros((n_cp * (n + 1), N_ic, n_rand_iterations))

    if x1.size == 0:
        print(f"Warning: No set point for ode system {ode_sys}. setting to x0")
        x1 = x0

    if x1.size == n:
        print("Single set point will be broadcast to shape of initial conditions array")
        x1 = np.broadcast_to(x1, x0.shape)

    if not x1.shape == x0.shape:
        raise ValueError(f"shapes x0.shape {x0.shape} and x1.shape {x1.shape} do not match")

    ode_func = ode_sys.get_ode()

    svd_saved = []
    if randomize_knots:
        print(f"Generating SVD for {n_rand_iterations} different randomized knot point vectors")
        for i in range(n_rand_iterations):
            print(f"{i}", end="\r")
            U, S, V_T, C, Ln, *rest = gen_SVD_of_Phi(
                ode_sys, n_cp, d, tt, x0, True, use_integrated
            )  # CHANGED
            svd_saved.append(
                {"U": U, "S": S, "V_T": V_T, "C": C, "Ln": Ln, "rest": rest}
            )  # CHANGED
    else:
        U, S, V_T, C, Ln, *rest = gen_SVD_of_Phi(
            ode_sys, n_cp, d, tt, x0, False, use_integrated
        )  # CHANGED
        svd_saved.append({"U": U, "S": S, "V_T": V_T, "C": C, "Ln": Ln, "rest": rest})  # CHANGED

    print()

    n_total = N_ic * n_rand_iterations
    n_params = n_cp * n

    rmse_all = np.zeros((N_ic, n_rand_iterations))
    param_errors_all = np.zeros((N_ic, n_rand_iterations, n_params))

    print("Progress:    ", end="")
    for enum, svd_info in enumerate(svd_saved):
        U = svd_info["U"]
        S = svd_info["S"]
        V_T = svd_info["V_T"]
        C = svd_info["C"]
        Ln = svd_info["Ln"]
        rest = svd_info["rest"]  # NEW

        for j in range(0, N_ic):

            xx = ode_func(x0[j, :], tt, (x1[j, :],))
            ym = xx.ravel(order="F")

            if use_integrated:
                C_int, U_int, S_int, V_T_int, Int_B = rest
                n_q = n // 2

                # Extract observed velocities and positions
                qdot_obs = xx[:, 0::2]  # (Ns, n_q) - velocities at even indices
                q_obs = xx[:, 1::2]  # (Ns, n_q) - positions at odd indices

                ym_int = qdot_obs.ravel(order="F")  # fit only velocities

                # Solve for velocity coefficients
                Psi_int = C_int.T @ ym_int
                c = V_T_int.T @ (U_int.T @ Psi_int / S_int)

                # Reconstruct velocities
                qdot_fit = (C_int @ c).reshape((Ns, n_q), order="F")

                # Integrate to get positions
                q0 = x0[j, 1::2]  # initial positions at odd indices
                Int_B_full = scipy.linalg.block_diag(*[Int_B for _ in range(n_q)])
                q_fit = q0 + (Int_B_full @ c).reshape((Ns, n_q), order="F")

                # Recombine into full state (interleaved)
                xx_fit = np.zeros((Ns, n))
                xx_fit[:, 0::2] = qdot_fit  # velocities at even indices
                xx_fit[:, 1::2] = q_fit  # positions at odd indices

                ym_ext = xx_fit.ravel(order="F")
                Psi = C.T @ ym_ext
                x = V_T.T @ (U.T @ Psi / S)
            else:
                Psi = C.T @ ym
                x = V_T.T @ (U.T @ Psi / S)

            x_tot[:-n_cp, j, enum] = x
            x_tot[-n_cp:, j, enum] = Ln[d - 1 : -d + 1]

            y_pred = C @ x
            residuals = ym - y_pred
            rss = np.sum(residuals**2)

            rmse_all[j, enum] = np.sqrt(rss / len(ym))
            sigma2 = rss / (len(ym) - len(x))

            S_inv_sq = 1 / S**2
            param_errors_all[j, enum] = np.sqrt(sigma2 * np.sum((V_T.T**2) * S_inv_sq, axis=1))

            print(f"\r{enum*N_ic+j+1}/{n_total} ({100*(enum*N_ic+j+1)/n_total:.0f}%)", end="")

    print("\n")
    return x_tot, param_errors_all, rmse_all


def trajectory_generator(batch_size, n_dim, A, R, tt, ode_func, min_radius=0.1):
    """Generate a batch of initial conditions and trajectories."""
    ics_np = gen_ic_shell(n_dim, A, batch_size, min_radius=min_radius).T
    ics = torch.from_numpy(ics_np).to(torch.float32)

    trajectories = torch.stack(
        [
            torch.from_numpy(ode_func(ics[i, :], tt, (np.zeros(n_dim),)).astype(np.float32))
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


def gen_ic_ellips(n: int, A: np.ndarray, num_points: int):
    """Generate initial conditions on an ellips

    Args:
        n (int): number of dimensions
        A (np.ndarray[n,n]): Rotation/translation matrix
        num_points (int): number of points to create

    Returns:
        np.ndarray:num_points x n array
    """

    # Ellipsoid parameters (center and positive definite matrix)
    xc = np.zeros(n)  # Center of the ellipsoid, a 12-dimensional vector of zeros (change as needed)

    # Generate random points on a unit sphere (12-dimensional)
    points_on_unit_sphere = np.random.normal(size=(n, num_points))
    points_on_unit_sphere /= np.linalg.norm(points_on_unit_sphere, axis=0)

    # Scale the points to the ellipsoid's shape and orientation
    points_on_ellipsoid = np.sqrt(np.random.uniform(0, 1, num_points))
    points_on_ellipsoid = A @ (points_on_ellipsoid * points_on_unit_sphere) + xc[:, np.newaxis]

    return points_on_ellipsoid


MATRIX_K = None


def load_matrix_K(folder=(Path(__file__).parent / "mat/control.mat").resolve()):
    global MATRIX_K
    if MATRIX_K is None:
        # K = sci.loadmat('/home/raffaele/Documents/FALSA/mat/control.mat')['K']
        return sci.loadmat(folder)["K"]
    return MATRIX_K


def bspline_basis(t, knots, d):
    """Vectorized B-spline basis using Cox-de Boor recursion."""
    Ns = t.shape[0]
    n_knots = knots.shape[0]

    # Degree 0: (Ns, n_knots-1)
    # t is (Ns,), knots is (n_knots,)
    t_ = t.unsqueeze(1)  # (Ns, 1)
    k_ = knots.unsqueeze(0)  # (1, n_knots)

    left = t_ >= k_[:, :-1]  # (Ns, n_knots-1)
    right = t_ < k_[:, 1:]  # (Ns, n_knots-1)
    B = (left & right).to(t.dtype)
    B[:, -1] = torch.where(t == knots[-1], torch.ones_like(t), B[:, -1])

    for k in range(1, d + 1):
        # knots[i+k] - knots[i] for i in range(n_knots - 1 - k)
        n_basis = n_knots - 1 - k

        denom_left = knots[k : k + n_basis] - knots[:n_basis]  # (n_basis,)
        denom_right = knots[k + 1 : k + 1 + n_basis] - knots[1 : 1 + n_basis]  # (n_basis,)

        safe_left = denom_left.clamp(min=1e-10)
        safe_right = denom_right.clamp(min=1e-10)

        left_term = (t_ - knots[:n_basis].unsqueeze(0)) / safe_left.unsqueeze(0) * B[:, :n_basis]
        right_term = (
            (knots[k + 1 : k + 1 + n_basis].unsqueeze(0) - t_)
            / safe_right.unsqueeze(0)
            * B[:, 1 : 1 + n_basis]
        )

        # Zero out where denom was ~0
        left_term = left_term * (denom_left > 1e-10).unsqueeze(0)
        right_term = right_term * (denom_right > 1e-10).unsqueeze(0)

        B = left_term + right_term

    startpoint_mask = (t == knots[0]).unsqueeze(1)
    startpoint_basis = torch.zeros_like(B)
    startpoint_basis[:, 0] = 1.0
    B = torch.where(startpoint_mask, startpoint_basis, B)
    endpoint_mask = (t == knots[-1]).unsqueeze(1)
    endpoint_basis = torch.zeros_like(B)
    endpoint_basis[:, -1] = 1.0
    B = torch.where(endpoint_mask, endpoint_basis, B)
    return B


def odd_reflect_pad(data, pad_width):
    # data: (B, C, L)
    origin = data[..., :1]
    ending = data[..., -1:]
    first_pad_width = data[..., 1 : pad_width + 1]
    last_pad_width = data[..., -(pad_width + 1) : -1]
    lhs = 2 * origin - first_pad_width.flip(-1)
    rhs = 2 * ending - last_pad_width.flip(-1)

    return torch.cat([lhs, data, rhs], dim=-1)


def compute_activity(
    trajectories,
    time_horizon,
    sigma_time=0.01,  # seconds
    return_2nd_deriv=False,
    max_translational_rate: float = 1.0,
    max_roll_pitch_rate: float = 1.0,
    max_yaw_rate: float = 1.0,
):
    """Compute adaptive spacing based on trajectory activity."""
    # always do this - if velocities_only:
    trajectories = trajectories[..., [0, 2, 4, 6, 8, 10]].permute(
        0, 2, 1
    )  # (batch, n_dim//2, Ns+2k)
    batch, n_dim, Ns = trajectories.shape
    normalization = torch.tensor(
        [
            max_translational_rate,
            max_translational_rate,
            max_translational_rate,
            max_roll_pitch_rate,
            max_roll_pitch_rate,
            max_yaw_rate,
        ]
    )
    normalization = normalization.reshape(1, n_dim, 1)
    dt = time_horizon / (Ns - 1)

    sigma = sigma_time / dt

    # Create kernel
    k = int(4 * sigma + 1)
    trajectories = odd_reflect_pad(trajectories, k)
    t = torch.arange(-k, k + 1, dtype=trajectories.dtype, device=trajectories.device)
    constant = (t**2 - sigma**2) / sigma**4
    kernel = constant * torch.exp(-(t**2) / (2 * sigma**2))

    scale = (kernel * t**2).sum()
    kernel = kernel * (2.0 / scale) / dt**2
    kernel = kernel / kernel.sum()
    kernel = kernel.tile(n_dim).reshape(n_dim, -1)
    kernel = torch.einsum("ij,ik->ijk", torch.eye(n_dim), kernel)

    d2 = torch.nn.functional.conv1d(
        trajectories / normalization,
        kernel,
    )
    d2 = d2.permute(0, 2, 1)
    activity = d2.abs().max(dim=-1).values

    if return_2nd_deriv:
        return (activity, d2, trajectories)
    return activity


def density_to_knots(density, max_density, min_density, t_max):
    # max_density knots_per_second
    # min_density knots_per_second
    # t_max [seconds]
    batch, n_samples = density.shape
    t_eval = torch.linspace(0, 1, n_samples, device=density.device)

    interval_size = t_max / (n_samples - 1)  # [seconds]
    density = torch.clamp(density, min=min_density, max=max_density)
    max_knots_possible = int(max_density * t_max)

    accum = torch.floor(density.cumsum(dim=1) * interval_size)  # [knots/second] x [seconds] = knots
    diff = accum.diff(dim=1)
    mask = diff == 1
    loc_row, loc_col = torch.where(mask)
    # loc_row, loc_col = torch.where(accum.diff() ==1)

    n_knots = mask.sum(dim=1)
    # _, n_knots = torch.unique_consecutive(loc_row, return_counts=True)

    positions = mask.long().cumsum(dim=1) - 1
    positions = positions[loc_row, loc_col]
    # try:
    # positions = torch.cat([torch.arange(c) for c in n_knots])
    # except RuntimeError as k:
    #    print(k)
    #    print("Runtime error in Density to Knots!")
    #    print(torch.min(n_knots), 'min value')
    #    print(torch.argmin(n_knots), 'where it occurs')
    #    idx = torch.argmin(n_knots)
    #    print(density[idx])

    knot_positions = torch.ones(
        batch, max_knots_possible, device=density.device
    )  # (batch, n_knot_max)
    knot_positions[loc_row, positions] = t_eval[loc_col]
    pos_idx, pos_loc = torch.where(
        (knot_positions < 1) & (knot_positions > 1 - (t_max / max_density))
    )
    knot_positions[pos_idx, pos_loc] = knot_positions[pos_idx, pos_loc] - (t_max / max_density)

    # knot positions are on normalized scale from 0 to 1 (tmax)
    return knot_positions, n_knots


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


def compute_controlpoints_from_trajectory(trajectories, knots, d, max_n_cp, reconstruct_fit=False):

    batch, Ns, n_state = trajectories.shape
    t = torch.linspace(0, 1, Ns, device=knots.device)

    control_points = torch.zeros(batch, max_n_cp, n_state, device=trajectories.device)
    fits = torch.zeros_like(trajectories)

    for b in range(batch):
        # Same logic as compute_trajectory_loss
        end_knots = torch.where(knots[b] == 1)[0]
        end_point = len(knots[b]) if len(end_knots) <= d + 1 else end_knots[d + 1]

        # Number of control points for this batch element
        n_cp_b = end_point - d - 1

        # Bit = bspline_basis(t, knots[b, :end_point], d)
        # solution = torch.linalg.lstsq(Bit, trajectories[b]).solution
        # control_points[b, :n_cp_b, :] = solution

        Bit = bspline_basis(t, knots[b, :end_point], d)
        cp0 = trajectories[b, 0, :]
        residual = trajectories[b] - Bit[:, 0:1] @ cp0.unsqueeze(0)
        solution_rest = torch.linalg.lstsq(Bit[:, 1:n_cp_b], residual).solution
        control_points[b, 0, :] = cp0
        control_points[b, 1:n_cp_b, :] = solution_rest

        if not reconstruct_fit:
            continue

        # Reconstruct the fit at the same sample points
        # fits[b,:,:] = Bit @ solution
        fits[b, :, :] = Bit[:, :n_cp_b] @ control_points[b, :n_cp_b, :]

    if not reconstruct_fit:
        return control_points

    # Residuals and fit quality
    residuals = trajectories - fits
    mse = (residuals**2).mean()
    rmse = mse.sqrt()
    # R^2 score
    ss_res = (residuals**2).sum()
    ss_tot = ((trajectories - trajectories.mean(dim=0)) ** 2).sum()
    r_squared = 1 - ss_res / ss_tot

    print(f"  Batch {b}: RMSE={rmse:.6f}, R_squared={r_squared:.4f}", end="")

    return control_points, fits


def compute_trajectory_loss_terms(pred_trajectories, trajectories, weighting_vector=None):
    """Compute weighted MSE and std-normalized MSE between matched-shape trajectories.

    Args:
        pred_trajectories: (batch, n_t, n_dim)
        trajectories: (batch, n_t, n_dim), same shape
        weighting_vector: optional (batch, n_t) per-timestep weights

    Returns:
        (mse_loss, mse_loss_normalized)
    """
    sq_err = (pred_trajectories - trajectories) ** 2
    if weighting_vector is not None:
        w = weighting_vector.to(pred_trajectories.device).unsqueeze(-1)
        sq_err = sq_err * w
    norm = trajectories.std(dim=(0, 1), keepdim=True).clamp(min=1e-2)
    return sq_err.mean(), (sq_err / norm).mean()


def compute_trajectory_loss(pred_controlpoints, knots, trajectories, weighting_vector, d):
    """Compute weighted MSE loss between predicted and true trajectories."""
    batch, Ns, n_state = trajectories.shape
    t = torch.linspace(0, 1, Ns, device=knots.device)
    trajectories = trajectories.to(pred_controlpoints.device)
    ones = torch.tensor(1.0, device=knots.device)
    pred_list = []
    for b in range(batch):
        end_knots = torch.where(torch.isclose(knots[b], ones))[0]
        end_point = len(knots[b]) if len(end_knots) <= d + 1 else end_knots[d + 1]
        Bit = bspline_basis(t, knots[b, :end_point], d).to(pred_controlpoints.device)
        pred_list.append(Bit @ pred_controlpoints[b, : end_point - d - 1])

    pred_trajectories = torch.stack(pred_list)
    total_loss1, total_loss2 = compute_trajectory_loss_terms(
        pred_trajectories, trajectories, weighting_vector
    )
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
    masked_controlpoints[invalid_mask] = last_valid_cp.unsqueeze(1).expand(-1, n_cp, -1)[
        invalid_mask
    ]

    return masked_controlpoints, valid_mask, n_valid_cp
