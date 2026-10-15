"""
bspline_net (Github)

Copyright 2026 Carnegie Mellon University.

NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

Licensed under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.

[DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.

This Software includes and/or makes use of Third-Party Software each subject to its own license.

DM26-0750
"""

import argparse
from typing import Optional
from bsplinenet.ff_net import gen_data_np, ODESYS
from quadrotor.quadrotor import jax_quadrotor_dx, load_matrix_K, quadrotor_dx
from scipy.integrate import odeint
import numpy as np
from matplotlib import pyplot as plt
from netCDF4 import Dataset
import scipy
from bsplinenet.Bspline_functions import Bs_comp, BsKnots, duffing_sys, leastSquaresInversion

def plot_trajectory_comparison(
    nc_file: str,
    ic_index: int = 0,
    knot_index: int = 0, 
    coords_to_plot: Optional[list] = None,
    sampling_interval: Optional[float] = None,
):
    nc = Dataset(nc_file, "r")
    if coords_to_plot is None:
        coords_to_plot = list(range(sd))
    
    system_name = nc.getncattr("system")
    n_cp = nc.getncattr("n_control")
    bs = nc.getncattr("bspline_order")
    sd = nc.getncattr("n_state_dimension")
    ti = nc.getncattr("time_interval")
    si = nc.getncattr("sampling_interval")
    if sampling_interval is not None:
        si = sampling_interval
    use_integrated = bool(nc.getncattr("use_integrated"))
    
    X = nc.variables["X"][:]
    Y = nc.variables["Y"][:]
    K = nc.variables["Knots"][:]
    nc.close()
    
    if system_name.lower() != 'quadrotor':
        raise NotImplementedError("Only know quadrotor")
    ode_sys = ODESYS.QUADROTOR 
    ode_func = ode_sys.get_ode()
    
    x0 = X[knot_index, ic_index, :]
    coeffs = Y[knot_index, ic_index, :]
    knots_unique = K[knot_index, ic_index, :]

    Ln = np.zeros(n_cp + bs + 1)
    Ln[bs:-bs] = knots_unique
    Ln[-bs:] = 1

    Ns = int(ti / si) + 1
    tt = np.linspace(0, ti, Ns)
    
    tk, Ln, Bit = BsKnots(n_cp, bs, Ns, Ln)
    C = scipy.linalg.block_diag(*[Bit for _ in range(sd)])
    
    ym_fit = C @ coeffs
    traj_fit = ym_fit.reshape((Ns, sd), order='F')
    x1 = np.zeros(12) 
    traj_true = ode_func(x0, tt, (x1,))
    
    error = traj_fit - traj_true
    rmse_per_coord = np.sqrt(np.mean(error**2, axis=0))
    
    # Compute Greville abscissae (control point locations in normalized time)
    greville = np.zeros(n_cp)
    for i in range(n_cp):
        greville[i] = np.mean(Ln[i+1 : i+bs+1])
    # Scale to actual time
    greville_t = greville * ti
    
    n_plots = len(coords_to_plot)
    n_q = sd // 2
    
    pos_labels = ['x', 'dx', 'y', 'dy', 'z', 'dz', 'theta', 'dtheta', 'phi', 'dphi', 'omega', 'domega']
    
    fig, axes = plt.subplots(n_plots, 2, figsize=(12, 3 * n_plots))
    if n_plots == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(
        f'{system_name} (use_integrated={use_integrated})\n'
        f'n_cp={n_cp}, degree={bs}, IC #{ic_index}',
        fontsize=12
    )
    
    for i, coord in enumerate(coords_to_plot):
        # Extract control points for this coordinate
        cp_coord = coeffs[coord * n_cp : (coord + 1) * n_cp]
        
        ax = axes[i, 0]
        ax.plot(tt, traj_true[:, coord], 'k-', label='True', linewidth=2)
        ax.plot(tt, traj_fit[:, coord], 'r--', label='Fitted', linewidth=1.5)
        ax.plot(greville_t, cp_coord, 'bo', markersize=3, label='Control pts', alpha=0.3)
        ax.set_xlabel('Time')
        ax.set_ylabel(pos_labels[coord])
        ax.set_title(f'{pos_labels[coord]}: RMSE = {rmse_per_coord[coord]:.6f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[i, 1]
        ax.plot(tt, error[:, coord], 'b-', linewidth=1)
        ax.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time')
        ax.set_ylabel(f'Error')
        ax.set_title(f'{pos_labels[coord]}: Error over time')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    print('DONE')
    plt.savefig("test.png")
    return fig, traj_true, traj_fit, error


def plot_kinematic_consistency(
    nc_file: str,
    ode_registry: dict,
    ic_index: int = 0,
    knot_index: int = 0,
):
    nc = Dataset(nc_file, "r")
    
    system_name = nc.getncattr("system")
    n_cp = nc.getncattr("n_control")
    bs = nc.getncattr("bspline_order")
    sd = nc.getncattr("n_state_dimension")
    ti = nc.getncattr("time_interval")
    si = nc.getncattr("sampling_interval")
    use_integrated = bool(nc.getncattr("use_integrated"))
    
    Y = nc.variables["Y"][:]
    K = nc.variables["Knots"][:]
    
    nc.close()
    
    coeffs = Y[knot_index, ic_index, :]
    knots_interior = K[knot_index, ic_index, :]
    
    d = bs - 1
    Ln = np.zeros(n_cp + d + 1)
    Ln[:d + 1] = 0
    Ln[-d - 1:] = 1
    Ln[d:d + 1 + len(knots_interior)] = knots_interior
    
    Ns = int(ti / si) + 1
    tt = np.linspace(0, ti, Ns)
    dt = tt[1] - tt[0]
    
    tk, Ln, Bit = BsKnots(n_cp, d, Ns, Ln)
    C = scipy.linalg.block_diag(*[Bit for _ in range(sd)])
    
    ym_fit = C @ coeffs
    traj_fit = ym_fit.reshape((sd, Ns), order='F').T
    
    n_q = sd // 2
    q_fit = traj_fit[:, :n_q]
    qdot_fit = traj_fit[:, n_q:]
    
    qdot_numerical = np.gradient(q_fit, dt, axis=0)
    consistency_error = qdot_fit - qdot_numerical
    
    pos_labels = ['x', 'y', 'z', 'θ', 'φ', 'ω'][:n_q]
    vel_labels = ['ẋ', 'ẏ', 'ż', 'θ̇', 'φ̇', 'ω̇'][:n_q]
    
    fig, axes = plt.subplots(n_q, 2, figsize=(12, 2.5 * n_q))
    if n_q == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(
        f'{system_name}: Kinematic Consistency (use_integrated={use_integrated})',
        fontsize=12
    )
    
    for i in range(n_q):
        ax = axes[i, 0]
        ax.plot(tt, qdot_fit[:, i], 'b-', label=f'Fitted {vel_labels[i]}', linewidth=1.5)
        ax.plot(tt, qdot_numerical[:, i], 'r--', label=f'd/dt({pos_labels[i]})', linewidth=1.5)
        ax.set_xlabel('Time')
        ax.set_ylabel(vel_labels[i])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[i, 1]
        ax.plot(tt, consistency_error[:, i], 'k-', linewidth=1)
        ax.axhline(0, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time')
        ax.set_ylabel(f'{vel_labels[i]} - d/dt({pos_labels[i]})')
        rmse = np.sqrt(np.mean(consistency_error[:, i]**2))
        ax.set_title(f'RMSE = {rmse:.6f}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, consistency_error

if __name__ == "__main__":
    
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--file", required=True, help="nc file")
    parser.add_argument("-i","--index", required=False, default = 0, type=int,   help="ic inidex")
    parser.add_argument("-k","--knot", required=False, default = 0, type=int, help="knot index")
    parser.add_argument("-d","--dim", required = False, nargs='+',default = [0], type=int,  help="dimension to plot")
    parser.add_argument("-s","--sampling-interval", required = False, default =0.01, type=float, help = "sampling interval")
    
    
    args = parser.parse_args()
    plot_trajectory_comparison(nc_file = args.file, 
                               ic_index = args.index, 
                               knot_index = args.knot, 
                               coords_to_plot = args.dim, 
                               sampling_interval = args.sampling_interval)