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
import os
import time

import numba as nb
import numpy as np

# Precompute constants once
_M = 0.6
_g = 9.81
_m = 0.410
_R = 0.0503513
_L = 0.2159 / 2
_m_prop = 0.00311
_m_m = 0.036 + _m_prop

_Jx = (2 * _m * _R) / 5 + 2 * (_L**2) * _m_m
_Jy = (2 * _m * _R) / 5 + 2 * (_L**2) * _m_m
_Jz = (2 * _m * _R) / 5 + 4 * (_L**2) * _m_m
_ff = _M * _g

_inv_M = 1.0 / _M
_inv_Jx = 1.0 / _Jx
_inv_Jy = 1.0 / _Jy
_inv_Jz = 1.0 / _Jz
_Jy_Jz_over_Jx = (_Jy - _Jz) / _Jx
_Jz_Jx_over_Jy = (_Jz - _Jx) / _Jy
_Jx_Jy_over_Jz = (_Jx - _Jy) / _Jz


@nb.njit(cache=True)
def quadrotor_dx_fast(x, t, x_sp, K):
    # Control: cu = -K @ (x - x_sp), then cu[0] += ff
    diff = x - x_sp
    cu0 = _ff
    cu1 = 0.0
    cu2 = 0.0
    cu3 = 0.0
    for j in range(12):
        d = diff[j]
        cu0 -= K[0, j] * d
        cu1 -= K[1, j] * d
        cu2 -= K[2, j] * d
        cu3 -= K[3, j] * d

    phi = x[7]
    theta = x[9]

    p = x[6]
    q = x[8]
    r = x[10]

    u = x[0]
    v = x[2]
    w = x[4]

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    du = r * v - q * w - _g * sin_theta
    dv = p * w - r * u + _g * cos_theta * np.sin(phi)
    dw = q * u - p * v + _g * cos_theta * np.cos(phi) - _inv_M * cu0
    dp = _Jy_Jz_over_Jx * q * r + _inv_Jx * cu1
    dq = _Jz_Jx_over_Jy * p * r + _inv_Jy * cu2
    dr = _Jx_Jy_over_Jz * p * q + _inv_Jz * cu3

    dx = np.empty(12)
    dx[0] = du
    dx[1] = x[0]
    dx[2] = dv
    dx[3] = x[2]
    dx[4] = dw
    dx[5] = x[4]
    dx[6] = dp
    dx[7] = x[6]
    dx[8] = dq
    dx[9] = x[8]
    dx[10] = dr
    dx[11] = x[10]
    return dx


@nb.njit(cache=True, parallel=True)
def integrate_batch(ics, tt, K, substeps=4):
    """RK4 with `substeps` internal steps per reported interval. Output at tt only."""
    B, T = ics.shape[0], tt.shape[0]
    x_sp = np.zeros(12)
    out = np.empty((B, T, 12))
    for i in nb.prange(B):
        x = ics[i].copy()
        out[i, 0] = x
        for k in range(T - 1):
            h = (tt[k + 1] - tt[k]) / substeps  # smaller internal step
            t = tt[k]
            for _ in range(substeps):  # integrate finely...
                k1 = quadrotor_dx_fast(x, t, x_sp, K)
                k2 = quadrotor_dx_fast(x + 0.5 * h * k1, t + 0.5 * h, x_sp, K)
                k3 = quadrotor_dx_fast(x + 0.5 * h * k2, t + 0.5 * h, x_sp, K)
                k4 = quadrotor_dx_fast(x + h * k3, t + h, x_sp, K)
                x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                t += h
            out[i, k + 1] = x  # ...but store only at tt
    return out


if __name__ == "__main__":
    # Warm up (trigger compilation)
    x0 = np.zeros(12)
    x0[5] = 0.4  # some altitude
    x_sp = np.zeros(12)
    K = np.zeros((4, 12))  # adjust to your actual K shape
    _ = quadrotor_dx_fast(x0, 0.0, x_sp, K)

    # Benchmark single call
    N = 100_000
    start = time.perf_counter()
    for _ in range(N):
        quadrotor_dx_fast(x0, 0.0, x_sp, K)
    elapsed = time.perf_counter() - start
    print(f"Single call: {elapsed / N * 1e6:.2f} mu s")

    # Benchmark full RK4 trajectory (more realistic)
    def rk4_step(x, t, dt, x_sp, K):
        k1 = quadrotor_dx_fast(x, t, x_sp, K)
        k2 = quadrotor_dx_fast(x + 0.5 * dt * k1, t + 0.5 * dt, x_sp, K)
        k3 = quadrotor_dx_fast(x + 0.5 * dt * k2, t + 0.5 * dt, x_sp, K)
        k4 = quadrotor_dx_fast(x + dt * k3, t + dt, x_sp, K)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # JIT the RK4 step too
    rk4_step_jit = nb.njit(cache=True)(rk4_step)
    _ = rk4_step_jit(x0, 0.0, 0.01, x_sp, K)  # warmup

    dt = 0.005
    N_steps = int(2.5 / dt)

    start = time.perf_counter()
    x = x0.copy()
    for i in range(N_steps):
        x = rk4_step_jit(x, i * dt, dt, x_sp, K)
    elapsed = time.perf_counter() - start
    print(f"RK4 step: {elapsed / N_steps * 1e6:.2f} mu s")
    print(f"Full {N_steps}-step trajectory: {elapsed * 1e3:.2f} ms")
