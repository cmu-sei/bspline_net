"""
bspline_net (Github)

Copyright 2026 Carnegie Mellon University.

NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

Licensed under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.

[DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.

This Software includes and/or makes use of Third-Party Software each subject to its own license.

DM26-0750
"""
import time

import numba as nb
import numpy as np

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

N_steps = 1000
dt = 0.001
start = time.perf_counter()
x = x0.copy()
for i in range(N_steps):
    x = rk4_step_jit(x, i * dt, dt, x_sp, K)
elapsed = time.perf_counter() - start
print(f"RK4 step: {elapsed / N_steps * 1e6:.2f} mu s")
print(f"Full {N_steps}-step trajectory: {elapsed * 1e3:.2f} ms")
