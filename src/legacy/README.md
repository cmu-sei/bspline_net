# Legacy code

Code from an earlier paper, kept for reference. Cleanup was done mostly very far after the fact.
We believe much of this code is related to the following paper

R. Romagnoli, J. Ratchford and M. H. Klein, "Building Hybrid B-Spline And Neural Network Operators,"
2024 IEEE 63rd Conference on Decision and Control (CDC), Milan, Italy, 2024, pp. 3611-3617,
doi: 10.1109/CDC56724.2024.10886426.

and to early development of the IEEE 2026 and TMLR paper. This code is not installed, not tested, and does not run on this branch. The older `.ini`-based pipeline that
this code belongs to is preserved on the `ieee_paper` git branch.

| Path | Contents |
|---|---|
| `scripts/` | Data generation, training, evaluation and plotting scripts |
| `net.py`, `ff_net.py` | Network definitions |
| `analysis/traj_cp_analysis.py`, `configs/cp_analysis.yaml` | Embedding plots for an earlier control-point model |
| `configs/optuna_cp.yaml`, `configs/optuna_joint.yaml` | Hyper-parameter sweep configs |
| `test_dataloading.py`, `test_modelloading.py`, `tests/` | Earlier smoke and unit tests |
| `benchmark_quadrotor.py` | Timing script for the quadrotor dynamics |
| `vis.py` | Training-diagnostics plot function for the old pipeline |

`src/bsplinenet/cp_model.py` and `src/bsplinenet/joint_model.py` are also retired but remain in the package
because `train.py` imports them.
