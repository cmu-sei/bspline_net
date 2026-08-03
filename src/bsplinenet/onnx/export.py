"""
bspline_net (Github)

Copyright 2026 Carnegie Mellon University.

NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

Licensed under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.

[DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.

This Software includes and/or makes use of Third-Party Software each subject to its own license.

DM26-0750
"""
import hydra
import torch
from omegaconf import DictConfig

from bsplinenet.knot_model import LitKnotModel
from bsplinenet.onnx.knot_onnx import KnotModelONNX
from bsplinenet.onnx.trajectory_onnx import TrajectoryModelONNX
from bsplinenet.traj_model import LitTrajModel


@hydra.main(version_base=None, config_path="../../configs", config_name="export_config")
def main(cfg: DictConfig):
    opset = int(cfg.get("opset", 17))

    lit_t = LitTrajModel.load_from_checkpoint(
        cfg.trajectory_checkpoint, map_location="cpu", weights_only=False
    )
    n_dim = len(lit_t.cfg.data.R)
    wrap_t = TrajectoryModelONNX(lit_t.model).eval()
    out_names = ["lo", "med", "hi"] if wrap_t.quantile_levels else ["traj"]
    torch.onnx.export(
        wrap_t,
        torch.randn(1, n_dim),
        cfg.get("trajectory_out", "trajectory.onnx"),
        input_names=["x0"],
        output_names=out_names,
        dynamic_axes={"x0": {0: "batch"}, **{n: {0: "batch"} for n in out_names}},
        opset_version=opset,
        dynamo=True,
    )

    lit_k = LitKnotModel.load_from_checkpoint(
        cfg.knotpoint_checkpoint, map_location="cpu", weights_only=False
    )
    wrap_k = KnotModelONNX(lit_k.model).eval()
    torch.onnx.export(
        wrap_k,
        torch.randn(1, n_dim),
        cfg.get("knot_out", "knot.onnx"),
        input_names=["x0"],
        output_names=["activity"],
        dynamic_axes={"x0": {0: "batch"}, "activity": {0: "batch"}},
        opset_version=opset,
        dynamo=True,
    )

    try:
        import numpy as np
        import onnxruntime as ort

        for path in [
            cfg.get("trajectory_out", "trajectory.onnx"),
            cfg.get("knot_out", "knot.onnx"),
        ]:
            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            outs = sess.run(None, {"x0": np.random.randn(1, n_dim).astype(np.float32)})
            print(f"  smoke {path}: {[o.shape for o in outs]}")
    except ImportError:
        print("  (onnxruntime not installed; skip smoke test)")


if __name__ == "__main__":
    main()
