"""
bspline_net (Github)

Copyright 2026 Carnegie Mellon University.

NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

Licensed under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.

[DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.

This Software includes and/or makes use of Third-Party Software each subject to its own license.

DM26-0750
"""
# PAPER: deployment (ONNX export)
"""Export TrajectoryModel -> ONNX for Raspberry Pi latency benchmarking."""

import torch
import torch.nn as nn
import torch.nn.functional as F
class SpectralDecoBaseline(nn.Module):
    """Export-safe, *unoptimized* baseline for SpectralDeco.forward.

    """

    def __init__(self, spec):  # spec: trained SpectralDeco
        super().__init__()
        self.N = spec.N
        with torch.no_grad():
            M = torch.complex(spec.M_real, spec.M_imag)
            Q = torch.matrix_exp(M - M.conj().T)  # (N, N) complex
            Qr, Qi = Q.real, Q.imag

            log_lambda = torch.complex(spec.log_lambda_real, spec.log_lambda_imag)
            lambda_t = torch.exp(spec.t.unsqueeze(1) * log_lambda.unsqueeze(0))  # (T, N)
            ltT = lambda_t.T.contiguous()  # (N, T)

        self.register_buffer("Qr", Qr.contiguous())      # (N, N)
        self.register_buffer("Qi", Qi.contiguous())      # (N, N)
        self.register_buffer("ltT_r", ltT.real.contiguous())  # (N, T)
        self.register_buffer("ltT_i", ltT.imag.contiguous())  # (N, T)

    def forward(self, phi):  # phi: (B, N) real -> (B, 2N, T) real
        # coeffs = phi @ conj(Q) = phi @ (Qr - i Qi)
        cr = phi @ self.Qr  # (B, N)
        ci = -(phi @ self.Qi)  # (B, N)

        # evolved = coeffs.unsqueeze(-1) * lambda_t.T.unsqueeze(0), complex
        cr_, ci_ = cr.unsqueeze(-1), ci.unsqueeze(-1)  # (B, N, 1)
        evolved_r = cr_ * self.ltT_r.unsqueeze(0) - ci_ * self.ltT_i.unsqueeze(0)  # (B, N, T)
        evolved_i = cr_ * self.ltT_i.unsqueeze(0) + ci_ * self.ltT_r.unsqueeze(0)  # (B, N, T)

        # out = Q @ evolved (complex matmul), applied via einsum like the original
        out_r = torch.einsum("ij,bjt->bit", self.Qr, evolved_r) - torch.einsum("ij,bjt->bit", self.Qi, evolved_i)
        out_i = torch.einsum("ij,bjt->bit", self.Qr, evolved_i) + torch.einsum("ij,bjt->bit", self.Qi, evolved_r)

        return torch.cat([out_r, out_i], dim=1)  # (B, 2N, T), real

class SpectralDecoONNX(nn.Module):
    """Real-valued, export-safe replacement for SpectralDeco.forward.

    Built from a trained SpectralDeco. Precomputes the constant complex
    objects (Q, lambda_t) and stores real/imag parts as buffers, so the
    forward pass contains no complex ops.

    Two optimizations are folded in, both exact:
    1. The ic_embed slice and the (bias-free, linear) decoders are absorbed
       into the eigenbasis projection: a single (n_out, 2N) matrix P maps
       evolved coefficients straight to [ic_embed, corr, lo_raw, hi_raw].
    2. The diagonal time evolution is commuted through P, so no (B, N, T)
       intermediates are ever materialized: per sample we column-scale P by
       the coefficients (small, (n_out, N)) and multiply by the constant
       lambda tables. Minimizes memory traffic for bandwidth-bound targets.
    """

    def __init__(self, spec, decoders, n_dim):  # spec: trained SpectralDeco
        super().__init__()
        self.n_dim = n_dim
        with torch.no_grad():
            # Q = matrix_exp(M - M^H), unitary, constant at inference.
            M = torch.complex(spec.M_real, spec.M_imag)
            Q = torch.matrix_exp(M - M.conj().T)  # (N, N) complex
            Qr, Qi = Q.real, Q.imag
            N = Qr.shape[0]

            # lambda_t = exp(t * log_lambda) -> stored transposed, (N, T).
            log_lambda = torch.complex(spec.log_lambda_real, spec.log_lambda_imag)
            lambda_t = torch.exp(spec.t.unsqueeze(1) * log_lambda.unsqueeze(0))  # (T, N)
            ltT = lambda_t.T.contiguous()  # (N, T)

            # Fold consumers of (out_r | out_i) through Q. With
            # out_r = Qr@er - Qi@ei and out_i = Qr@ei + Qi@er:
            #   ic_embed = out_r[:n_dim]
            #   decoder(dec_in) = V @ [out_r[n_dim:]; out_i],  V = dec.weight
            rows_r = [Qr[:n_dim]]  # acts on er
            rows_i = [-Qi[:n_dim]]  # acts on ei
            for dec in decoders:
                V = dec.weight  # (n_dim, 2N - n_dim)
                Va, Vb = V[:, : N - n_dim], V[:, N - n_dim :]
                rows_r.append(Va @ Qr[n_dim:] + Vb @ Qi)  # (n_dim, N)
                rows_i.append(Vb @ Qr - Va @ Qi[n_dim:])
            P1 = torch.cat(rows_r)  # (n_out, N), er side
            P2 = torch.cat(rows_i)  # (n_out, N), ei side

        self.register_buffer("Qr", Qr.contiguous())  # (N, N)
        self.register_buffer("Qi", Qi.contiguous())
        self.register_buffer("ltT_r", ltT.real.contiguous())  # (N, T)
        self.register_buffer("ltT_i", ltT.imag.contiguous())
        self.register_buffer("P1", P1.contiguous())  # (n_out, N)
        self.register_buffer("P2", P2.contiguous())

    def forward(self, phi):  # phi: (B, N) real
        # coeffs = phi @ conj(Q) = phi @ (Qr - i Qi)
        cr = phi @ self.Qr  # (B, N)
        ci = -(phi @ self.Qi)
        # z = P1@er + P2@ei with er/ei = diag-scaled lambda tables; commuting
        # the diagonal through P gives per-sample column-scaled projections:
        cr, ci = cr.unsqueeze(1), ci.unsqueeze(1)  # (B, 1, N)
        G1 = self.P1 * cr + self.P2 * ci  # (B, n_out, N)
        G2 = self.P2 * cr - self.P1 * ci
        return G1 @ self.ltT_r + G2 @ self.ltT_i  # (B, n_out, T)


class TrajectoryModelONNX(nn.Module):
    def __init__(self, model, baseline=True):
        super().__init__()
        self.n_dim = model.n_dim
        self.ic_lift = model.ic_lift
        self.quantile_levels = model.quantile_levels is not None
        self.baseline = baseline

        decoders = [model.decoder]
        if self.quantile_levels:
            decoders += [model.decoder_lo, model.decoder_hi]

        if baseline:
            self.rollout = SpectralDecoBaseline(model.ic_rollout)
            self.decoder = model.decoder
            self.decoder_lo = model.decoder_lo
            self.decoder_hi = model.decoder_hi
        else:
            self.rollout = SpectralDecoONNX(model.ic_rollout, decoders, model.n_dim)

    def forward(self, norm_ics):
        extras = self.ic_lift(norm_ics)
        phi = torch.cat([norm_ics, extras], dim=-1)
        nd = self.n_dim

        if self.baseline:
            rolled = self.rollout(phi).permute(0, 2, 1)  # (B, T, 2N)
            ic_embed = rolled[:, :, :nd]
            decoder_input = rolled[:, :, nd:]
            correction = self.decoder(decoder_input)
            correction = correction - correction[:, :1, :]
            median = ic_embed + correction
            if not self.quantile_levels:
                return median
            lo_raw = self.decoder_lo(decoder_input)
            lo = median - F.relu(lo_raw - lo_raw[:, :1, :])
            hi_raw = self.decoder_hi(decoder_input)
            hi = median + F.relu(hi_raw - hi_raw[:, :1, :])
            return lo, median, hi

        z = self.rollout(phi).permute(0, 2, 1)  # (B, T, n_out) -- already folded
        ic_embed = z[:, :, :nd]
        corr = z[:, :, nd:2*nd]
        corr = corr - corr[:, :1, :]
        median = ic_embed + corr
        if not self.quantile_levels:
            return median
        lo_raw = z[:, :, 2*nd:3*nd]
        lo = median - F.relu(lo_raw - lo_raw[:, :1, :])
        hi_raw = z[:, :, 3*nd:]
        hi = median + F.relu(hi_raw - hi_raw[:, :1, :])
        return lo, median, hi
    

def export(model, n_dim, path="traj.onnx", opset=17):
    """model: trained TrajectoryModel (eval mode). Writes ONNX to `path`."""
    model.eval()
    wrapper = TrajectoryModelONNX(model).eval()
    dummy = torch.randn(1, n_dim)

    out_names = ["lo", "mean", "hi"] if wrapper.quantile_levels else ["traj"]
    torch.onnx.export(
        wrapper,
        dummy,
        path,
        input_names=["x0"],
        output_names=out_names,
        dynamic_axes={"x0": {0: "batch"}, **{n: {0: "batch"} for n in out_names}},
        opset_version=opset,
    )
    print(f"exported -> {path}")
    return wrapper
