"""SuperMimic world model: predicts frame-local increments of every state component (paper §3.2, Table A.4)."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

from ..core.nets import ACTIVATIONS, mlp, run_mlp
from ..core.normalizer import RunningNorm
from ..core.rotations import so3_exp, so3_log, yaw_mat
from ..robots.g1 import NUM_J
from ..state import CONTROL_DT, INV_DIM, XI_DIM, invariant_features
from .rigidformer import RigidFormerNet

WM_OUT = 3 + 3 + 3 + 3 + NUM_J + NUM_J


class IncrementWorldModel(nn.Module):
    """Deterministic dynamics s_{t+1} = f(s_t, a_t, xi) predicting frame-local state increments.

    ``structured=True`` changes only the final layer's target layout: velocities remain increments,
    while positions/rotation/joint positions are residual corrections to a semi-implicit trapezoidal
    drift. ``targets`` keeps the plain increment definition for callers that need physical increments;
    fitting uses ``regression_targets`` and error statistics use ``inc_norm``.
    """

    DRIFT_NEW_VELOCITY_WEIGHT = 0.625

    def __init__(self, hidden=(2048, 2048, 2048), act: str = "silu", layer_norm: bool = True, rigidformer: dict | None = None,
                 structured: bool = False, dt: float = CONTROL_DT):
        super().__init__()
        if dt <= 0:
            raise ValueError(f"IncrementWorldModel dt must be positive, got {dt}")
        self.structured, self.dt = bool(structured), float(dt)
        if rigidformer is None:
            self.in_dim = INV_DIM + NUM_J + XI_DIM
            self.net = mlp(self.in_dim, list(hidden), WM_OUT, act=ACTIVATIONS[act], layer_norm=layer_norm)
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
            self.in_norm = RunningNorm(self.in_dim)
        else:  # (dp, dth, dv, dw) from the pelvis token, (dq, dqd) from the joint tokens
            self.net = RigidFormerNet(root_out=12, joint_out=2, **rigidformer)
            self.in_dim, self.in_norm = self.net.in_dim, self.net.make_input_norm()
        self.out_norm = RunningNorm(WM_OUT, eps=1e-4)
        # Error statistics remain in physical increment units even though a structured output layer
        # predicts velocity increments plus position residuals.
        self.inc_norm = RunningNorm(WM_OUT, eps=1e-4)
        self.amp = True  # bf16 autocast in step (if enabled); --wm_bptt_fp32 turns it off during the policy's BPTT

    def inputs(self, S, a, xi_n):
        if isinstance(self.net, RigidFormerNet):
            return self.net.inputs(S, a, xi_n)
        return torch.cat((invariant_features(S), a, xi_n), -1)

    @staticmethod
    def targets(S, S1):
        """Frame-local physical increments from S to S1."""
        p, R, v, w, q, qd = S
        p1, R1, v1, w1, q1, qd1 = S1
        H = yaw_mat(R)
        Rt = R.transpose(-1, -2)
        return torch.cat(
            (
                (H.transpose(-1, -2) @ (p1 - p).unsqueeze(-1)).squeeze(-1),
                so3_log(Rt @ R1),
                (Rt @ (v1 - v).unsqueeze(-1)).squeeze(-1),
                (Rt @ (w1 - w).unsqueeze(-1)).squeeze(-1),
                q1 - q,
                qd1 - qd,
            ),
            -1,
        )

    def _drift(self, S, next_v, next_w, next_qd):
        """Semi-implicit position drift in each increment's local frame.

        Four control-substeps with a linearly changing velocity give the new velocity weight
        ``(n + 1) / (2n) = 0.625``. The angular drift is expressed in the old root frame, matching
        the right multiplication used by ``integrate``.
        """
        _, R, v, w, _, qd = S
        H = yaw_mat(R)
        old_w = self.dt * (1.0 - self.DRIFT_NEW_VELOCITY_WEIGHT)
        new_w = self.dt * self.DRIFT_NEW_VELOCITY_WEIGHT
        return (
            (H.transpose(-1, -2) @ (old_w * v + new_w * next_v).unsqueeze(-1)).squeeze(-1),
            (R.transpose(-1, -2) @ (old_w * w + new_w * next_w).unsqueeze(-1)).squeeze(-1),
            old_w * qd + new_w * next_qd,
        )

    def regression_targets(self, S, S1):
        """The final network layer's targets, in the layout consumed by ``integrate``."""
        inc = self.targets(S, S1)
        if not self.structured:
            return inc
        dp, dth, dv, dw, dq, dqd = inc.split((3, 3, 3, 3, NUM_J, NUM_J), -1)
        pos_drift, rot_drift, q_drift = self._drift(S, S1[2], S1[3], S1[5])
        return torch.cat((dp - pos_drift, dth - rot_drift, dv, dw, dq - q_drift, dqd), -1)

    def integrate(self, S, d):
        """Integrate a denormalized output in the layout of ``regression_targets``."""
        p, R, v, w, q, qd = S
        H = yaw_mat(R)
        dp, dth, dv, dw, dq, dqd = d.split((3, 3, 3, 3, NUM_J, NUM_J), -1)
        v1 = v + (R @ dv.unsqueeze(-1)).squeeze(-1)
        w1 = w + (R @ dw.unsqueeze(-1)).squeeze(-1)
        qd1 = qd + dqd
        if self.structured:
            pos_drift, rot_drift, q_drift = self._drift(S, v1, w1, qd1)
            dp, dth, dq = dp + pos_drift, dth + rot_drift, dq + q_drift
        p1 = p + (H @ dp.unsqueeze(-1)).squeeze(-1)
        R1 = R @ so3_exp(dth)
        return p1, R1, v1, w1, q + dq, qd1

    def step(self, S, a, xi_n, input_noise: float = 0.0):
        """``input_noise``: std of Gaussian noise on the normalized inputs (only while fitting, as a regularizer)."""
        x = self.in_norm(self.inputs(S, a, xi_n))
        if input_noise != 0:
            x = x + input_noise * torch.randn_like(x)
        y = run_mlp(self.net, x, amp=self.amp)
        d = y * self.out_norm.std() + self.out_norm.mean
        return self.integrate(S, d)

    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor], strict: bool = True, assign: bool = False):
        """Load pre-structured checkpoints; copy increment statistics from ``out_norm`` for old unstructured ones."""
        values = dict(state_dict)
        missing_inc = not any(key.startswith("inc_norm.") for key in values)
        if missing_inc and not self.structured:
            for suffix in ("mean", "var", "count"):
                values[f"inc_norm.{suffix}"] = values[f"out_norm.{suffix}"]
        return super().load_state_dict(values, strict=strict, assign=assign)

    @staticmethod
    def state_error(S_hat, S):
        """Frame-local state error components (same layout as ``targets``)."""
        p, R, v, w, q, qd = S
        ph, Rh, vh, wh, qh, qdh = S_hat
        H = yaw_mat(R)
        Rt = R.transpose(-1, -2)
        return torch.cat(
            (
                (H.transpose(-1, -2) @ (ph - p).unsqueeze(-1)).squeeze(-1),
                so3_log(Rt @ Rh),
                (Rt @ (vh - v).unsqueeze(-1)).squeeze(-1),
                (Rt @ (wh - w).unsqueeze(-1)).squeeze(-1),
                qh - q,
                qdh - qd,
            ),
            -1,
        )
