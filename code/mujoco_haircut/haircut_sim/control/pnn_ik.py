"""Projection neural network (PNN) IK for Piper end-effector tracking: the velocity-level QP is solved by a neurodynamic ODE, without any QP solver.

Each control cycle (Δt):

  Standard QP:
      min_u  ½ uᵀ H u + fᵀ u      s.t.  u⁻ ≤ u ≤ u⁺,      u = q̇
      H = Jᵀ J + w I_n,           f = −Jᵀ (ẋ_d − λ e)

  Projection neural network:
      u̇(t) = −γ [ u(t) − P_Ω( u(t) − (H u(t) + f) ) ],     Ω = [u⁻, u⁺]

  Integrating gives u*; the next joint command is q ← q + u*·Δt.

Correspondence with teleoperation (e = x − x_d is "actual − target", so ẋ_d − λe pulls the end effector back to the target):
  J        6×6 geometric Jacobian (position rows + orientation rows; orientation rows scaled by w_rot, w_rot=1 is the original form)
  ẋ_d      velocity feedforward of the target pose: [(p_d(k) − p_d(k−1))/Δt ; log(R_d(k) R_d(k−1)ᵀ)/Δt]
  e        [p − p_d ; −log(R_d Rᵀ)]   (position error + orientation error as a rotation vector)
  Ω        joint speed limits combined with joint ranges: u⁻ = max(−q̇_max, (q_min − q)/Δt), u⁺ = min(q̇_max, (q_max − q)/Δt)

The ODE is integrated with explicit Euler for substeps steps per control cycle (step h = Δt / substeps); the state u is kept across cycles (warm start),
so the network state evolves continuously with the task. For a stable Euler discretization, γh is limited to 1/max(1, λ_max(H)) each cycle using the largest eigenvalue of H.
To change the algorithm, edit this file; offline self-test: python tests/test_pnn_ik.py
"""

from dataclasses import dataclass, field

import numpy as np

from .piper_kinematics import piper_fk_jacobian
from .transforms import matrix_to_rotvec


@dataclass
class PNNIKParams:
    # ---- QP ----
    # error feedback gain λ (1/s)
    lam: float = 30.0
    # regularization w (H = JᵀJ + wI)
    w: float = 0.005
    # orientation row weight (scales the last three rows of J and of ẋ_d − λe; 1.0 = position and orientation weighted equally)
    w_rot: float = 1.0
    # scale of the target velocity feedforward ẋ_d (0 = off, 1 = original form)
    feedforward: float = 1.0

    # ---- Projection neural network ----
    # convergence gain γ (1/s)
    gamma: float = 3000.0
    # Euler integration steps per control cycle
    substeps: int = 50

    # ---- Constraints ----
    joint_limits_deg: list[list[float]] = field(
        default_factory=lambda: [[-150, 150], [15, 150], [-160, -15], [-100, 100], [-50, 50], [-100, 100]]
    )
    max_joint_speed_deg_s: float | list[float] = 90.0


class PNNIK:
    def __init__(self, params: PNNIKParams | None = None):
        self.p = params or PNNIKParams()
        self.last_info: dict = {}
        self._u = None  # network state u (kept across cycles)
        self._prev_target = None  # previous cycle's (p_d, R_d), for the velocity feedforward

    def build(self, q: np.ndarray, p_d: np.ndarray, R_d: np.ndarray, dt: float):
        """Return (H, f, u_lo, u_hi, e)."""
        P = self.p
        T, J = piper_fk_jacobian(q)
        e = np.concatenate([T[:3, 3] - p_d, -matrix_to_rotvec(R_d @ T[:3, :3].T)])  # e = x − x_d
        xd_dot = np.zeros(6)
        if self._prev_target is not None and P.feedforward > 0.0:
            p0, R0 = self._prev_target
            xd_dot = P.feedforward * np.concatenate([(p_d - p0) / dt, matrix_to_rotvec(R_d @ R0.T) / dt])
        self._prev_target = (np.array(p_d, dtype=float), np.array(R_d, dtype=float))

        S = np.diag([1.0, 1.0, 1.0, P.w_rot, P.w_rot, P.w_rot])
        Jw = S @ J
        n = q.size
        H = Jw.T @ Jw + P.w * np.eye(n)
        f = -Jw.T @ (S @ (xd_dot - P.lam * e))

        lim = np.deg2rad(np.asarray(P.joint_limits_deg, dtype=float))
        vmax = np.deg2rad(np.broadcast_to(np.asarray(P.max_joint_speed_deg_s, dtype=float), (n,)))
        lo = np.minimum(np.maximum(-vmax, (lim[:, 0] - q) / dt), 0.0)
        hi = np.maximum(np.minimum(vmax, (lim[:, 1] - q) / dt), 0.0)
        return H, f, lo, hi, e

    def solve(self, H, f, lo, hi, dt: float, u0=None) -> tuple[np.ndarray, float]:
        """Integrate u̇ = −γ[u − P_Ω(u − (Hu + f))] over one control cycle. Returns (u, final-step residual ‖u − P_Ω(u − (Hu+f))‖∞)."""
        P = self.p
        u = np.zeros_like(f) if u0 is None else np.clip(u0, lo, hi)
        h = dt / P.substeps
        gh = min(P.gamma * h, 1.0 / max(1.0, float(np.linalg.eigvalsh(H)[-1])))
        r = np.zeros_like(u)
        for _ in range(P.substeps):
            r = u - np.clip(u - (H @ u + f), lo, hi)
            u = u - gh * r
        return u, float(np.max(np.abs(r)))

    def reset(self) -> None:
        self._u, self._prev_target = None, None

    def step(self, q: np.ndarray, p_d: np.ndarray, R_d: np.ndarray, dt: float) -> np.ndarray:
        """Same interface as MRNNIK.step / QPIK.step: takes the current joint angles q (rad) and the target pose, returns the next cycle's joint angles (rad)."""
        q = np.asarray(q, dtype=float)
        H, f, lo, hi, e = self.build(q, p_d, R_d, dt)
        u, res = self.solve(H, f, lo, hi, dt, self._u)
        self._u = u
        u = np.clip(u, lo, hi)
        self.last_info = {
            "err_pos_m": float(np.linalg.norm(e[:3])),
            "err_rot_rad": float(np.linalg.norm(e[3:])),
            "qp_iters": self.p.substeps,
            "pnn_residual": res,
        }
        return q + u * dt
