"""MRNN solver for Piper end-effector tracking (momentum recurrent neural network + majorization-minimization MM), without the CNO collaboration framework.

Method from: H. Huang, L. Jin, Z. Zeng, "A Momentum Recurrent Neural Network for Sparse Motion Planning of
Redundant Manipulators With Majorization-Minimization", IEEE TIE, vol. 72, no. 12, 2025. Equation numbers follow the paper.

-- To change the algorithm, edit this file; decoupled from lerobot, offline self-test in tests/test_mrnn_ik.py.

Each control cycle (dt) solves for the joint velocity ẋ ∈ R^n (n=6):

  Sparse motion planning (4):
      min  ½ ẋᵀ W ẋ + ε‖ẋ‖_p              (0<p<1, ε is the weight of the sparsity term)
      s.t. J ẋ = ẏ_d
           ẋ⁻ ≤ ẋ ≤ ẋ⁺

  MM handles the box constraint (5)(6): ρ/2 ‖ẋ − P_C(ẋ(k))‖² replaces the constraint ẋ ∈ C, where P_C is the Euclidean projection onto the box C
  The stationarity condition (7) is written as the nonlinear equations (8)(9):
      A z = b(z),   A = [[ρI, Jᵀ], [J, 0]],   z = [ẋ; λ]
      b(z) = [ρ P_C(ẋ) − ∇φ(ẋ);  ẏ_d],   ∇φ(ẋ) = W ẋ + ε ∇‖ẋ‖_p   (10)(11)

  MRNN (17):
      z(k+1) = s(k) − τ d(z(k)),   d(z) = A⁻¹ (A z − b(z))
      s(k+1) = z(k+1) + ω (z(k+1) − z(k))            (ω < 1 is the momentum coefficient, Theorem 1)

Correspondence with teleoperation:
  ẏ_d = K e / dt, e = [p_d − p ; log(R_d Rᵀ)] (task dimension m set by task: "pose" m=6, "position" m=3)
  Box C: joint speed limits combined with joint ranges, ẋ⁻ = max(−q̇_max, (q_min − q)/dt), ẋ⁺ = min(q̇_max, (q_max − q)/dt)
  Next joint angles q + ẋ·dt. The MM penalty is only a soft constraint, so the output is projected onto box C once more to stay within bounds.

Note: Piper has 6 axes, so with task="pose" (m=6) J is square with no redundancy; the equality constraint essentially fixes ẋ and the sparsity term and W have almost no effect;
with task="position" (m=3) there are 3 redundant DOFs, so the sparsity term ‖ẋ‖_p can keep some joints still (the paper's setting), but orientation is not tracked.
"""

from dataclasses import dataclass, field

import numpy as np

from .piper_kinematics import piper_fk_jacobian
from .transforms import matrix_to_rotvec


@dataclass
class MRNNIKParams:
    # ---- Task ----
    # "pose": position + orientation as equality constraints (m=6, Piper has no redundancy)
    # "position": only position as an equality constraint (m=3, 3 redundant DOFs, sparsity term active); orientation not tracked
    # "position_orient": position as an equality constraint, orientation as a quadratic tracking term ½ w_orient ‖J_ω ẋ − ω_d‖² in φ, using the redundant DOFs to follow orientation
    task: str = "position_orient"
    # Weight of the orientation tracking term in position_orient mode
    w_orient: float = 1.0
    # Scale of the orientation error in ẏ_d (pose only); 1 means weighted equally with position
    w_rot: float = 1.0
    # Fraction K of the error corrected per cycle (ẏ_d = K e / dt)
    gain: float = 0.8

    # ---- Optimization problem (4) ----
    # W = w_diag * I (weight of the quadratic joint-velocity term)
    w_diag: float = 1.0
    # Weight ε and norm order p (0<p<1) of the sparsity term ε‖ẋ‖_p; ε=0 disables it
    sparse_weight: float = 0.0
    p_norm: float = 0.5
    # Smoothing threshold of Eq. (11) for |ẋ_i| ≤ δ (avoids |ẋ|^{p-1} diverging at 0)
    sparse_delta: float = 1e-3
    # MM penalty coefficient ρ (larger stays closer to the box constraint)
    rho: float = 50.0

    # ---- MRNN (17) ----
    # Step size τ, momentum ω (ω<1), iterations per control cycle
    tau: float = 1.0
    momentum: float = 0.8
    iterations: int = 60
    # Stop early when z changes less than this between iterations
    tol: float = 1e-9

    # ---- Constraints ----
    joint_limits_deg: list[list[float]] = field(
        default_factory=lambda: [[-150, 150], [15, 150], [-160, -15], [-100, 100], [-50, 50], [-100, 100]]
    )
    max_joint_speed_deg_s: float | list[float] = 90.0


def _grad_lp(v: np.ndarray, p: float, delta: float) -> np.ndarray:
    """∂‖v‖_p / ∂v, Eq. (11): analytic gradient for |v_i| > δ, linear smoothing for |v_i| ≤ δ."""
    a = np.abs(v)
    s = np.sum(np.maximum(a, delta) ** p) ** ((1.0 - p) / p)
    g = s * np.maximum(a, delta) ** (p - 1.0) * np.sign(v)
    small = a <= delta
    g[small] = s * delta ** (p - 1.0) * v[small] / delta
    return g


class MRNNIK:
    def __init__(self, params: MRNNIKParams | None = None):
        self.p = params or MRNNIKParams()
        if self.p.task not in ("pose", "position", "position_orient"):
            raise ValueError(f"task must be pose, position or position_orient, got {self.p.task}")
        if not self.p.momentum < 1.0:
            raise ValueError("MRNN requires momentum coefficient ω < 1 (Theorem 1 of the paper)")
        self.last_info: dict = {}
        self._xdot = None  # previous cycle's solution, used as the next cycle's initial value (warm start)

    def build(self, q: np.ndarray, p_d: np.ndarray, R_d: np.ndarray, dt: float):
        """Build this cycle's J, ẏ_d and box C. Returns (J, yd, lo, hi, e)."""
        P = self.p
        T, J6 = piper_fk_jacobian(q)
        e = np.concatenate([p_d - T[:3, 3], matrix_to_rotvec(R_d @ T[:3, :3].T)])
        if P.task == "pose":
            J = J6.copy()
            J[3:] *= P.w_rot
            yd = P.gain * np.concatenate([e[:3], P.w_rot * e[3:]]) / dt
        else:
            J = J6[:3]
            yd = P.gain * e[:3] / dt
        # Secondary orientation task (position_orient): add ½ w_orient ‖J_ω ẋ − ω_d‖² to φ
        self._Jw = J6[3:] if P.task == "position_orient" else None
        self._wd = P.gain * e[3:] / dt
        lim = np.deg2rad(np.asarray(P.joint_limits_deg, dtype=float))
        vmax = np.deg2rad(np.broadcast_to(np.asarray(P.max_joint_speed_deg_s, dtype=float), q.shape))
        lo = np.maximum(-vmax, (lim[:, 0] - q) / dt)
        hi = np.minimum(vmax, (lim[:, 1] - q) / dt)
        # Allow moving back into range when already out of range
        lo = np.minimum(lo, 0.0)
        hi = np.maximum(hi, 0.0)
        return J, yd, lo, hi, e

    def solve(self, J: np.ndarray, yd: np.ndarray, lo: np.ndarray, hi: np.ndarray, x0=None):
        """MRNN (17) solves the nonlinear equations (8). Returns (ẋ, number of iterations)."""
        P = self.p
        m, n = J.shape
        W = P.w_diag * np.eye(n)
        A = np.block([[P.rho * np.eye(n), J.T], [J, np.zeros((m, m))]])
        A_inv = np.linalg.pinv(A)  # equals A⁻¹ when J has full rank; pinv is more robust near singularities

        def b_of(z):
            xd = z[:n]
            grad_phi = W @ xd
            if self._Jw is not None:
                grad_phi = grad_phi + P.w_orient * self._Jw.T @ (self._Jw @ xd - self._wd)
            if P.sparse_weight > 0.0:
                grad_phi = grad_phi + P.sparse_weight * _grad_lp(xd, P.p_norm, P.sparse_delta)
            return np.concatenate([P.rho * np.clip(xd, lo, hi) - grad_phi, yd])

        z = np.zeros(n + m)
        if x0 is not None:
            z[:n] = np.clip(x0, lo, hi)
        s = z.copy()
        it = 0
        for it in range(1, P.iterations + 1):
            d = A_inv @ (A @ z - b_of(z))  # d(z(k)) = A⁻¹(A z(k) − b)
            z_new = s - P.tau * d  # z(k+1) = s(k) − τ d(z(k))
            s = z_new + P.momentum * (z_new - z)  # s(k+1) = z(k+1) + ω (z(k+1) − z(k))
            done = np.max(np.abs(z_new - z)) < P.tol
            z = z_new
            if done:
                break
        return z[:n], it

    def step(self, q: np.ndarray, p_d: np.ndarray, R_d: np.ndarray, dt: float) -> np.ndarray:
        """Same interface as QPIK.step: takes the current joint angles q (rad) and the target pose, returns the next cycle's joint angles (rad)."""
        q = np.asarray(q, dtype=float)
        J, yd, lo, hi, e = self.build(q, p_d, R_d, dt)
        xdot, it = self.solve(J, yd, lo, hi, self._xdot)
        viol = float(np.max(np.maximum(lo - xdot, 0.0) + np.maximum(xdot - hi, 0.0)))
        xdot = np.clip(xdot, lo, hi)  # MM is a soft constraint; project once at the end to stay within bounds
        self._xdot = xdot
        self.last_info = {
            "err_pos_m": float(np.linalg.norm(e[:3])),
            "err_rot_rad": float(np.linalg.norm(e[3:])),
            "qp_iters": it,
            "box_violation_rad_s": viol,
        }
        return q + xdot * dt
