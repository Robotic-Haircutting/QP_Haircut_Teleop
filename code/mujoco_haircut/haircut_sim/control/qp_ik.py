"""Quadratic-programming inverse kinematics (QP-IK) for Piper end-effector tracking. To change the algorithm, edit this file.

Decoupled from lerobot: depends only on numpy and this package's piper_kinematics / qp_solver, and can be tested offline on its own (tests/test_qp_ik.py).

Solved once per control cycle (velocity-level / differential IK):

    decision variable  Δq ∈ R^6 (joint increment this cycle, rad)

    min   ½‖W (J Δq − K e)‖²          end-effector tracking: move the end effector by K·e this step
        + ½ λ ‖Δq‖²                   damping: suppress large joint velocities near singularities
        + ½ μ ‖Δq − (q_rest − q)‖²    posture preference: drift slowly toward q_rest in redundant/insensitive directions (μ=0 disables)

    s.t.  q_min − q ≤ Δq ≤ q_max − q   joint angle limits (including safety margins)
          |Δq| ≤ q̇_max · dt            joint velocity limits

    e = [p_d − p ;  log(R_d Rᵀ)]      position error (m) + orientation error (rotation vector, rad)

In standard form  min ½ Δqᵀ H Δq + gᵀ Δq,  lb ≤ Δq ≤ ub:
    H = Jᵀ W² J + λ I + μ I
    g = −Jᵀ W² K e − μ (q_rest − q)

Beware of the wrist singularity: at J5 = 0 the J4 and J6 axes are collinear and orientation tracking degrades badly (in mock tests the position error went from 0.2 mm to 12 mm).
So the qp-mode start pose must keep J5 away from 0; the default is [0, 50, -50, 0, 30, 0].

To add a cost term: accumulate it onto H and g in build_qp().
To add general linear inequality constraints (e.g. A Δq ≤ b for self-collision/workspace limits), the box solver is not enough:
switch the solver to a general QP (e.g. osqp / quadprog) and also return A, b from build_qp().
"""

from dataclasses import dataclass, field

import numpy as np

from .piper_kinematics import piper_fk_jacobian
from .qp_solver import solve_box_qp
from .transforms import matrix_to_rotvec


@dataclass
class QPIKParams:
    # ---- Cost weights ----
    # Weights of the position / orientation errors (diagonal of W). Orientation is in rad; 0.1 means 1 rad of orientation error ≈ 0.1 m of position error (position first)
    w_pos: float = 1.0
    w_rot: float = 0.1
    # Fraction K (0-1) of the error corrected per cycle; 1 means in one step, smaller is smoother
    gain: float = 0.8
    # Damping λ (larger is less erratic near singularities, but tracking is slower)
    damping: float = 1e-4
    # Posture preference μ and preferred joint angles (deg); μ=0 disables
    posture_weight: float = 0.0
    q_rest_deg: list[float] = field(default_factory=lambda: [0.0, 50.0, -50.0, 0.0, 30.0, 0.0])

    # ---- Constraints ----
    # Joint angle ranges (deg), [[min, max]] × 6
    joint_limits_deg: list[list[float]] = field(
        default_factory=lambda: [[-150, 150], [15, 150], [-160, -15], [-100, 100], [-50, 50], [-100, 100]]
    )
    # Joint velocity limit (deg/s), a single number or 6 numbers
    max_joint_speed_deg_s: float | list[float] = 90.0

    # ---- Solving ----
    # Linearize-and-solve repetitions per control cycle (1 is usually enough; 2-3 converge better for large errors)
    iterations: int = 1


class QPIK:
    def __init__(self, params: QPIKParams | None = None, solver=solve_box_qp):
        """solver(H, g, lb, ub, x0) -> (x, n_iter); can be replaced by any QP solver with this interface."""
        self.p = params or QPIKParams()
        self.solver = solver
        self.last_info: dict = {}

    def build_qp(self, q: np.ndarray, p_d: np.ndarray, R_d: np.ndarray, dt: float):
        """Build this cycle's QP. q: current joint angles (rad). Returns H, g, lb, ub and the error e."""
        P = self.p
        T, J = piper_fk_jacobian(q)
        e = np.concatenate([p_d - T[:3, 3], matrix_to_rotvec(R_d @ T[:3, :3].T)])

        W2 = np.diag([P.w_pos] * 3 + [P.w_rot] * 3) ** 2
        n = q.size
        H = J.T @ W2 @ J + P.damping * np.eye(n)
        g = -J.T @ W2 @ (P.gain * e)
        if P.posture_weight > 0.0:
            H += P.posture_weight * np.eye(n)
            g += -P.posture_weight * (np.deg2rad(P.q_rest_deg) - q)

        lim = np.deg2rad(np.asarray(P.joint_limits_deg, dtype=float))
        vmax = np.deg2rad(np.broadcast_to(np.asarray(P.max_joint_speed_deg_s, dtype=float), (n,))) * dt
        lb = np.maximum(lim[:, 0] - q, -vmax)
        ub = np.minimum(lim[:, 1] - q, vmax)
        # If the current angle is already out of range (e.g. right after start-up), allow moving back into range instead of jumping back
        lb = np.minimum(lb, 0.0)
        ub = np.maximum(ub, 0.0)
        return H, g, lb, ub, e

    def step(self, q: np.ndarray, p_d: np.ndarray, R_d: np.ndarray, dt: float) -> np.ndarray:
        """Given the current joint angles q (rad) and the target end-effector pose, return the next cycle's joint command (rad)."""
        q = np.asarray(q, dtype=float).copy()
        q_start = q.copy()
        iters = 0
        for _ in range(max(1, self.p.iterations)):
            # With multiple iterations the velocity limit applies to the total for this cycle, so subtract what has already been moved
            H, g, lb, ub, e = self.build_qp(q, p_d, R_d, dt)
            moved = q - q_start
            vmax = np.deg2rad(np.broadcast_to(np.asarray(self.p.max_joint_speed_deg_s, float), q.shape)) * dt
            lb = np.minimum(np.maximum(lb, -vmax - moved), 0.0)
            ub = np.maximum(np.minimum(ub, vmax - moved), 0.0)
            dq, it = self.solver(H, g, lb, ub, np.zeros_like(q))
            iters += it
            q = q + dq
        self.last_info = {"err_pos_m": float(np.linalg.norm(e[:3])), "err_rot_rad": float(np.linalg.norm(e[3:])), "qp_iters": iters}
        return q
