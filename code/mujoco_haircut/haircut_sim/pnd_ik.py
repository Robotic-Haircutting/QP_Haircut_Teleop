"""Velocity-level box QP solved by projected neural dynamics (PND).

u = qdot (rad/s); q_next = q + u * dt. Solver pseudo-time is
independent of robot dt. H/f are divided by lambda_max(H), preserving
the QP minimizer, to stabilize Euler integration of the projected ODE.
No momentum, MM, sparse penalty, or external QP solver is used.
"""
from dataclasses import dataclass, field
import numpy as np
from .control.piper_kinematics import piper_fk_jacobian
from .control.transforms import matrix_to_rotvec

@dataclass
class PNDIKParams:
    gain: float = 2.0  # feedback gain, 1/s; v = gain * pose_error
    w_pos: float = 1.0
    w_rot: float = 0.1
    damping: float = 0.01  # w in H = J.T W^2 J + w I
    relaxation: float = 1.0  # gamma * pseudo-time step, (0, 1]
    iterations: int = 500
    tol: float = 1e-6  # normalized projected residual, rad/s
    max_joint_speed_deg_s: float = 45.0
    joint_limits_deg: list[list[float]] = field(default_factory=lambda:
        [[-150,150],[15,150],[-160,-15],[-100,100],[-50,50],[-100,100]])

class PNDIK:
    def __init__(self, params=None):
        self.p = params or PNDIKParams()
        p = self.p
        if not (0 < p.relaxation <= 1) or p.damping <= 0 or p.gain <= 0 or p.iterations < 1 or p.tol <= 0 or p.max_joint_speed_deg_s <= 0:
            raise ValueError('Invalid PND gains, damping, iteration count or speed limit')
        if p.w_pos <= 0 or p.w_rot < 0:
            raise ValueError('Invalid task weights')
        self._u = None
        self.last_info = {}

    def build_qp(self, q, p_d, R_d, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be positive and finite')
        T, J = piper_fk_jacobian(q)
        e = np.r_[p_d - T[:3,3], matrix_to_rotvec(R_d @ T[:3,:3].T)]
        weights = np.array([self.p.w_pos]*3 + [self.p.w_rot]*3)
        JW = weights[:,None] * J
        vW = weights * (self.p.gain * e)
        H = JW.T @ JW + self.p.damping * np.eye(len(q))
        f = -JW.T @ vW
        limits = np.deg2rad(np.asarray(self.p.joint_limits_deg, float))
        if limits.shape != (len(q), 2) or not np.all(np.isfinite(limits)) or np.any(limits[:,0] > limits[:,1]):
            raise ValueError('Invalid joint angle limits')
        vmax = np.deg2rad(self.p.max_joint_speed_deg_s)
        lo = np.maximum(-vmax, (limits[:,0] - q)/dt)
        hi = np.minimum(vmax, (limits[:,1] - q)/dt)
        if np.any(lo > hi):
            raise ValueError('Current joints outside limits: no feasible one-cycle command')
        return H, f, lo, hi, e

    def solve(self, H, f, lo, hi, x0=None):
        if not all(np.all(np.isfinite(a)) for a in (H,f,lo,hi)) or np.any(lo > hi):
            raise ValueError('Nonfinite QP data or empty bounds')
        scale = float(np.linalg.eigvalsh(H)[-1])
        if scale <= 0:
            raise ValueError('H must be positive definite')
        Hn, fn = H/scale, f/scale
        u = np.clip(np.zeros_like(f) if x0 is None else x0, lo, hi)
        for it in range(1, self.p.iterations + 1):
            residual = u - np.clip(u - (Hn @ u + fn), lo, hi)
            if np.max(np.abs(residual)) <= self.p.tol:
                break
            u = u - self.p.relaxation * residual
        residual = u - np.clip(u - (Hn @ u + fn), lo, hi)
        self.last_info = {'qp_iters': it,
            'projected_residual_rad_s': float(np.max(np.abs(residual))),
            'converged': bool(np.max(np.abs(residual)) <= self.p.tol)}
        return np.clip(u, lo, hi), it

    def step(self, q, p_d, R_d, dt):
        q = np.asarray(q, float)
        H, f, lo, hi, e = self.build_qp(q, np.asarray(p_d), R_d, dt)
        u, _ = self.solve(H, f, lo, hi, self._u)
        self._u = u
        self.last_info.update(err_pos_m=float(np.linalg.norm(e[:3])),
            err_rot_rad=float(np.linalg.norm(e[3:])))
        return q + u * dt
