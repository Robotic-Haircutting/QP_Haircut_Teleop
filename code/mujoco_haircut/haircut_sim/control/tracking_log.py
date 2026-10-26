"""Teleoperation tracking log: every frame writes the leader target rd and the actual Piper end effector r to a CSV, to be plotted later with tools/plot_tracking.py.

Fields per row (position in m, orientation as rotation vector in rad, joints in deg):
  t                      seconds since start
  rd_x rd_y rd_z rd_wx rd_wy rd_wz        target end-effector pose from the leader (converted to the Piper base frame and clipped to the workspace)
  rc_x ... rc_wz         end-effector pose from forward kinematics of the QP joint command (= tracking performance of the QP itself)
  r_x  ... r_wz          end-effector pose from forward kinematics of the joint angles fed back by Piper (= real tracking performance, including motor lag)
  e_pos_mm e_rot_deg     position error (mm) and orientation error (deg) of r relative to rd
  ec_pos_mm ec_rot_deg   error of rc relative to rd
  qc1..qc6 / q1..q6      joint commands / actual joint angles
  solve_ms qp_iters      QP solve time and active-set iterations for this frame
"""

import csv
import time
from pathlib import Path

import numpy as np

from .piper_kinematics import piper_fk
from .transforms import matrix_to_rotvec

_POSE = ["x", "y", "z", "wx", "wy", "wz"]
HEADER = (
    ["t"]
    + [f"rd_{k}" for k in _POSE]
    + [f"rc_{k}" for k in _POSE]
    + [f"r_{k}" for k in _POSE]
    + ["e_pos_mm", "e_rot_deg", "ec_pos_mm", "ec_rot_deg"]
    + [f"qc{i}" for i in range(1, 7)]
    + [f"q{i}" for i in range(1, 7)]
    + ["solve_ms", "qp_iters"]
)


def _pose(T):
    return np.concatenate([T[:3, 3], matrix_to_rotvec(T[:3, :3])])


def _err(Td_p, Td_R, T):
    e_p = np.linalg.norm(T[:3, 3] - Td_p) * 1000.0
    e_r = np.rad2deg(np.linalg.norm(matrix_to_rotvec(Td_R @ T[:3, :3].T)))
    return e_p, e_r


class TrackingLogger:
    def __init__(self, log_dir: str, prefix: str = "qp"):
        d = Path(log_dir)
        d.mkdir(parents=True, exist_ok=True)
        self.path = d / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        self._f = open(self.path, "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        self._w.writerow(HEADER)
        self._t0 = time.perf_counter()
        self._n = 0

    def log(self, p_d, R_d, q_cmd_rad, q_meas_deg, solve_ms: float, qp_iters: int) -> None:
        Tc = piper_fk(q_cmd_rad)
        Tm = piper_fk(np.deg2rad(q_meas_deg))
        rd = np.concatenate([p_d, matrix_to_rotvec(R_d)])
        e_m = _err(p_d, R_d, Tm)
        e_c = _err(p_d, R_d, Tc)
        row = [time.perf_counter() - self._t0, *rd, *_pose(Tc), *_pose(Tm), *e_m, *e_c]
        row += [*np.rad2deg(q_cmd_rad), *np.asarray(q_meas_deg, dtype=float), solve_ms, qp_iters]
        self._w.writerow([f"{v:.6g}" if isinstance(v, float) else v for v in map(_to_py, row)])
        self._n += 1
        if self._n % 60 == 0:
            self._f.flush()

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()


def _to_py(v):
    return v.item() if isinstance(v, np.generic) else v
