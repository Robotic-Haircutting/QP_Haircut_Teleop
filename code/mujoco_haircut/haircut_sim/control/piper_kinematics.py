"""Piper forward kinematics and geometric Jacobian (modified DH, parameters match piper_sdk C_PiperForwardKinematics(dh_is_offset=1)).

The output is the flange (joint 6 end) pose, the same end pose the Piper controller reports via GetArmEndPoseMsgs. Units: m, rad.
"""

import numpy as np

_A = np.array([0.0, 0.0, 285.03, -21.98, 0.0, 0.0]) / 1000.0
_ALPHA = np.array([0.0, -np.pi / 2, 0.0, np.pi / 2, -np.pi / 2, np.pi / 2])
_THETA0 = np.array([0.0, -np.deg2rad(172.22), -np.deg2rad(102.78), 0.0, 0.0, 0.0])
_D = np.array([123.0, 0.0, 0.0, 250.75, 0.0, 91.0]) / 1000.0


def _link_tf(alpha: float, a: float, theta: float, d: float) -> np.ndarray:
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
    return np.array(
        [
            [ct, -st, 0.0, a],
            [st * ca, ct * ca, -sa, -sa * d],
            [st * sa, ct * sa, ca, ca * d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def piper_frames(q_rad) -> list[np.ndarray]:
    """Return the poses T_0i (i=1..6) of the six joint frames in the base frame. Joint i rotates about the z axis of T_0i."""
    q = np.asarray(q_rad, dtype=float)
    T = np.eye(4)
    frames = []
    for i in range(6):
        T = T @ _link_tf(_ALPHA[i], _A[i], q[i] + _THETA0[i], _D[i])
        frames.append(T)
    return frames


def piper_fk(q_rad) -> np.ndarray:
    """Flange 4x4 pose (m)."""
    return piper_frames(q_rad)[-1]


def piper_fk_jacobian(q_rad) -> tuple[np.ndarray, np.ndarray]:
    """Return (T_06, J). J is the 6x6 geometric Jacobian, rows = [v; ω] (base frame), columns = joint velocities (rad/s)."""
    frames = piper_frames(q_rad)
    p_e = frames[-1][:3, 3]
    J = np.zeros((6, 6))
    for i, T in enumerate(frames):
        z, p = T[:3, 2], T[:3, 3]
        J[:3, i] = np.cross(z, p_e - p)
        J[3:, i] = z
    return frames[-1], J
