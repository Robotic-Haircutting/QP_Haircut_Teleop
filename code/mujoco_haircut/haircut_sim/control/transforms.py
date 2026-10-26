"""Pure-numpy pose utilities: homogeneous transforms, rotation vectors, Piper Euler angles (R = Rz·Ry·Rx)."""

import numpy as np


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF / Piper convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll), in rad."""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def matrix_to_rpy(R: np.ndarray) -> np.ndarray:
    """Inverse of rpy_to_matrix, consistent with piper_sdk C_PiperForwardKinematics; returns rad."""
    if R[2, 0] < -1 + 1e-6:
        return np.array([np.arctan2(R[0, 1], R[1, 1]), np.pi / 2, 0.0])
    if R[2, 0] > 1 - 1e-6:
        return np.array([-np.arctan2(R[0, 1], R[1, 1]), -np.pi / 2, 0.0])
    pitch = np.arctan2(-R[2, 0], np.hypot(R[0, 0], R[1, 0]))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    roll = np.arctan2(R[2, 1], R[2, 2])
    return np.array([roll, pitch, yaw])


def make_tf(R: np.ndarray, p) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def rotvec_to_matrix(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    theta = np.linalg.norm(v)
    if theta < 1e-12:
        return np.eye(3)
    k = v / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    cos_t = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    theta = np.arccos(cos_t)
    if theta < 1e-9:
        return np.zeros(3)
    if np.pi - theta < 1e-6:
        # Near 180°: take the axis from (R + I)/2 = k k^T
        M = (R + np.eye(3)) / 2
        k = np.sqrt(np.clip(np.diag(M), 0, None))
        i = int(np.argmax(k))
        k[i] = np.sqrt(M[i, i])
        for j in range(3):
            if j != i:
                k[j] = M[i, j] / k[i]
        return theta * k / np.linalg.norm(k)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return theta * w / (2 * np.sin(theta))
