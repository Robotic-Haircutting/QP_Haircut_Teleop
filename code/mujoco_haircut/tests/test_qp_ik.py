import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
"""Offline QP-IK self-test (no hardware, no lerobot needed): python tests/test_qp_ik.py"""

import time

import numpy as np

from haircut_sim.control.piper_kinematics import piper_fk, piper_fk_jacobian
from haircut_sim.control.qp_ik import QPIK, QPIKParams
from haircut_sim.control.qp_solver import solve_box_qp
from haircut_sim.control.transforms import matrix_to_rotvec, rotvec_to_matrix

rng = np.random.default_rng(0)
LIM = np.deg2rad(np.array(QPIKParams().joint_limits_deg, dtype=float))


def rand_q():
    return rng.uniform(LIM[:, 0], LIM[:, 1])


def test_fk_matches_piper_sdk():
    from piper_sdk.kinematics.piper_fk import C_PiperForwardKinematics

    sdk = C_PiperForwardKinematics(dh_is_offset=0x01)
    for _ in range(20):
        q = rand_q()
        ref = np.array(sdk.CalFK(q.tolist())[-1][:3]) / 1000.0
        assert np.allclose(piper_fk(q)[:3, 3], ref, atol=1e-9)


def test_jacobian_matches_finite_difference():
    for _ in range(10):
        q = rand_q()
        T, J = piper_fk_jacobian(q)
        h = 1e-6
        for i in range(6):
            dq = np.zeros(6)
            dq[i] = h
            T2 = piper_fk(q + dq)
            v = (T2[:3, 3] - T[:3, 3]) / h
            w = matrix_to_rotvec(T2[:3, :3] @ T[:3, :3].T) / h
            assert np.allclose(J[:3, i], v, atol=1e-5) and np.allclose(J[3:, i], w, atol=1e-5)


def _projected_gradient(H, g, lb, ub, n_iter=20000):
    L = np.linalg.eigvalsh(H).max()
    x = np.clip(np.zeros_like(g), lb, ub)
    for _ in range(n_iter):
        x = np.clip(x - (H @ x + g) / L, lb, ub)
    return x


def test_box_qp_matches_reference():
    for _ in range(200):
        A = rng.normal(size=(6, 6))
        H = A.T @ A + 1e-2 * np.eye(6)
        g = rng.normal(size=6) * 3
        lb = -rng.uniform(0.05, 1.0, 6)
        ub = rng.uniform(0.05, 1.0, 6)
        x, _ = solve_box_qp(H, g, lb, ub)
        ref = _projected_gradient(H, g, lb, ub)
        f = lambda z: 0.5 * z @ H @ z + g @ z  # noqa: E731
        assert f(x) <= f(ref) + 1e-7, (f(x), f(ref))
        assert np.all(x >= lb - 1e-12) and np.all(x <= ub + 1e-12)


def test_tracking_reachable_targets_at_60hz():
    ik = QPIK(QPIKParams(max_joint_speed_deg_s=90.0))
    dt = 1 / 60
    q0 = np.deg2rad([0, 40, -40, 0, 0, 0])
    worst_pos, worst_rot, times = 0.0, 0.0, []
    for _ in range(20):
        # Target: a random reachable pose near the ready pose
        q_goal = np.clip(q0 + np.deg2rad(rng.uniform(-30, 30, 6)), LIM[:, 0], LIM[:, 1])
        Td = piper_fk(q_goal)
        q = q0.copy()
        for _ in range(int(3.0 / dt)):
            t = time.perf_counter()
            q = ik.step(q, Td[:3, 3], Td[:3, :3], dt)
            times.append(time.perf_counter() - t)
            assert np.all(q >= LIM[:, 0] - 1e-9) and np.all(q <= LIM[:, 1] + 1e-9)
        T = piper_fk(q)
        worst_pos = max(worst_pos, np.linalg.norm(T[:3, 3] - Td[:3, 3]))
        worst_rot = max(worst_rot, np.linalg.norm(matrix_to_rotvec(Td[:3, :3] @ T[:3, :3].T)))
    print(
        f"    max residual after 3 s: position {worst_pos * 1000:.2f} mm, orientation {np.rad2deg(worst_rot):.3f}°; "
        f"time per step: mean {np.mean(times) * 1e3:.3f} ms, worst {np.max(times) * 1e3:.3f} ms"
    )
    assert worst_pos < 1e-3 and worst_rot < np.deg2rad(0.5)
    assert np.mean(times) < 2e-3


def test_speed_limit_and_unreachable_target():
    ik = QPIK(QPIKParams(max_joint_speed_deg_s=30.0))
    dt = 1 / 60
    q = np.deg2rad([0, 40, -40, 0, 0, 0])
    Td = piper_fk(q)
    p_far = Td[:3, 3] + np.array([2.0, 0, 0])  # 2 m away, out of reach
    for _ in range(300):
        q_new = ik.step(q, p_far, Td[:3, :3], dt)
        assert np.max(np.abs(q_new - q)) <= np.deg2rad(30.0) * dt + 1e-9
        assert np.all(np.isfinite(q_new))
        q = q_new
    assert np.all(q >= LIM[:, 0] - 1e-9) and np.all(q <= LIM[:, 1] + 1e-9)


def test_orientation_only_target():
    ik = QPIK()
    q = np.deg2rad([0, 60, -60, 0, 20, 0])
    T0 = piper_fk(q)
    Rd = rotvec_to_matrix([0, 0, np.deg2rad(20)]) @ T0[:3, :3]
    for _ in range(240):
        q = ik.step(q, T0[:3, 3], Rd, 1 / 60)
    T = piper_fk(q)
    assert np.linalg.norm(T[:3, 3] - T0[:3, 3]) < 2e-3
    assert np.linalg.norm(matrix_to_rotvec(Rd @ T[:3, :3].T)) < np.deg2rad(1.0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
