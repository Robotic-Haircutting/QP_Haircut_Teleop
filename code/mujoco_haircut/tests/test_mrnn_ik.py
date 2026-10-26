import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
"""Offline MRNN IK self-test (no hardware, no lerobot needed): python tests/test_mrnn_ik.py"""

import time

import numpy as np

from haircut_sim.control.mrnn_ik import MRNNIK, MRNNIKParams, _grad_lp
from haircut_sim.control.piper_kinematics import piper_fk
from haircut_sim.control.transforms import matrix_to_rotvec, rotvec_to_matrix

rng = np.random.default_rng(1)
LIM = np.deg2rad(np.array(MRNNIKParams().joint_limits_deg, dtype=float))


def test_grad_lp_matches_finite_difference():
    p, delta = 0.5, 1e-3
    for _ in range(50):
        v = rng.normal(size=6)
        v[np.abs(v) < 0.05] = 0.1  # stay away from the smoothing region
        f = lambda x: np.sum(np.abs(x) ** p) ** (1 / p)  # noqa: E731
        num = np.array([(f(v + h) - f(v - h)) / 2e-6 for h in np.eye(6) * 1e-6])
        assert np.allclose(_grad_lp(v, p, delta), num, rtol=1e-4, atol=1e-6)


def test_fixed_point_satisfies_stationary_condition():
    """The converged point satisfies Eq. (7): ∇φ + ρ(ẋ − P_C(ẋ)) + Jᵀλ = 0, Jẋ = ẏ_d."""
    ik = MRNNIK(MRNNIKParams(task="position", iterations=5000, tol=1e-13))
    q = np.deg2rad([10, 60, -60, 20, 30, -10])
    J, yd, lo, hi, _ = ik.build(q, piper_fk(q)[:3, 3] + [0.01, 0, 0.005], piper_fk(q)[:3, :3], 1 / 60)
    x, _ = ik.solve(J, yd, lo, hi)
    assert np.allclose(J @ x, yd, atol=1e-6)


def test_momentum_accelerates_convergence():
    q = np.deg2rad([10, 60, -60, 20, 30, -10])
    T = piper_fk(q)
    Rd = rotvec_to_matrix([0.05, 0, 0.04]) @ T[:3, :3]
    errs = {}
    for om in (0.0, 0.8):
        ik = MRNNIK(MRNNIKParams(momentum=om, iterations=3000, tol=1e-14))
        J, yd, lo, hi, _ = ik.build(q, T[:3, 3] + [0.01, -0.005, 0.008], Rd, 1 / 60)
        ref, _ = ik.solve(J, yd, lo, hi)
        ik.p.iterations = 60
        x, _ = ik.solve(J, yd, lo, hi)
        errs[om] = np.linalg.norm(x - ref)
    print(f"    distance to the converged point after 60 iterations: no momentum {errs[0.0]:.1e}, ω=0.8 {errs[0.8]:.1e}")
    assert errs[0.8] < 0.2 * errs[0.0]


def test_tracking_reachable_targets_at_60hz():
    ik = MRNNIK()
    dt = 1 / 60
    q0 = np.deg2rad([0, 50, -50, 0, 30, 0])
    worst_p, worst_r, times = 0.0, 0.0, []
    for _ in range(10):
        q_goal = np.clip(q0 + np.deg2rad(rng.uniform(-25, 25, 6)), LIM[:, 0], LIM[:, 1])
        Td = piper_fk(q_goal)
        q = q0.copy()
        for _ in range(int(4.0 / dt)):
            t = time.perf_counter()
            q = ik.step(q, Td[:3, 3], Td[:3, :3], dt)
            times.append(time.perf_counter() - t)
            assert np.all(q >= LIM[:, 0] - 1e-9) and np.all(q <= LIM[:, 1] + 1e-9)
        T = piper_fk(q)
        worst_p = max(worst_p, np.linalg.norm(T[:3, 3] - Td[:3, 3]))
        worst_r = max(worst_r, np.linalg.norm(matrix_to_rotvec(Td[:3, :3] @ T[:3, :3].T)))
    print(
        f"    max residual after 4 s: position {worst_p * 1000:.2f} mm, orientation {np.rad2deg(worst_r):.2f}°; "
        f"time per step: mean {np.mean(times) * 1e3:.2f} ms, worst {np.max(times) * 1e3:.2f} ms"
    )
    assert worst_p < 2e-3
    assert np.mean(times) < 5e-3


def test_speed_limit_respected():
    ik = MRNNIK(MRNNIKParams(max_joint_speed_deg_s=30.0))
    dt = 1 / 60
    q = np.deg2rad([0, 50, -50, 0, 30, 0])
    T = piper_fk(q)
    for _ in range(200):
        qn = ik.step(q, T[:3, 3] + [1.5, 0, 0], T[:3, :3], dt)
        assert np.max(np.abs(qn - q)) <= np.deg2rad(30.0) * dt + 1e-9 and np.all(np.isfinite(qn))
        q = qn


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
