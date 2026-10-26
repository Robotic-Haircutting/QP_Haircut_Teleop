import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import time
"""Hardware-free self-test: python -m pytest tests -q (or run python tests/test_mock.py directly)"""

import numpy as np

from haircut_sim.control import PiperFollower, PiperFollowerConfig, SO101LeaderEE, SO101LeaderEEConfig
from haircut_sim.control.so101_kinematics import so101_fk
from haircut_sim.control.transforms import matrix_to_rotvec, matrix_to_rpy, rotvec_to_matrix, rpy_to_matrix


def _random_R(rng):
    return rotvec_to_matrix(rng.normal(size=3))


def test_rotation_roundtrips():
    rng = np.random.default_rng(0)
    for _ in range(500):
        R = _random_R(rng)
        assert np.allclose(rotvec_to_matrix(matrix_to_rotvec(R)), R, atol=1e-9)
        assert np.allclose(rpy_to_matrix(*matrix_to_rpy(R)), R, atol=1e-9)


def test_so101_fk_home_reaches_forward():
    p = so101_fk([0, 0, 0, 0, 0])[:3, 3]
    assert 0.3 < p[0] < 0.45 and abs(p[1]) < 1e-3  # SO-101 base frame: x points forward


def _ee_action(q, gripper=50.0):
    T = so101_fk(q)
    rv = matrix_to_rotvec(T[:3, :3])
    return {"ee.x": T[0, 3], "ee.y": T[1, 3], "ee.z": T[2, 3],
            "ee.wx": rv[0], "ee.wy": rv[1], "ee.wz": rv[2], "gripper.pos": gripper}


def test_ee_relative_mapping_follows_delta():
    cfg = PiperFollowerConfig(id="t", mock=True, max_linear_speed_m_s=100, max_angular_speed_deg_s=1e5)
    robot = PiperFollower(cfg)
    robot.connect()
    p0, _ = robot.backend.get_ee()
    q0 = [0, -40, 40, 0, 0]
    robot.send_action(_ee_action(q0))  # first frame sets the anchor; no motion
    assert np.allclose(robot.backend.get_ee()[0], p0)
    q1 = [0, -30, 40, 0, 0]
    robot.send_action(_ee_action(q1, gripper=100))
    dp_leader = so101_fk(q1)[:3, 3] - so101_fk(q0)[:3, 3]
    expected = np.clip(p0 + cfg.position_scale * dp_leader, cfg.workspace_min_m, cfg.workspace_max_m)
    assert np.allclose(robot.backend.get_ee()[0], expected, atol=1e-9)
    assert abs(robot.backend.get_gripper_mm() - cfg.gripper_max_mm) < 1e-9
    robot.disconnect()


def test_ee_speed_limit_and_workspace():
    cfg = PiperFollowerConfig(id="t", mock=True, position_scale=10.0)
    robot = PiperFollower(cfg)
    robot.connect()
    robot.send_action(_ee_action([0, 0, 0, 0, 0]))
    p_prev = robot.backend.get_ee()[0]
    for _ in range(200):
        robot.send_action(_ee_action([60, -60, 60, 0, 0]))
        p = robot.backend.get_ee()[0]
        assert np.linalg.norm(p - p_prev) <= cfg.max_linear_speed_m_s * 0.1 + 1e-9
        assert np.all(p >= np.array(cfg.workspace_min_m) - 1e-9) and np.all(p <= np.array(cfg.workspace_max_m) + 1e-9)
        p_prev = p
    robot.disconnect()


def test_joint_mode_relative():
    cfg = PiperFollowerConfig(id="t", mock=True, control_mode="joint", joint_signs=[1, -1, 1, 1, -1],
                              max_joint_speed_deg_s=1e5)
    robot = PiperFollower(cfg)
    robot.connect()
    robot.backend.joints = np.array([0.0, 30.0, -40.0, 5.0, 0.0, 0.0])
    names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    a0 = {f"{n}.pos": 0.0 for n in names} | {"gripper.pos": 0.0}
    robot.send_action(a0)
    a1 = {f"{n}.pos": v for n, v in zip(names, [10, 10, 10, 10, 10])} | {"gripper.pos": 0.0}
    robot.send_action(a1)
    assert np.allclose(robot.backend.get_joints_deg(), [10, 20, -30, 5, 10, -10])
    obs = robot.get_observation()
    assert set(obs) == set(robot.observation_features)
    robot.disconnect()


def test_leader_mock_outputs_features():
    leader = SO101LeaderEE(SO101LeaderEEConfig(port="none", id="t", mock=True, include_joints=True))
    leader.connect()
    a = leader.get_action()
    assert set(a) == set(leader.action_features)
    leader.disconnect()


def test_qp_mode_tracks_leader_ee():
    """Mock leader (sinusoidal motion) -> mock Piper in qp mode, 60 Hz for 6 s; report end-effector tracking error."""
    import numpy as np

    from haircut_sim.control import PiperFollower, PiperFollowerConfig, SO101LeaderEE, SO101LeaderEEConfig
    from haircut_sim.control.piper_kinematics import piper_fk

    leader = SO101LeaderEE(SO101LeaderEEConfig(port="none", mock=True))
    robot = PiperFollower(
        PiperFollowerConfig(mock=True, control_mode="qp", start_joints_deg=[0, 50, -50, 0, 30, 0], position_scale=1.0)
    )
    leader.connect()
    robot.connect()
    errs = []
    t_end = time.perf_counter() + 6.0
    while time.perf_counter() < t_end:
        sent = robot.send_action(leader.get_action())
        T = piper_fk(np.deg2rad(robot.backend.get_joints_deg()))
        errs.append(np.linalg.norm(T[:3, 3] - np.array([sent["ee.x"], sent["ee.y"], sent["ee.z"]])))
        time.sleep(1 / 60)
    robot.disconnect()
    leader.disconnect()
    errs = np.array(errs[30:])
    print(f"    qp tracking error: mean {errs.mean() * 1000:.1f} mm, max {errs.max() * 1000:.1f} mm")
    assert errs.mean() < 0.01


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
