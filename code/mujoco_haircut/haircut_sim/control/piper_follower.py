import logging
import time
from functools import cached_property

import numpy as np
from lerobot.cameras import make_cameras_from_configs
from lerobot.robots.robot import Robot

from .config_piper_follower import PiperFollowerConfig
from .piper_backend import MockPiperBackend, PiperSDKBackend
from .piper_kinematics import piper_fk
from .mrnn_ik import MRNNIK
from .qp_ik import QPIK
from ..pnd_ik import PNDIK
from .tracking_log import TrackingLogger
from .so101_kinematics import JOINT_NAMES as SO101_JOINTS
from .transforms import matrix_to_rotvec, matrix_to_rpy, rot_z, rotvec_to_matrix, rpy_to_matrix

logger = logging.getLogger(__name__)

EE_KEYS = ["ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz"]
PIPER_JOINTS = [f"joint_{i}" for i in range(1, 7)]
# The 5 SO-101 joints map to Piper J1 J2 J3 J5 J6 (0-based indices); SO-101 has no equivalent of J4 (forearm roll)
SO101_TO_PIPER_IDX = [0, 1, 2, 4, 5]
# Piper joint limits (deg), slightly more conservative than the official manual; piper_sdk clips again itself
PIPER_JOINT_LIMITS_DEG = np.array(
    [[-150.0, 150.0], [0.0, 180.0], [-170.0, 0.0], [-100.0, 100.0], [-70.0, 70.0], [-100.0, 100.0]]
)


class PiperFollower(Robot):
    """AgileX Piper follower arm.

    ee-mode action: ee.x/y/z (m, leader base frame), ee.wx/wy/wz (rotation vector), gripper.pos (0-100)
    joint-mode action: shoulder_pan.pos … wrist_roll.pos (deg), gripper.pos (0-100), i.e. the output of the stock so101_leader
    """

    config_class = PiperFollowerConfig
    name = "piper_follower"

    def __init__(self, config: PiperFollowerConfig):
        super().__init__(config)
        self.config = config
        self.backend = MockPiperBackend() if config.mock else PiperSDKBackend(config.can_port)
        self.cameras = make_cameras_from_configs(config.cameras)
        self._connected = False
        self._R_align = rot_z(np.deg2rad(config.base_yaw_offset_deg))
        # IK solver for qp mode: both share the interface step(q, p_d, R_d, dt) -> q_next
        self.qp_ik = None
        if config.control_mode == "qp":
            if config.ik_solver == "pnd":
                # Both solver and output enforce the configured operational limits.
                config.pnd.joint_limits_deg = config.joint_limits_deg
                self.qp_ik = PNDIK(config.pnd)
            else:
                self.qp_ik = MRNNIK(config.mrnn) if config.ik_solver == "mrnn" else QPIK(config.qp)
        self.tracking_log: TrackingLogger | None = None
        self.reset_anchor()

    def reset_anchor(self) -> None:
        """Relative mapping: the next action becomes the new corresponding origin of both arms (acts as a clutch)."""
        self._anchor = None
        self._last_cmd = None
        self._last_t = None

    # ---------------- features ----------------
    @property
    def _state_ft(self) -> dict[str, type]:
        ft = {f"{j}.pos": float for j in PIPER_JOINTS}
        ft.update({k: float for k in EE_KEYS})
        ft["gripper.pos"] = float
        return ft

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {cam: (self.cameras[cam].height, self.cameras[cam].width, 3) for cam in self.cameras}

    @cached_property
    def observation_features(self) -> dict:
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        keys = EE_KEYS if self.config.control_mode in ("ee", "qp") else [f"{j}.pos" for j in SO101_JOINTS]
        return {**{k: float for k in keys}, "gripper.pos": float}

    # ---------------- lifecycle ----------------
    @property
    def is_connected(self) -> bool:
        return self._connected and all(cam.is_connected for cam in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        if self._connected:
            raise RuntimeError(f"{self} already connected")
        self.backend.connect()
        for cam in self.cameras.values():
            cam.connect()
        self._connected = True
        if self.config.start_joints_deg is not None:
            self.move_to_joints(self.config.start_joints_deg, self.config.start_speed_rate)
        self.reset_anchor()

    def move_to_joints(self, target_deg, speed_rate: int, tol_deg: float = 1.5, timeout_s: float = 20.0) -> None:
        """Blocking move to the target joint angles (deg) at Piper's internal speed limit; returns when reached or on timeout."""
        target = np.clip(np.asarray(target_deg, dtype=float), PIPER_JOINT_LIMITS_DEG[:, 0], PIPER_JOINT_LIMITS_DEG[:, 1])
        logger.info(f"Moving to the ready pose {np.round(target, 1)}° at speed {speed_rate}%")
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout_s:
            self.backend.send_joints(target, speed_rate)
            if np.max(np.abs(self.backend.get_joints_deg() - target)) < tol_deg:
                return
            time.sleep(0.02)
        logger.warning(f"Ready pose not reached within {timeout_s}s; current {np.round(self.backend.get_joints_deg(), 1)}°")
        logger.info(f"{self} connected ({'mock' if self.config.mock else self.config.can_port}).")

    def disconnect(self) -> None:
        if not self._connected:
            return
        if self.config.return_on_disconnect and self.config.start_joints_deg is not None:
            try:
                self.backend.recover()
                self.move_to_joints(self.config.start_joints_deg, self.config.start_speed_rate)
            except Exception as e:
                logger.warning(f"Piper homing failed: {e}")
        if self.tracking_log is not None:
            self.tracking_log.close()
            logger.info(f"Tracking log saved: {self.tracking_log.path}")
            self.tracking_log = None
        self.backend.disconnect(self.config.disable_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()
        self._connected = False
        logger.info(f"{self} disconnected.")

    # ---------------- observation ----------------
    def get_observation(self) -> dict:
        if not self._connected:
            raise RuntimeError(f"{self} is not connected")
        obs = {f"{j}.pos": float(v) for j, v in zip(PIPER_JOINTS, self.backend.get_joints_deg(), strict=True)}
        pos, rpy = self.backend.get_ee()
        rv = matrix_to_rotvec(rpy_to_matrix(*rpy))
        obs.update(dict(zip(EE_KEYS, [*pos.tolist(), *rv.tolist()], strict=True)))
        obs["gripper.pos"] = self._mm_to_gripper_pos(self.backend.get_gripper_mm())
        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.read_latest()
        return obs

    # ---------------- action ----------------
    def send_action(self, action: dict) -> dict:
        if not self._connected:
            raise RuntimeError(f"{self} is not connected")
        self._n_actions = getattr(self, "_n_actions", 0) + 1
        if self._n_actions % 10 == 0:
            bad = self.backend.faulted_joints()
            if bad:
                raise RuntimeError(f"Piper joint(s) {bad} disabled by stall/overcurrent protection; teleoperation stopped. Current joint angles {np.round(self.backend.get_joints_deg(), 1)}°")
        now = time.perf_counter()
        dt = 0.0 if self._last_t is None else float(np.clip(now - self._last_t, 1e-3, 0.1))
        self._last_t = now

        if self.config.control_mode == "ee":
            sent = self._send_ee(action, dt)
        elif self.config.control_mode == "qp":
            sent = self._send_qp(action, dt)
        else:
            sent = self._send_joint(action, dt)

        if "gripper.pos" in action:
            mm = self._gripper_pos_to_mm(float(action["gripper.pos"]))
            self.backend.send_gripper(mm, self.config.gripper_effort)
            sent["gripper.pos"] = self._mm_to_gripper_pos(mm)
        return sent

    def _send_ee(self, action: dict, dt: float) -> dict:
        cfg = self.config
        p_l = np.array([action["ee.x"], action["ee.y"], action["ee.z"]], dtype=float)
        R_l = rotvec_to_matrix([action["ee.wx"], action["ee.wy"], action["ee.wz"]])
        A = self._R_align

        if self._anchor is None:
            p_f0, rpy_f0 = self.backend.get_ee()
            self._anchor = (p_l, R_l, p_f0, rpy_to_matrix(*rpy_f0))
            self._last_cmd = (p_f0, rpy_to_matrix(*rpy_f0))
            logger.info(f"Anchor: leader {np.round(p_l, 3)} m ↔ Piper {np.round(p_f0, 3)} m")
        p_l0, R_l0, p_f0, R_f0 = self._anchor

        if cfg.mapping == "relative":
            p = p_f0 + cfg.position_scale * (A @ (p_l - p_l0))
        else:
            p = cfg.position_scale * (A @ p_l) + np.asarray(cfg.ee_offset_m, dtype=float)
        # Orientation always follows increments: Piper rotates in its base frame by as much as the leader rotates in its own base frame
        R = A @ (R_l @ R_l0.T) @ A.T @ R_f0

        p = self._clip_workspace(p)
        p, R = self._rate_limit_ee(p, R, dt)
        self._last_cmd = (p, R)

        self.backend.send_ee(p, matrix_to_rpy(R), cfg.speed_rate)
        rv = matrix_to_rotvec(R)
        return dict(zip(EE_KEYS, [*p.tolist(), *rv.tolist()], strict=True))

    def _send_qp(self, action: dict, dt: float) -> dict:
        """Leader end-effector target rd → local IK (MRNN: mrnn_ik.py or QP: qp_ik.py) → joint command. Piper's built-in end-effector IK is not used."""
        cfg = self.config
        p_l = np.array([action["ee.x"], action["ee.y"], action["ee.z"]], dtype=float)
        R_l = rotvec_to_matrix([action["ee.wx"], action["ee.wy"], action["ee.wz"]])
        A = self._R_align

        tool = np.asarray(cfg.tool_offset_m, dtype=float)  # tip position in the flange frame
        if self._anchor is None:
            q0 = np.deg2rad(self.backend.get_joints_deg())
            T0 = piper_fk(q0)
            # The anchor is the tip position: the tip moves however the leader end effector moves, and when the leader rotates about its end, the tip rotates about itself
            self._anchor = (p_l, R_l, T0[:3, 3] + T0[:3, :3] @ tool, T0[:3, :3].copy())
            self._last_cmd = q0  # QP linearizes about the previous cycle's command (rad)
            logger.info(f"QP anchor: leader {np.round(p_l, 3)} m ↔ Piper {np.round(T0[:3, 3], 3)} m")
        p_l0, R_l0, p_f0, R_f0 = self._anchor

        if cfg.mapping == "relative":
            p_d = p_f0 + cfg.position_scale * (A @ (p_l - p_l0))
        else:
            p_d = cfg.position_scale * (A @ p_l) + np.asarray(cfg.ee_offset_m, dtype=float)
        R_d = A @ (R_l @ R_l0.T) @ A.T @ R_f0
        # Tip target → flange target; the IK still solves for the flange. The offset uses the current feedback orientation rather than R_d: orientation is only a soft constraint (w_rot),
        # so the tip position stays on target even when orientation lags (mock: leader swinging ±20° about its end gives 1.9 mm mean tip drift; 7 mm with R_d)
        p_d = self._clip_workspace(p_d - piper_fk(np.deg2rad(self.backend.get_joints_deg()))[:3, :3] @ tool)

        t_solve = time.perf_counter()
        if dt <= 0.0:
            q = self._last_cmd
        else:
            q_base = np.deg2rad(self.backend.get_joints_deg()) if cfg.ik_solver == "pnd" else self._last_cmd
            q = self.qp_ik.step(q_base, p_d, R_d, dt)
        solve_ms = (time.perf_counter() - t_solve) * 1e3
        # Last safeguard: whatever the QP does, the joint angles sent never exceed joint_limits_deg
        lim = np.deg2rad(np.asarray(cfg.joint_limits_deg, dtype=float))
        q = np.clip(q, lim[:, 0], lim[:, 1])
        self._last_cmd = q
        self.backend.send_joints(np.rad2deg(q), cfg.speed_rate)
        if cfg.log_dir:
            if self.tracking_log is None:
                self.tracking_log = TrackingLogger(cfg.log_dir, prefix=cfg.ik_solver)
                logger.info(f"Writing tracking log to {self.tracking_log.path}")
            self.tracking_log.log(p_d, R_d, q, self.backend.get_joints_deg(), solve_ms, self.qp_ik.last_info.get("qp_iters", 0))
        return dict(zip(EE_KEYS, [*p_d.tolist(), *matrix_to_rotvec(R_d).tolist()], strict=True))

    def _clip_workspace(self, p: np.ndarray) -> np.ndarray:
        cfg = self.config
        p = np.clip(p, cfg.workspace_min_m, cfg.workspace_max_m)
        # Limit the reach with a sphere centered on the shoulder joint (0.123 m above the base)
        shoulder = np.array([0.0, 0.0, 0.123])
        d = p - shoulder
        r = np.linalg.norm(d)
        if r > cfg.max_reach_m:
            p = shoulder + d * (cfg.max_reach_m / r)
        elif 1e-6 < r < cfg.min_reach_m:
            p = shoulder + d * (cfg.min_reach_m / r)
        return p

    def _rate_limit_ee(self, p: np.ndarray, R: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        p_prev, R_prev = self._last_cmd
        if dt <= 0.0:
            return p_prev, R_prev
        dp = p - p_prev
        max_dp = self.config.max_linear_speed_m_s * dt
        n = np.linalg.norm(dp)
        if n > max_dp:
            p = p_prev + dp * (max_dp / n)
        rv = matrix_to_rotvec(R @ R_prev.T)
        ang = np.linalg.norm(rv)
        max_ang = np.deg2rad(self.config.max_angular_speed_deg_s) * dt
        if ang > max_ang:
            R = rotvec_to_matrix(rv * (max_ang / ang)) @ R_prev
        return p, R

    def _send_joint(self, action: dict, dt: float) -> dict:
        cfg = self.config
        q_l = np.array([action[f"{j}.pos"] for j in SO101_JOINTS], dtype=float)
        signs = np.asarray(cfg.joint_signs, dtype=float)

        if self._anchor is None:
            q_f0 = self.backend.get_joints_deg()
            self._anchor = (q_l, q_f0)
            self._last_cmd = q_f0.copy()
            logger.info(f"Anchor: leader {np.round(q_l, 1)}° ↔ Piper {np.round(q_f0, 1)}°")
        q_l0, q_f0 = self._anchor

        q = q_f0.copy()
        if cfg.mapping == "relative":
            q[SO101_TO_PIPER_IDX] = q_f0[SO101_TO_PIPER_IDX] + signs * (q_l - q_l0)
        else:
            q[SO101_TO_PIPER_IDX] = signs * q_l + np.asarray(cfg.joint_offsets_deg, dtype=float)
            q[3] = cfg.joint4_deg
        lim = np.asarray(cfg.joint_limits_deg, dtype=float)
        q = np.clip(q, lim[:, 0], lim[:, 1])

        if dt > 0.0:
            step = cfg.max_joint_speed_deg_s * dt
            q = self._last_cmd + np.clip(q - self._last_cmd, -step, step)
        else:
            q = self._last_cmd.copy()
        self._last_cmd = q

        self.backend.send_joints(q, cfg.speed_rate)
        return {f"{j}.pos": float(q[i]) for i, j in zip(SO101_TO_PIPER_IDX, SO101_JOINTS, strict=True)}

    # ---------------- gripper ----------------
    def _gripper_pos_to_mm(self, pos: float) -> float:
        frac = float(np.clip(pos, 0.0, 100.0)) / 100.0
        if self.config.invert_gripper:
            frac = 1.0 - frac
        return frac * self.config.gripper_max_mm

    def _mm_to_gripper_pos(self, mm: float) -> float:
        frac = float(np.clip(mm / self.config.gripper_max_mm, 0.0, 1.0))
        if self.config.invert_gripper:
            frac = 1.0 - frac
        return 100.0 * frac
