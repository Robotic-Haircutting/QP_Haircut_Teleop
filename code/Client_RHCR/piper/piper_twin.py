"""Piper MuJoCo digital twin (used by piper_receiver.py).

Same approach as the SO-101-teleoperates-Piper project (mujoco_haircut):
  - The MuJoCo window is a separate process ``python -m haircut_sim.viewer`` (lerobot environment, with mujoco installed),
    scene = Piper + head model + clipper-tip trajectory/hair coverage, configured in mujoco_haircut/scene_config.json;
  - this process only sends ``{"s":"piper","q":[J1..J6 deg],"g":gripper mm}`` over UDP 127.0.0.1:50555,
    fire-and-forget; control is unaffected whether the window is open or closed.

dry-run: end-effector target → damped least-squares IK in this file → joint angles → window.
Real robot: shows the joint angles fed back by GetArmJointMsgs directly (same as the SO-101 project).

This file depends only on numpy, so it runs in the Client_RHCR venv without installing mujoco there.
Kinematic parameters are copied from mujoco_haircut/haircut_sim/control/piper_kinematics.py (that project is not modified).
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Piper kinematics (modified DH, consistent with piper_sdk dh_is_offset=1; flange pose = GetArmEndPoseMsgs)
# ---------------------------------------------------------------------------

_A = np.array([0.0, 0.0, 285.03, -21.98, 0.0, 0.0]) / 1000.0
_ALPHA = np.array([0.0, -np.pi / 2, 0.0, np.pi / 2, -np.pi / 2, np.pi / 2])
_THETA0 = np.array([0.0, -np.deg2rad(172.22), -np.deg2rad(102.78), 0.0, 0.0, 0.0])
_D = np.array([123.0, 0.0, 0.0, 250.75, 0.0, 91.0]) / 1000.0

# Piper joint limits (deg), with a 2° margin
JOINT_MIN_DEG = np.array([-148.0, 2.0, -168.0, -98.0, -68.0, -118.0])
JOINT_MAX_DEG = np.array([148.0, 178.0, -2.0, 98.0, 68.0, 118.0])


def _link_tf(alpha: float, a: float, theta: float, d: float) -> np.ndarray:
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st, 0.0, a],
        [st * ca, ct * ca, -sa, -sa * d],
        [st * sa, ct * sa, ca, ca * d],
        [0.0, 0.0, 0.0, 1.0],
    ])


def fk_jacobian(q_rad) -> tuple[np.ndarray, np.ndarray]:
    """Return (T_06 in m, J the 6x6 geometric Jacobian, rows = [v; ω] in the base frame)."""
    q = np.asarray(q_rad, dtype=float)
    T = np.eye(4)
    frames = []
    for i in range(6):
        T = T @ _link_tf(_ALPHA[i], _A[i], q[i] + _THETA0[i], _D[i])
        frames.append(T)
    p_e = frames[-1][:3, 3]
    J = np.zeros((6, 6))
    for i, F in enumerate(frames):
        z, p = F[:3, 2], F[:3, 3]
        J[:3, i] = np.cross(z, p_e - p)
        J[3:, i] = z
    return frames[-1], J


def rot_log(r: np.ndarray) -> np.ndarray:
    """Rotation matrix → rotation vector (rad)."""
    c = np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)
    ang = np.arccos(c)
    w = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    if ang < 1e-6:
        return 0.5 * w
    if np.pi - ang < 1e-4:
        # Near 180°: take the dominant column of (R+I)/2
        b = (r + np.eye(3)) / 2.0
        k = int(np.argmax(np.diag(b)))
        axis = b[:, k] / np.sqrt(max(b[k, k], 1e-12))
        return axis / np.linalg.norm(axis) * ang
    return w * (ang / (2.0 * np.sin(ang)))


def solve_ik(p_m: np.ndarray, r: np.ndarray, q0_deg, iters: int = 30, damping: float = 1e-3,
             rot_weight: float = 0.3) -> tuple[np.ndarray, float, float]:
    """Damped least-squares IK iterating from q0, with joint limits. Returns (q in deg, position error in mm, orientation error in deg)."""
    q = np.deg2rad(np.asarray(q0_deg, dtype=float))
    lo, hi = np.deg2rad(JOINT_MIN_DEG), np.deg2rad(JOINT_MAX_DEG)
    w = np.array([1.0, 1.0, 1.0, rot_weight, rot_weight, rot_weight])
    for _ in range(iters):
        T, J = fk_jacobian(q)
        e = np.concatenate([p_m - T[:3, 3], rot_log(r @ T[:3, :3].T)])
        if np.linalg.norm(e[:3]) < 1e-4 and np.linalg.norm(e[3:]) < 1e-3:
            break
        jw = J * w[:, None]
        dq = jw.T @ np.linalg.solve(jw @ jw.T + damping * np.eye(6), e * w)
        n = np.max(np.abs(dq))
        if n > 0.2:  # at most 0.2 rad per iteration, to avoid jumping around near singularities
            dq *= 0.2 / n
        q = np.clip(q + dq, lo, hi)
    T, _ = fk_jacobian(q)
    pos_err = float(np.linalg.norm(p_m - T[:3, 3]) * 1000.0)
    rot_err = float(np.degrees(np.linalg.norm(rot_log(r @ T[:3, :3].T))))
    return np.rad2deg(q), pos_err, rot_err


# ---------------------------------------------------------------------------
# Twin window
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
# Look upward for the mujoco_haircut folder next to this project (HHE-robot\Client_RHCR\piper → HHE-robot\mujoco_haircut)
DEFAULT_HAIRCUT_DIR = next((d / "mujoco_haircut" for d in HERE.parents if (d / "mujoco_haircut").is_dir()),
                           HERE.parent.parent / "mujoco_haircut")
_LEROBOT_PYTHON = Path.home() / "miniconda3" / "envs" / "lerobot" / "python.exe"
DEFAULT_VIEWER_PYTHON = _LEROBOT_PYTHON if _LEROBOT_PYTHON.exists() else Path(sys.executable)


def load_init_joints(haircut_dir: Path) -> list[float]:
    """Initial twin joint angles = piper.init_joints_deg in mujoco_haircut/scene_config.json (the pose shown when the window starts)."""
    try:
        cfg = json.loads((haircut_dir / "scene_config.json").read_text(encoding="utf-8"))
        q = cfg["piper"]["init_joints_deg"]
        if len(q) == 6:
            return [float(v) for v in q]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return [0.0, 50.0, -50.0, 0.0, 30.0, 0.0]


class MujocoTwin:
    """A background thread sends the latest joint angles to the viewer at a fixed rate; show_* only update state, never block and never raise."""

    def __init__(self, port: int = 50555, spawn: bool = True, viewer_python: str | None = None,
                 haircut_dir: str | None = None, rate: float = 60.0, log=print, start_deg=None,
                 tool_view: bool = False):
        self.addr = ("127.0.0.1", port)
        # True: single window twin_view.py (overview + picture-in-picture tool view); False: the original mujoco_haircut overview window
        self.tool_view = tool_view
        self.spawn = spawn
        self.viewer_python = Path(viewer_python) if viewer_python else DEFAULT_VIEWER_PYTHON
        self.haircut_dir = Path(haircut_dir) if haircut_dir else DEFAULT_HAIRCUT_DIR
        self.period = 1.0 / rate
        self.log = log
        # Initial joint angles: the argument (haircut working pose) if given, otherwise init_joints_deg from scene_config
        self.q = np.array(start_deg if start_deg is not None else load_init_joints(self.haircut_dir), dtype=float)
        self.g: float | None = 0.0
        self.state = ""  # receiver state (following/paused/waiting), shown in the twin window
        self.lock = threading.Lock()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.running = False
        self._last_warn = 0.0

    # --- Lifecycle ---
    def start(self) -> None:
        if self.spawn:
            if not self.viewer_python.exists():
                self.log(f"Twin: {self.viewer_python} not found; open the window yourself or pass the lerobot environment's python with --twin-python")
            else:
                if self.tool_view:
                    self._spawn([str(self.viewer_python), str(HERE / "twin_view.py"), "--port", str(self.addr[1])],
                                f"MuJoCo window: overview + tool view at the bottom-right (udp {self.addr[1]})")
                else:
                    self._spawn([str(self.viewer_python), "-m", "haircut_sim.viewer", "--port", str(self.addr[1])],
                                f"MuJoCo overview window ({self.haircut_dir.name}, udp {self.addr[1]})")
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _spawn(self, cmd: list[str], name: str) -> None:
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        try:
            # Separate process group: Ctrl+C in the receiver does not close the window, so the trajectory can still be viewed after exit
            subprocess.Popen(cmd, cwd=str(self.haircut_dir), creationflags=flags)
            self.log(f"Twin: started {name}")
        except OSError as error:
            self.log(f"Twin: failed to start {name}: {error}")

    def close(self) -> None:
        self.running = False  # the window stays open to view the full trajectory; just close it

    def _loop(self) -> None:
        while self.running:
            with self.lock:
                msg = {"s": "piper", "q": [round(float(v), 3) for v in self.q], "g": self.g}
            try:
                self.sock.sendto(json.dumps(msg, separators=(",", ":")).encode(), self.addr)
                if self.state:
                    self.sock.sendto(json.dumps({"s": "status", "state": self.state}).encode(), self.addr)
            except OSError:
                pass  # window not open or closed: drop the packet
            time.sleep(self.period)

    # --- State ---
    def home_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Flange pose for the twin's current joint angles: (position in mm, rotation matrix)."""
        with self.lock:
            T, _ = fk_jacobian(np.deg2rad(self.q))
        return T[:3, 3] * 1000.0, T[:3, :3]

    def show_target(self, p_mm: np.ndarray, r: np.ndarray) -> None:
        """dry-run: end-effector target → IK → joint angles."""
        with self.lock:
            q0 = self.q.copy()
        target_p = np.asarray(p_mm, dtype=float) / 1000.0
        current, _ = fk_jacobian(np.deg2rad(q0))
        # An actual-pose clutch anchor must leave the displayed joints untouched.
        if (np.linalg.norm(target_p - current[:3, 3]) < 1e-9
                and np.linalg.norm(rot_log(r @ current[:3, :3].T)) < 1e-9):
            return
        q, pos_err, rot_err = solve_ik(target_p, r, q0)
        with self.lock:
            self.q = q
        now = time.monotonic()
        if (pos_err > 2.0 or rot_err > 2.0) and now - self._last_warn > 1.0:
            self._last_warn = now
            self.log(f"Twin: IK did not fully converge (joint limits/outside the workspace), error {pos_err:.1f}mm {rot_err:.1f}deg")

    def show_joints(self, q_deg, gripper_mm: float | None = None) -> None:
        """Real robot: show the joint angles from feedback."""
        with self.lock:
            self.q = np.asarray(q_deg, dtype=float)[:6].copy()
            if gripper_mm is not None:
                self.g = float(gripper_mm)

    def gripper(self, opening_mm: float) -> None:
        with self.lock:
            self.g = float(opening_mm)

    def status(self, text: str) -> None:
        self.state = text
