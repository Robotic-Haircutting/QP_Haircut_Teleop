"""Head position calibration: teleoperate Piper so the clipper tip touches the top (highest point) of the real head, press Enter in the MuJoCo window to record it,
the whole head is translated there (orientation unchanged, from head.euler_deg in scene_config.json), and Y writes it back to scene_config.json.

Only the top of the head is used: the highest point on the real head is aligned with the highest point on the model to get the x/y/z translation. The top is fairly flat, so touching 1-2 cm off center
turns directly into horizontal position error; touch as close to the highest center point as possible. For higher accuracy, change POINTS back to multiple points (back/left/right/top).
"""

import json
import shutil
from pathlib import Path

import mujoco
import numpy as np

from .coverage import tip_offset, tip_pos
from .scene import DEFAULT_CONFIG

POINTS = [
    ("top", "TOP of the head (highest point, center of the crown)"),
]
RAW_RGBA = np.array([0.15, 0.4, 1.0, 1.0], dtype=np.float32)
FIT_RGBA = np.array([0.1, 0.8, 0.3, 0.9], dtype=np.float32)
TIP_RGBA = np.array([1.0, 0.85, 0.0, 1.0], dtype=np.float32)


def head_landmarks_local(model, data, face_axis_local) -> dict[str, np.ndarray]:
    """The 5 most prominent points on the head mesh, expressed in the head body frame."""
    gid = model.geom("head").id
    mid = model.geom_dataid[gid]
    va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    world = (data.geom_xmat[gid].reshape(3, 3) @ model.mesh_vert[va:va + vn].T).T + data.geom_xpos[gid]
    bid = model.body("head").id
    R, p = data.xmat[bid].reshape(3, 3), data.xpos[bid]
    local = (world - p) @ R  # = R^T (w - p)
    up_local = R.T @ np.array([0.0, 0.0, 1.0])
    f = np.asarray(face_axis_local, dtype=float)
    f = f - (f @ up_local) * up_local
    f /= np.linalg.norm(f)
    left = np.cross(up_local, f)
    return {
        "front": local[np.argmax(local @ f)],
        "back": local[np.argmin(local @ f)],
        "left": local[np.argmax(local @ left)],
        "right": local[np.argmin(local @ left)],
        "top": local[np.argmax(local @ up_local)],
    }


def fit_pose(local_pts: np.ndarray, world_pts: np.ndarray, R_cur: np.ndarray, fit_yaw: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve world ≈ Rz(θ)·R_cur·local + t. Returns (R_new, t, per-point residuals in m).

    fit_yaw=False (default): keep the orientation from euler_deg in scene_config and solve only for the translation. The left and right sides are fairly flat,
    so touching a few centimeters forward or back would skew the angle by tens of degrees; the orientation is therefore set directly in the config (by default the face points away from Piper).
    """
    a = local_pts @ R_cur.T  # points in the current orientation, centered on the body origin
    if not fit_yaw:
        t = np.mean(world_pts - a, axis=0)
        return R_cur, t, np.linalg.norm(a + t - world_pts, axis=1)
    ca, cw = a[:, :2].mean(0), world_pts[:, :2].mean(0)
    A, W = a[:, :2] - ca, world_pts[:, :2] - cw
    th = np.arctan2(np.sum(A[:, 0] * W[:, 1] - A[:, 1] * W[:, 0]), np.sum(A[:, 0] * W[:, 0] + A[:, 1] * W[:, 1]))
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    t = np.zeros(3)
    t[:2] = cw - (Rz[:2, :2] @ ca)
    t[2] = np.mean(world_pts[:, 2] - a[:, 2])
    pred = a @ Rz.T + t
    return Rz @ R_cur, t, np.linalg.norm(pred - world_pts, axis=1)


class Calibrator:
    def __init__(self, scene, cfg: dict, config_path=None):
        self.scene, self.cfg = scene, cfg
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG
        self.offset = tip_offset(cfg)
        m, d = scene.model, scene.data
        self.bid = m.body("head").id
        self.orig_pos = m.body_pos[self.bid].copy()
        self.orig_quat = m.body_quat[self.bid].copy()
        face = cfg.get("hair", {}).get("face_axis_local", [0, 1, 0])
        self.local = head_landmarks_local(m, d, face)
        self.recorded: list[np.ndarray] = []
        self.result = None  # (R_new, t, residuals)
        self.saved_msg = ""
        self.scn = None

    # ---------- Keys ----------
    def record(self) -> None:
        if self.result is not None or len(self.recorded) >= len(POINTS):
            return
        self.recorded.append(self.tip().copy())
        if len(self.recorded) == len(POINTS):
            self._fit()

    def undo(self) -> None:
        if self.result is not None:
            self.restart()
        elif self.recorded:
            self.recorded.pop()

    def restart(self) -> None:
        self.recorded, self.result, self.saved_msg = [], None, ""
        self._set_head(self.orig_pos, self.orig_quat)

    def save(self) -> str:
        if self.result is None:
            return ""
        R, t, _ = self.result
        raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        head = raw["head"]
        stand = float(head.get("stand_height_m", 0.0))
        euler = np.asarray(head.get("euler_deg", [0, 0, 0]), dtype=float)
        dyaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]) - np.arctan2(self._R_orig()[1, 0], self._R_orig()[0, 0]))
        euler[2] = (euler[2] + dyaw + 180.0) % 360.0 - 180.0
        head["pos"] = [round(float(t[0]), 4), round(float(t[1]), 4), round(float(t[2] - stand), 4)]
        head["euler_deg"] = [round(float(v), 2) for v in euler]
        backup = self.config_path.with_name(self.config_path.stem + ".backup.json")
        shutil.copyfile(self.config_path, backup)
        self.config_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        self.saved_msg = f"Saved: head pos {head['pos']} m, yaw {head['euler_deg'][2]} deg"
        print(f"Head position written to {self.config_path}: pos={head['pos']}, euler_deg={head['euler_deg']} (old config backed up as {backup.name})", flush=True)
        return self.saved_msg

    # ---------- Internals ----------
    def tip(self) -> np.ndarray:
        return tip_pos(self.scene.model, self.scene.data, self.offset)

    def _R_orig(self) -> np.ndarray:
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, self.orig_quat)
        return R.reshape(3, 3)

    def _fit(self) -> None:
        names = [k for k, _ in POINTS]
        L = np.array([self.local[k] for k in names])
        W = np.array(self.recorded)
        # With fewer than 3 points the orientation cannot be determined; solve only for the translation
        fit_yaw = bool(self.cfg.get("calibration", {}).get("fit_yaw", False)) and len(names) >= 3
        R_new, t, res = fit_pose(L, W, self._R_orig(), fit_yaw)
        try:
            pts = {n: [round(float(v), 5) for v in p] for n, p in zip(names, W, strict=True)}
            self.config_path.with_name("calibration_points.json").write_text(json.dumps(pts, indent=2), encoding="utf-8")
        except OSError:
            pass
        self.result = (R_new, t, res)
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, R_new.flatten())
        self._set_head(t, q)
        print("Head calibration result: " + ", ".join(f"{n} error {r * 1000:.1f} mm" for n, r in zip(names, res, strict=True)), flush=True)

    def _set_head(self, pos, quat) -> None:
        m, d = self.scene.model, self.scene.data
        m.body_pos[self.bid] = pos
        m.body_quat[self.bid] = quat
        mujoco.mj_forward(m, d)

    # ---------- Display ----------
    def draw(self, scn) -> None:
        eye = np.eye(3).flatten()
        n = 0

        def sphere(p, r, rgba):
            nonlocal n
            if n < scn.maxgeom:
                mujoco.mjv_initGeom(scn.geoms[n], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0, 0]), np.asarray(p, float), eye, rgba)
                n += 1

        sphere(self.tip(), 0.006, TIP_RGBA)
        for p in self.recorded:
            sphere(p, 0.008, RAW_RGBA)
        if self.result is not None:
            m, d = self.scene.model, self.scene.data
            R, p = d.xmat[self.bid].reshape(3, 3), d.xpos[self.bid]
            for k, _ in POINTS:
                sphere(R @ self.local[k] + p, 0.006, FIT_RGBA)
        scn.ngeom = n

    def text(self) -> tuple[str, str]:
        k = len(self.recorded)
        if self.result is None:
            name, desc = POINTS[k]
            return (f"CALIBRATION {k + 1}/{len(POINTS)}\n\nKeys",
                    f"Touch the clipper tip to: {desc}\n\nENTER = record point   U = undo last   R = restart")
        res = self.result[2] * 1000
        lines = "   ".join(f"{n} {r:.0f}mm" for (n, _), r in zip(POINTS, res, strict=True))
        warn = "  <-- large error, check points / tool offset" if res.max() > 20 else ""
        status = self.saved_msg or "Y = save to scene_config.json   R = redo"
        return ("CALIBRATION done\nPer-point error\n\nKeys", f"head moved to fitted pose{warn}\n{lines}\n\n{status}")
