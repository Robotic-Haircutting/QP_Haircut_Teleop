"""Two small simulated views in the MuJoCo window (rendered offscreen, overlaid on the main window and composited into the recording):

1. Gripper view: the camera follows behind and above the gripper, looking forward along the clipper (moves with the end effector, but the horizon stays level instead of rolling);
   the gripper and flange are drawn semi-transparent so you can see through them how far the clipper tip is from the head. A line is drawn from the tip to the nearest scalp point, labeled with the distance.
2. Haircut progress view: looks at the hair from behind and above the head; the arm is hidden except for the semi-transparent gripper and clipper; red/green dots = not cut/cut, orange line = tip trajectory.

The main view follows the tip in viewer.py (F toggles follow/free).
"""

import cv2
import mujoco
import numpy as np

from .coverage import tip_offset, tip_pos

SEE_THROUGH_BODIES = ("piper/link6", "piper/link7", "piper/link8")
NEAR_RGBA = np.array([0.1, 0.85, 0.3, 1.0], dtype=np.float32)
FAR_RGBA = np.array([1.0, 0.55, 0.0, 1.0], dtype=np.float32)


def _label(img: np.ndarray, lines: list[str]) -> None:
    for k, txt in enumerate(lines):
        org = (10, 26 + 26 * k)
        cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)


class SimViews:
    def __init__(self, model, data, cfg: dict, width: int = 640, height: int = 360, tip_label: str = "Tip"):
        self.model, self.data, self.cfg = model, data, cfg
        self.tip_label = tip_label  # shows "Flange" when no tool is mounted (AprilTag flange reference)
        self.w, self.h = width, height
        self.renderer = mujoco.Renderer(model, height, width)
        self.offset = tip_offset(cfg)
        vcfg = cfg.get("views", {})
        self.alpha_gripper = float(vcfg.get("gripper", {}).get("see_through_alpha", 0.15))
        self.near_m = float(cfg.get("hair", {}).get("cover_radius_m", 0.03))

        g = vcfg.get("gripper", {})
        self.grip_back_m = float(g.get("back_m", 0.22))  # how far behind the tip the camera is (along the clipper axis)
        self.grip_up_m = float(g.get("up_m", 0.06))  # how far it is raised (world vertical)
        self.grip_ahead_m = float(g.get("look_ahead_m", 0.05))  # how far ahead of the tip it looks
        self.grip_cam = mujoco.MjvCamera()
        self.grip_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.link6 = model.body("piper/link6").id

        self.see_through = {model.body(n).id for n in SEE_THROUGH_BODIES}
        self.piper_bodies = {b for b in range(model.nbody) if model.body(b).name.startswith("piper/")}
        self.clipper_gid = model.geom("clipper").id if "clipper" in [model.geom(i).name for i in range(model.ngeom)] else -1

        # Head mesh vertices (head body frame); transformed to the world frame with the current head pose every frame for distances (the head moves during calibration)
        gid = model.geom("head").id
        self.head_gid = gid if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH else -1
        if self.head_gid >= 0:
            mid = model.geom_dataid[gid]
            self.head_local = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]].astype(float)

        h = vcfg.get("hair_map", {})
        self.map_cam = mujoco.MjvCamera()
        self.map_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.map_cam.distance = float(h.get("distance_m", 0.45))
        self.map_cam.elevation = float(h.get("elevation_deg", -40.0))
        self._map_extra_az = float(h.get("azimuth_offset_deg", 0.0))
        self.dist_m = float("nan")
        self.near_pt = None

    def _head_world(self) -> np.ndarray:
        g = self.head_gid
        return self.head_local @ self.data.geom_xmat[g].reshape(3, 3).T + self.data.geom_xpos[g]

    def tip_to_head(self) -> tuple[np.ndarray, float, np.ndarray | None]:
        tip = tip_pos(self.model, self.data, self.offset)
        if self.head_gid < 0:
            return tip, float("nan"), None
        verts = self._head_world()
        d2 = np.sum((verts - tip) ** 2, axis=1)
        k = int(np.argmin(d2))
        return tip, float(np.sqrt(d2[k])), verts[k]

    def _copy_user(self, user_scn) -> None:
        s = self.renderer.scene
        if user_scn is None:
            return
        for i in range(user_scn.ngeom):
            if s.ngeom >= s.maxgeom:
                break
            src, g = user_scn.geoms[i], s.geoms[s.ngeom]
            mujoco.mjv_initGeom(g, src.type, src.size, src.pos, src.mat.flatten(), src.rgba)
            g.size[:] = src.size
            s.ngeom += 1

    def _fade(self, bodies: set[int], alpha: float, clipper_alpha: float) -> None:
        s = self.renderer.scene
        for i in range(s.ngeom):
            g = s.geoms[i]
            if g.objtype != mujoco.mjtObj.mjOBJ_GEOM:
                continue
            if g.objid == self.clipper_gid:
                g.rgba[3] = min(g.rgba[3], clipper_alpha)
            elif self.model.geom_bodyid[g.objid] in bodies:
                g.rgba[3] = min(g.rgba[3], alpha)

    def _line(self, a, b, rgba, width=4.0) -> None:
        s = self.renderer.scene
        if s.ngeom >= s.maxgeom:
            return
        g = s.geoms[s.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.zeros(3), np.zeros(3), np.eye(3).flatten(), rgba)
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, width, np.asarray(a, float), np.asarray(b, float))
        s.ngeom += 1

    def _aim_gripper_cam(self, tip: np.ndarray) -> None:
        axis = self.data.xmat[self.link6].reshape(3, 3)[:, 2]  # clipper direction = flange z axis
        eye = tip - self.grip_back_m * axis + np.array([0.0, 0.0, self.grip_up_m])
        target = tip + self.grip_ahead_m * axis
        d = target - eye
        c = self.grip_cam
        c.lookat[:] = target
        c.distance = float(np.linalg.norm(d))
        c.azimuth = float(np.degrees(np.arctan2(d[1], d[0])))
        c.elevation = float(np.degrees(np.arcsin(np.clip(d[2] / c.distance, -1, 1))))

    def render_gripper(self, user_scn=None) -> np.ndarray:
        tip, dist, near = self.tip_to_head()
        self.dist_m = dist
        self._aim_gripper_cam(tip)
        self.renderer.update_scene(self.data, self.grip_cam)
        self._copy_user(user_scn)
        self._fade(self.see_through, self.alpha_gripper, 0.45)
        if near is not None:
            self._line(tip, near, NEAR_RGBA if dist < self.near_m else FAR_RGBA)
        img = np.ascontiguousarray(self.renderer.render())
        _label(img, ["Gripper view (see-through)", f"{self.tip_label} to head: {dist * 1000:.0f} mm" if np.isfinite(dist) else f"{self.tip_label} to head: -"])
        return img

    def render_hair_map(self, user_scn=None, center=None, face_dir=None) -> np.ndarray:
        if center is None:
            center = self.data.xpos[self.model.body("head").id] + np.array([0.0, 0.0, 0.18])
        self.map_cam.lookat[:] = center
        if face_dir is not None:
            # Looking from behind the head: view direction = face direction
            self.map_cam.azimuth = float(np.degrees(np.arctan2(face_dir[1], face_dir[0]))) + self._map_extra_az
        self.renderer.update_scene(self.data, self.map_cam)
        self._copy_user(user_scn)
        self._fade(self.piper_bodies - self.see_through, 0.0, 0.8)  # hide the rest of the arm so it does not block the hair
        self._fade(self.see_through, 0.25, 0.8)
        img = np.ascontiguousarray(self.renderer.render())
        _label(img, ["Hair map (from behind, arm hidden)"])
        return img

    def close(self) -> None:
        self.renderer.close()


# Main view for each haircut region: which side of the head the camera is on (angle from the face direction: 0 = in front of the face, 180 = directly behind, positive = the person's left) and elevation
REGION_VIEWS = {
    "back": (135.0, -30.0),  # back of the head: obliquely from behind-left (same as the default view, with the leader also in frame)
    "left": (100.0, -25.0),  # left side: from the person's left, slightly behind
    "right": (-100.0, -40.0),  # right side: from the person's right, slightly behind and higher (over the Piper base)
    "top": (150.0, -65.0),  # top: looking down from behind and above
}


def _wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


class CameraDirector:
    """Turn the main view to the region being cut: when the tip is near the hair, turn smoothly to the preset angle of the region (top/back/left/right)
    containing the nearest hair point. Dragging the view with the mouse pauses auto-turning for a few seconds so it does not fight the user."""

    def __init__(self, cov, regions: list[str], cfg: dict):
        c = cfg.get("camera", {})
        self.cov, self.regions = cov, regions
        self.phi = float(np.degrees(np.arctan2(cov.face_dir[1], cov.face_dir[0])))
        self.near_m = float(c.get("region_switch_distance_m", 0.08))
        self.rate = float(c.get("region_turn_rate", 0.04))  # fraction of the remaining angle turned per frame
        self.pause_s = float(c.get("manual_pause_s", 5.0))
        self.views = {k: tuple(c.get("region_views", {}).get(k, v)) for k, v in REGION_VIEWS.items()}
        self.region = "back"
        self._last = None  # (azimuth, elevation) set in the previous frame
        self._paused_until = 0.0

    def target(self) -> tuple[float, float]:
        pos_deg, el = self.views[self.region]
        return _wrap(self.phi + pos_deg + 180.0), el  # view direction = opposite of the camera position direction

    def update(self, cam, tip: np.ndarray, now: float) -> str:
        if self._last is not None and (abs(_wrap(cam.azimuth - self._last[0])) > 0.5 or abs(cam.elevation - self._last[1]) > 0.5):
            self._paused_until = now + self.pause_s  # the user dragged the view
        d2 = np.sum((self.cov.pts - tip) ** 2, axis=1)
        k = int(np.argmin(d2))
        if d2[k] < self.near_m ** 2:
            self.region = self.regions[int(self.cov.region[k])]
        if now >= self._paused_until:
            az, el = self.target()
            cam.azimuth = _wrap(cam.azimuth + self.rate * _wrap(az - cam.azimuth))
            cam.elevation = cam.elevation + self.rate * (el - cam.elevation)
        self._last = (float(cam.azimuth), float(cam.elevation))
        return self.region
