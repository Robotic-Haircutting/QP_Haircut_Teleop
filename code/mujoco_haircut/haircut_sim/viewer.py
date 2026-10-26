"""MuJoCo window: receives joint angles from the teleoperation process and shows the SO-101 leader + Piper + head in real time.

  python -m haircut_sim.viewer            # wait for real-robot data (with teleop_sim or a bat file)
  python -m haircut_sim.viewer --mock     # nothing connected; both arms swing sinusoidally to check the scene
"""

import argparse
import json
import math
import socket
import time

import mujoco
import numpy as np

from .coverage import REGIONS, HairCoverage, tip_offset, tip_pos
from .scene import Scene, load_config

DOT_UNCOVERED = np.array([0.86, 0.15, 0.15, 1.0], dtype=np.float32)
DOT_COVERED = np.array([0.09, 0.64, 0.29, 1.0], dtype=np.float32)
PATH_NEAR = np.array([0.92, 0.35, 0.05, 1.0], dtype=np.float32)
PATH_FAR = np.array([0.35, 0.55, 0.95, 0.8], dtype=np.float32)


class TrailOverlay:
    """Drawn in the viewer's user_scn: hair sample points (red = not cut, green = cut), the clipper-tip trajectory, and a small ball at the tip.

    Only appends/recolors incrementally instead of rebuilding every frame, so thousands of geoms do not slow the frame rate. Press C to clear and start over.
    """

    def __init__(self, scene: Scene, cfg: dict, max_dots: int = 2500, step_m: float = 0.002):
        self.scene, self.step = scene, step_m
        self.offset = tip_offset(cfg)
        try:
            self.cov = HairCoverage(scene.model, scene.data, cfg)
        except Exception:
            self.cov = None  # no head mesh: draw only the trajectory
        n = 0 if self.cov is None else len(self.cov.pts)
        rng = np.random.default_rng(0)
        self.dot_idx = np.sort(rng.choice(n, size=min(n, max_dots), replace=False)) if n else np.zeros(0, int)
        self.slot_of = np.full(n, -1)
        self.slot_of[self.dot_idx] = np.arange(len(self.dot_idx))
        self.last = None
        self.scn = None

    def attach(self, scn) -> None:
        self.scn = scn
        self.clear()

    def clear(self) -> None:
        if self.cov is not None:
            self.cov.reset()
        self.last = None
        scn = self.scn
        if scn is None:
            return
        eye = np.eye(3).flatten()
        for k, i in enumerate(self.dot_idx):
            mujoco.mjv_initGeom(scn.geoms[k], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.0035, 0, 0]), self.cov.pts[i], eye, DOT_UNCOVERED)
        k = len(self.dot_idx)
        mujoco.mjv_initGeom(scn.geoms[k], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.006, 0, 0]), np.zeros(3), eye, np.array([1, 0.85, 0, 1], dtype=np.float32))
        scn.ngeom = k + 1

    def update(self) -> None:
        if self.scn is None:
            return
        scn = self.scn
        p = tip_pos(self.scene.model, self.scene.data, self.offset)
        scn.geoms[len(self.dot_idx)].pos[:] = p
        if self.last is not None and np.linalg.norm(p - self.last) < self.step:
            return
        near = False
        if self.cov is not None:
            new, dist = self.cov.update(p)
            near = dist < self.cov.radius
            for s in self.slot_of[new]:
                if s >= 0:
                    scn.geoms[s].rgba[:] = DOT_COVERED
        if self.last is not None and scn.ngeom < scn.maxgeom:
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.zeros(3), np.zeros(3), np.eye(3).flatten(), PATH_NEAR if near else PATH_FAR)
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, 3.0 if near else 1.5, self.last, p)
            scn.ngeom += 1
        self.last = p

    def text(self) -> str:
        return "" if self.cov is None else f"{self.cov.ratio * 100:.1f}%   (press C to clear)"

    def region_text(self) -> str:
        """Coverage of each region: Top / Back / Left / Right."""
        if self.cov is None:
            return ""
        r = self.cov.region_ratios()
        return "  ".join(f"{n.capitalize()} {r[n] * 100:.0f}%" for n in REGIONS)


def _show_insets(v, left_bottom_up: list, right_bottom_up: list, width_frac: float = 0.26) -> None:
    """Small pictures overlaid in the MuJoCo window: a column in the bottom-left stacked bottom-up (haircut progress, gripper view), and the real camera in the bottom-right. Images are RGB."""
    import cv2

    vp = v.viewport
    if vp is None or vp.width < 200:
        return
    w = int(vp.width * width_frac)
    items = []
    for x, column in ((10, left_bottom_up), (vp.width - w - 10, right_bottom_up)):
        y = 10
        for img in column:
            if img is None:
                continue
            h = int(w * img.shape[0] / img.shape[1])
            if y + h > vp.height:
                break
            items.append((mujoco.MjrRect(x, y, w, h), cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)))
            y += h + 8
    if items:
        v.set_images(items)


class Receiver:
    def __init__(self, port: int):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", port))
        self.sock.setblocking(False)
        self.last = {"piper": 0.0, "so101": 0.0}
        self.count = {"piper": 0, "so101": 0}

    def drain(self) -> dict:
        """Drain the receive buffer, keeping only the latest frame from each source."""
        latest = {}
        while True:
            try:
                buf, _ = self.sock.recvfrom(4096)
            except (BlockingIOError, ConnectionResetError):
                break
            try:
                msg = json.loads(buf)
                latest[msg["s"]] = msg
            except (ValueError, KeyError):
                continue
        now = time.perf_counter()
        for k in latest:
            self.last[k] = now
            self.count[k] = self.count.get(k, 0) + 1
        return latest


def apply(scene: Scene, latest: dict) -> None:
    if "piper" in latest:
        scene.set_piper(latest["piper"]["q"], latest["piper"].get("g"))
    if "so101" in latest:
        scene.set_so101(latest["so101"]["q"], latest["so101"].get("g"))


def mock_frame(t: float) -> dict:
    s = math.sin
    return {
        "so101": {"q": [20 * s(0.5 * t), 10 * s(0.3 * t), 10 * s(0.4 * t), 15 * s(0.6 * t), 30 * s(0.2 * t)], "g": 50 + 50 * s(0.7 * t)},
        "piper": {"q": [20 * s(0.5 * t), 50 + 10 * s(0.3 * t), -50 + 10 * s(0.4 * t), 0, 30 + 15 * s(0.6 * t), 30 * s(0.2 * t)], "g": 35 + 35 * s(0.7 * t)},
    }


def status_text(rx: Receiver | None) -> str:
    if rx is None:
        return "mock"
    now = time.perf_counter()
    parts = []
    for k, name in (("so101", "SO-101"), ("piper", "Piper")):
        age = now - rx.last[k]
        parts.append(f"{name}: {'live' if age < 0.5 else 'waiting'}")
    return "   ".join(parts)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="scene config JSON, default scene_config.json")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--mock", action="store_true", help="receive no data; both arms swing on their own")
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--calibrate", action="store_true", help="head position calibration mode (touch with the clipper tip, press Enter to record)")
    ap.add_argument("--record", action="store_true", help="start recording as soon as the window opens (V in the window starts/stops)")
    ap.add_argument("--out_dir", default=None, help="directory for recordings, default mujoco_haircut/results")
    ap.add_argument("--camera", type=int, default=None, help="real camera index, -1 for none; defaults to real_camera in scene_config.json")
    ap.add_argument("--duration", type=float, default=0.0, help="optional: close the window automatically after this many seconds, for hardware-free checks")
    ap.add_argument("--headless", type=float, default=0.0, help="run without a window for this many seconds, then print packet statistics (self-test)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    scene = Scene(cfg)
    rx = None if args.mock else Receiver(args.port or cfg.get("udp_port", 50555))
    period = 1.0 / args.fps
    t0 = time.perf_counter()

    control_events = []

    def tick():
        latest = mock_frame(time.perf_counter() - t0) if rx is None else rx.drain()
        if latest.get("control", {}).get("command") == "stop_recording":
            control_events.append("stop_recording")
        apply(scene, latest)
        scene.forward()

    if args.headless > 0:
        while time.perf_counter() - t0 < args.headless:
            tick()
            time.sleep(period)
        q = scene.data.qpos
        print(json.dumps({"count": rx.count if rx else None, "piper_deg": np.rad2deg(q[scene._piper] - scene._piper_offsets).round(2).tolist(),
                          "so101_deg": np.rad2deg(q[scene._so101]).round(2).tolist()}))
        return

    import mujoco.viewer

    calib = None
    if args.calibrate:
        from .calibrate import Calibrator

        calib = Calibrator(scene, cfg, args.config)
    trail = TrailOverlay(scene, cfg) if calib is None and cfg.get("trail", {}).get("enabled", True) else None
    want_clear = {"flag": False}
    keys: list[int] = []

    def on_key(keycode: int) -> None:
        if keycode == ord("C"):
            want_clear["flag"] = True
        keys.append(keycode)

    from .recorder import Recorder

    # Real camera: opens a preview window alongside; when recording, it is composited to the right of the MuJoCo view
    rc = cfg.get("real_camera", {})
    cam_index = args.camera if args.camera is not None else (int(rc.get("index", 1)) if rc.get("enabled", False) else -1)
    camera = None
    if cam_index >= 0 and args.headless <= 0:
        from . import camera as camera_mod

        try:
            camera = camera_mod.CameraStream(cam_index, int(rc.get("width", 640)), int(rc.get("height", 480)))
            print(f"Camera {cam_index} opened; it will be recorded together with MuJoCo", flush=True)
        except RuntimeError as e:
            print(f"Camera not opened, recording the MuJoCo view only: {e}", flush=True)
    rec_kw = {} if args.out_dir is None else {"out_dir": args.out_dir}
    # Recording: left column gripper view + haircut progress, main view in the middle, camera on the right (no right column without a camera)
    rec = Recorder(scene.model, scene.data, left_w=640, right_w=640 if camera is not None else 0, **rec_kw)
    view_cfg = cfg.get("camera", {})
    follow = {"on": bool(view_cfg.get("follow_tip", True))}
    tip_off = tip_offset(cfg)

    try:
        with mujoco.viewer.launch_passive(scene.model, scene.data, show_left_ui=False, show_right_ui=False, key_callback=on_key) as v:
            if args.record:
                rec.start()
            cam = cfg.get("camera", {})
            v.cam.lookat[:] = cam.get("lookat", [0.25, 0.2, 0.2])
            v.cam.distance = cam.get("distance", 1.5)
            v.cam.azimuth = cam.get("azimuth", 200)
            v.cam.elevation = cam.get("elevation", -25)
            if follow["on"]:
                # Follow view: obliquely from behind and above the head, with the clipper tip, the back of the head and the leader all in frame
                v.cam.distance = float(view_cfg.get("follow_distance_m", 0.9))
                v.cam.azimuth = float(view_cfg.get("follow_azimuth_deg", -45))
                v.cam.elevation = float(view_cfg.get("follow_elevation_deg", -30))
            leader_w = float(view_cfg.get("follow_leader_weight", 0.45))
            leader_bid = scene.model.body("so101/gripper_frame_link").id
            if trail is not None:
                with v.lock():
                    trail.attach(v.user_scn)
            from .views import CameraDirector, SimViews

            views = SimViews(scene.model, scene.data, cfg)
            # When cutting moves to another region, the main view turns to it automatically (toward the region whose hair the tip is near)
            director = None
            if trail is not None and trail.cov is not None and view_cfg.get("follow_region", True):
                director = CameraDirector(trail.cov, REGIONS, cfg)
                if follow["on"]:
                    v.cam.azimuth, v.cam.elevation = director.target()
            hair_center = trail.cov.pts.mean(axis=0) if trail is not None and trail.cov is not None else None
            face_dir = trail.cov.face_dir if trail is not None and trail.cov is not None else None
            grip_img = map_img = cam_rgb = None
            t_views = 0.0
            while v.is_running():
                if args.duration > 0 and time.perf_counter() - t0 >= args.duration:
                    break
                t = time.perf_counter()
                with v.lock():
                    tick()
                    if control_events:
                        rec.stop()
                        control_events.clear()
                    if trail is not None:
                        if want_clear["flag"]:
                            trail.clear()
                            want_clear["flag"] = False
                        if rx is None or time.perf_counter() - rx.last["piper"] < 0.5:
                            trail.update()
                    if ord("V") in keys:
                        rec.toggle()
                        keys[:] = [k for k in keys if k != ord("V")]
                    if ord("F") in keys:
                        follow["on"] = not follow["on"]
                        keys[:] = [k for k in keys if k != ord("F")]
                    if follow["on"]:
                        # Main view follow: the look-at point lies between the clipper tip and the leader end effector (biased toward the tip); when cutting the sides the leader is on the other side of the head, so it is biased further toward the tip.
                        # Azimuth/elevation turn automatically with the region being cut; dragging with the mouse pauses auto-turning for a few seconds
                        tip = tip_pos(scene.model, scene.data, tip_off)
                        region = director.update(v.cam, tip, time.perf_counter()) if director is not None else "back"
                        w = leader_w if region in ("back", "top") else 0.15
                        target = (1 - w) * tip + w * scene.data.xpos[leader_bid]
                        v.cam.lookat[:] = v.cam.lookat + 0.15 * (target - v.cam.lookat)
                    if calib is not None:
                        piper_live = rx is None or time.perf_counter() - rx.last["piper"] < 0.5
                        while keys:
                            k = keys.pop(0)
                            if k == 257 and piper_live:  # Enter
                                calib.record()
                            elif k == ord("U"):
                                calib.undo()
                            elif k == ord("R"):
                                calib.restart()
                            elif k == ord("Y"):
                                calib.save()
                        calib.draw(v.user_scn)
                    cam_frame = camera.latest() if camera is not None else None
                    if time.perf_counter() - t_views >= 1.0 / 30:  # the two small simulated views render at 30 Hz
                        t_views = time.perf_counter()
                        grip_img = views.render_gripper(v.user_scn)
                        map_img = views.render_hair_map(v.user_scn, hair_center, face_dir)
                        if cam_frame is not None:
                            import cv2

                            cam_rgb = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
                    if rec.active:
                        lines = []
                        if trail is not None and trail.cov is not None:
                            lines = [f"Hair coverage {trail.cov.ratio * 100:.1f}%", trail.region_text()]
                        rec.capture(v.cam, v.user_scn, lines or None, left=[grip_img, map_img], right=[cam_rgb])
                if hasattr(v, "set_texts"):
                    if calib is not None:
                        left, right = calib.text()
                        v.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, f"Haircut sim\n{left}", f"{status_text(rx)}\n{right}"))
                    else:
                        extra = trail.text() if trail is not None else ""
                        regions = trail.region_text() if trail is not None else ""
                        dist = f"{views.dist_m * 1000:.0f} mm" if np.isfinite(views.dist_m) else "-"
                        view_txt = (f"follow tip, {director.region if director else 'back'} (F = free)") if follow["on"] else "free (F = follow tip)"
                        v.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, "Haircut sim\nHair covered\nRegions\nTip to head\nView\nVideo", f"{status_text(rx)}\n{extra}\n{regions}\n{dist}\n{view_txt}\n{'REC  (V = stop)' if rec.active else 'V = start recording'}"))
                keys.clear()
                v.sync()
                if hasattr(v, "set_images"):
                    _show_insets(v, [map_img, grip_img], [cam_rgb])
                time.sleep(max(0.0, period - (time.perf_counter() - t)))
    finally:
        rec.close()
        if "views" in locals():
            views.close()
        if camera is not None:
            camera.close()


if __name__ == "__main__":
    main()
