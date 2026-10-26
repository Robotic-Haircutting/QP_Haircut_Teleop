"""Single digital-twin window (since 2026-10-07 unified with the SO-101 haircut window of mujoco_haircut; see main()).

Older description below: overview + picture-in-picture "tool view" in the bottom-right.

The overview is the same as the mujoco_haircut viewer (reuses its Scene / Receiver / TrailOverlay / Recorder without modifying that project):
Piper + head + clipper-tip trajectory + hair coverage; V records, C clears the trajectory, and the mouse rotates the view freely.
Bottom-right small pictures: bottom = tool view (camera behind and above the clipper, looking at the head along the clipper, moving with the end effector), top = side view (clearance between the clipper tip and the hair).
In the small pictures the arm is semi-transparent; a line is drawn from the clipper tip (orange ball) to the nearest hair point, green >30 mm, yellow 10-30 mm, red <10 mm, with the distance in large text at the top right.

  python twin_view.py --port 50555      # started automatically by piper_receiver --twin mujoco
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(os.environ.get("HHE_ROBOT_ROOT", str(Path(__file__).resolve().parent.parent)))/"Client_RHCR/piper"))

import argparse
import json
import math
import socket
import time

import cv2
from twin_stream import TwinFrameServer
import mujoco
import mujoco.viewer
import numpy as np

from haircut_sim.coverage import tip_offset, tip_pos
from haircut_sim.recorder import Recorder
from haircut_sim.scene import Scene, load_config
from haircut_sim.viewer import Receiver, TrailOverlay, apply
from hair_progress import PersistentTrailOverlay
from head_alignment import flange_config,hide_absent_tool,apply_alignment,translate_coverage


def view_angles(d: np.ndarray) -> tuple[float, float]:
    """View direction → azimuth / elevation (deg) of the MuJoCo free camera."""
    d = d / np.linalg.norm(d)
    return math.degrees(math.atan2(d[1], d[0])), math.degrees(math.asin(np.clip(d[2], -1.0, 1.0)))


GREEN = np.array([0.1, 0.8, 0.2, 1.0], dtype=np.float32)
YELLOW = np.array([1.0, 0.8, 0.0, 1.0], dtype=np.float32)
RED = np.array([0.95, 0.1, 0.1, 1.0], dtype=np.float32)
TIP = np.array([1.0, 0.45, 0.0, 1.0], dtype=np.float32)


def copy_scene_overlays(destination, source) -> int:
    """Include desktop hair markers, trail, tip and distance line in phone renders."""
    count = min(source.ngeom, destination.maxgeom - destination.ngeom)
    first = destination.ngeom
    for i in range(count):
        src, dst = source.geoms[i], destination.geoms[first + i]
        mujoco.mjv_initGeom(dst, src.type, src.size.astype(float), src.pos.astype(float), src.mat.astype(float).ravel(), src.rgba.copy())
        dst.emission = src.emission
        dst.specular = src.specular
        dst.shininess = src.shininess
        dst.reflectance = src.reflectance
    destination.ngeom += count
    return count


def paint_distance_badge(frame, dist_mm, flange=False):
    if dist_mm is None:
        return
    color = (40, 185, 40) if dist_mm > 30 else (0, 180, 235) if dist_mm > 10 else (45, 45, 225)
    cv2.rectangle(frame, (8, 8), (260, 62), color, -1)
    cv2.putText(frame, f"{'Flange to hair' if flange else 'Tip to hair'}: {dist_mm:.0f} mm", (16, 44),
                cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255, 255), 2, cv2.LINE_AA)


def dist_color(mm: float) -> np.ndarray:
    return GREEN if mm > 30 else YELLOW if mm > 10 else RED


def add_distance_geoms(scn, tip: np.ndarray, near: np.ndarray, color: np.ndarray, tip_r: float = 0.005) -> int:
    """Append to the end of scn: a ball at the clipper tip, a ball at the nearest hair point, and a line between them. Returns the number of geoms appended."""
    eye = np.eye(3).flatten()
    n0 = scn.ngeom
    if n0 + 3 > scn.maxgeom:
        return 0
    mujoco.mjv_initGeom(scn.geoms[n0], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([tip_r, 0, 0]), tip, eye, TIP)
    mujoco.mjv_initGeom(scn.geoms[n0 + 1], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.004, 0, 0]), near, eye, color)
    g = scn.geoms[n0 + 2]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), eye, color)
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0015, tip, near)
    scn.ngeom = n0 + 3
    return 3


class ToolCamera:
    """Camera that follows the clipper tip, with smoothing."""

    def __init__(self, distance: float, ahead: float, tilt: float, smooth: float):
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance = distance
        self.ahead, self.tilt, self.smooth = ahead, tilt, smooth
        self.state = None

    def update(self, tip: np.ndarray, d: np.ndarray) -> None:
        az, el = view_angles(d)
        el = max(-89.0, min(89.0, el + self.tilt))
        look = tip + self.ahead * d
        if self.state is None:
            self.state = [look, az, el]
        else:
            a = self.smooth
            daz = (az - self.state[1] + 180.0) % 360.0 - 180.0  # take the shortest angle to avoid jumps at ±180°
            self.state = [self.state[0] + a * (look - self.state[0]), self.state[1] + a * daz,
                          self.state[2] + a * (el - self.state[2])]
        self.cam.lookat[:] = self.state[0]
        self.cam.azimuth = self.state[1]
        self.cam.elevation = self.state[2]


class FollowCamera:
    """Automatic follow for the main view: the camera sits outside the clipper tip, looking at the head, and orbits the head with the end effector.

    Look-at point = clipper tip; azimuth = view direction opposite to the horizontal direction head center → clipper tip;
    fixed elevation. Distance is unchanged (zoom with the wheel as you like). Dragging with the mouse pauses following, which resumes 3 s after release; F toggles it.
    """

    def __init__(self, center: np.ndarray, elevation: float, smooth: float):
        self.center, self.elevation, self.smooth = center, elevation, smooth
        self.enabled = False  # Stable overview for direction verification; F opts into following.
        self.state = None        # smoothed [lookat, azimuth, elevation]
        self.written = None      # values written to the viewer in the previous frame, used to detect user dragging
        self.user_until = 0.0

    def apply(self, cam, tip: np.ndarray) -> None:
        now = time.perf_counter()
        if self.written is not None:
            look, az, el = self.written
            moved = (np.linalg.norm(np.asarray(cam.lookat) - look) > 1e-4 or abs(cam.azimuth - az) > 1e-3
                     or abs(cam.elevation - el) > 1e-3)
            if moved:
                self.user_until = now + 3.0   # the user is dragging: do not take over the view for 3 s
                self.state = None
        if not self.enabled or now < self.user_until:
            self.written = None
            return
        h = tip[:2] - self.center[:2]
        if np.linalg.norm(h) > 0.02:
            az = math.degrees(math.atan2(-h[1], -h[0]))  # view direction from outside the clipper tip toward the head
        else:
            az = self.state[1] if self.state is not None else cam.azimuth
        look = tip.copy()  # Keep the tool tip at the center of the main view.
        if self.state is None:
            self.state = [np.asarray(cam.lookat, dtype=float).copy(), float(cam.azimuth), float(cam.elevation)]
        a = self.smooth
        daz = (az - self.state[1] + 180.0) % 360.0 - 180.0
        self.state = [self.state[0] + a * (look - self.state[0]), self.state[1] + a * daz,
                      self.state[2] + a * (self.elevation - self.state[2])]
        cam.lookat[:] = self.state[0]
        cam.azimuth = self.state[1]
        cam.elevation = self.state[2]
        self.written = (np.asarray(cam.lookat, dtype=float).copy(), float(cam.azimuth), float(cam.elevation))

    def label(self) -> str:
        if not self.enabled:
            return "off  (F = on)"
        return "paused (mouse)" if time.perf_counter() < self.user_until else "on  (F = off)"


def side_camera(cam, tip: np.ndarray, near: np.ndarray, d: np.ndarray, distance: float) -> None:
    """Side view: look at the gap between the clipper tip and the nearest hair point from the horizontal direction perpendicular to the clipper."""
    az, _ = view_angles(d)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (tip + near) / 2.0
    cam.distance = distance
    cam.azimuth = az + 90.0
    cam.elevation = -10.0


TWIN_VIDEO_DIR = Path(__file__).resolve().parent / "results" / "twin_video"
TWIN_SEGMENT_S = 10.0
TWIN_KEEP_HOURS = 6.0


def prune_twin_video(folder: Path, keep_hours: float = TWIN_KEEP_HOURS) -> None:
    """Old segments of the continuous recording are only for session_report to cut from; delete those older than keep_hours (the cut session videos are in their own result folders)."""
    cutoff = time.time() - keep_hours * 3600
    for f in folder.glob("twin_*"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
        except OSError:
            pass


def main() -> None:
    """Window and recording unified with mujoco_haircut (SO-101 haircut):
    the main view follows the tool point and turns automatically to the region being cut (top/back/left/right); F toggles follow/free, and dragging with the mouse pauses it for 5 s;
    a column of two small simulated views at the bottom-left (gripper view + haircut progress), the real camera at the bottom-right;
    recording layout = left column with the two simulated views | main view | camera; recording runs continuously in segments to results/twin_video as soon as the window opens,
    and on each Save session_report cuts out the session time range and stores it in results/pnd_date_time/."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=50555)
    ap.add_argument("--config", default=None, help="mujoco_haircut scene config, default scene_config.json")
    ap.add_argument('--flange-reference',action='store_true')
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--control-port", type=int, default=9003, help="UDP port of the receiver that Space (clutch) is sent to")
    ap.add_argument("--no-follow", action="store_true", help="main view does not follow (F also toggles it)")
    ap.add_argument("--no-record", action="store_true", help="no automatic continuous recording (V still records manually)")
    ap.add_argument("--stream-port", type=int, default=50556, help="port of the local phone video frame service")
    ap.add_argument("--coverage-path", type=Path, default=None, help="path of this run's haircut log; coverage and trajectory are restored only within the same session")
    ap.add_argument('--observer-url',default=None,help='shared real camera, shown at the bottom-right of the desktop window')
    # Old arguments (picture-in-picture tool view/side view), kept for compatibility with old launch commands, no longer used
    for name in ("--inset", "--distance", "--ahead", "--tilt", "--smooth", "--side-distance", "--follow-elevation"):
        ap.add_argument(name, type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--inset-res", type=int, nargs=2, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    from observer_view import ObserverView
    from haircut_sim.coverage import REGIONS
    from haircut_sim.recorder import _fit
    from haircut_sim.viewer import _show_insets
    from haircut_sim.views import CameraDirector, SimViews
    observer=ObserverView(args.observer_url) if args.observer_url else None

    cfg = load_config(args.config)
    if args.flange_reference:flange_config(cfg)
    scene = Scene(cfg)
    scene.forward()
    model, data = scene.model, scene.data
    if args.flange_reference:hide_absent_tool(model)
    alignment_revision=None
    rx = Receiver(args.port)
    offset = tip_offset(cfg)
    coverage_path=args.coverage_path or Path(__file__).parent/'results/coverage'/f'standalone_{time.time_ns()}.npz'
    trail = PersistentTrailOverlay(scene, cfg,coverage_path) if cfg.get("trail", {}).get("enabled", True) else None
    hair = trail.cov.pts if trail is not None and trail.cov is not None else None
    from head_alignment import TOOL_MOUNTED
    tip_name = "Flange" if args.flange_reference and not TOOL_MOUNTED else "Tip"

    view_cfg = cfg.get("camera", {})
    follow = {"on": not args.no_follow}
    director = CameraDirector(trail.cov, REGIONS, cfg) if hair is not None else None
    TWIN_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    prune_twin_video(TWIN_VIDEO_DIR)
    rec = Recorder(model, data, out_dir=TWIN_VIDEO_DIR, left_w=640, right_w=640 if observer else 0,
                   segment_s=0.0 if args.no_record else TWIN_SEGMENT_S, prefix="twin")
    keys: list[int] = []
    period = 1.0 / args.fps
    ctrl = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    state = ""
    stream = None
    stream_renderer = None
    last_stream = 0.0
    views = None
    try:
        stream = TwinFrameServer(args.stream_port)
        stream_renderer = mujoco.Renderer(model, 384, 512)
    except (OSError, RuntimeError) as error:
        print(f"Phone video unavailable: {error}", flush=True)
        if stream is not None:
            stream.close()
            stream = None

    try:
        with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False,
                                          key_callback=keys.append) as v:
            cam = cfg.get("camera", {})
            v.cam.lookat[:] = cam.get("lookat", [0.25, 0.2, 0.2])
            v.cam.distance = float(view_cfg.get("follow_distance_m", 0.9))
            v.cam.azimuth = cam.get("azimuth", 200)
            v.cam.elevation = cam.get("elevation", -25)
            if follow["on"] and director is not None:
                v.cam.azimuth, v.cam.elevation = director.target()
            if trail is not None:
                with v.lock():
                    trail.attach(v.user_scn)
            views = SimViews(model, data, cfg, tip_label=tip_name)
            if not args.no_record:
                rec.start()  # continuous segmented recording: session_report cuts the session video by time
            grip_img = map_img = obs_img = None
            t_views = 0.0
            extra = 0  # number of geoms this program appends to the end of viewer.user_scn (after TrailOverlay)
            while v.is_running():
                t0 = time.perf_counter()
                with v.lock():
                    latest = rx.drain()
                    if latest.get("control", {}).get("command") == "stop_recording":
                        rec.stop()
                    apply(scene, latest)
                    scene.forward()
                    if "status" in latest:
                        state = latest["status"].get("state", "")
                    live = time.perf_counter() - rx.last["piper"] < 0.5
                    v.user_scn.ngeom -= extra
                    extra = 0
                    alignment=latest.get('head_alignment',{}).get('alignment')
                    if args.flange_reference and alignment and alignment['revision']!=alignment_revision:
                        delta=apply_alignment(model,data,alignment)
                        translate_coverage(trail,delta)
                        v.cam.lookat[:]+=delta
                        alignment_revision=alignment['revision']
                    if args.flange_reference and alignment:
                        ctrl.sendto(json.dumps(dict(type='head_alignment_applied',revision=alignment_revision)).encode(),('127.0.0.1',args.control_port))
                    if trail is not None:
                        if ord("C") in keys:
                            trail.clear()
                            trail.save(force=True)
                        trail.update_if_cutting(state,live)
                    if ord("V") in keys:
                        rec.toggle()
                    if ord("F") in keys:
                        follow["on"] = not follow["on"]
                    if 32 in keys:  # Space: clutch pause/resume
                        try:
                            ctrl.sendto(json.dumps({"type": "clutch", "action": "toggle"}).encode(),
                                        ("127.0.0.1", args.control_port))
                        except OSError:
                            pass
                    keys.clear()
                    tip = tip_pos(model, data, offset)
                    region = "back"
                    if follow["on"]:
                        # Same as the SO-101 haircut: the look-at point smoothly follows the tool point, and azimuth/elevation turn with the region being cut
                        if director is not None:
                            region = director.update(v.cam, tip, time.perf_counter())
                        v.cam.lookat[:] = v.cam.lookat + 0.15 * (tip - v.cam.lookat)
                    elif director is not None:
                        region = director.region
                    near, dist_mm = None, None
                    if hair is not None:
                        k = int(np.argmin(np.linalg.norm(hair - tip, axis=1)))
                        near = hair[k]
                        dist_mm = float(np.linalg.norm(near - tip) * 1000.0)
                        extra = add_distance_geoms(v.user_scn, tip, near, dist_color(dist_mm))
                    if time.perf_counter() - t_views >= 1.0 / 30:  # the two small simulated views at 30 Hz
                        t_views = time.perf_counter()
                        grip_img = views.render_gripper(v.user_scn)
                        map_img = views.render_hair_map(v.user_scn, hair.mean(axis=0) if hair is not None else None,
                                                        trail.cov.face_dir if hair is not None else None)
                        obs_img = observer.image(640, 480) if observer else None
                    if rec.active:
                        lines = []
                        if hair is not None:
                            lines = [f"Hair coverage {trail.cov.ratio * 100:.1f}%", trail.region_text()]
                        rec.capture(v.cam, v.user_scn, lines or None, left=[grip_img, map_img], right=[obs_img])
                    if stream is not None and time.perf_counter() - last_stream > 0.125:
                        last_stream = time.perf_counter()
                        stream_renderer.update_scene(data, v.cam)
                        copy_scene_overlays(stream_renderer.scene, v.user_scn)
                        overview = stream_renderer.render().copy()
                        panels=[overview,_fit(grip_img,512,384),_fit(map_img,512,384)]
                        frame=np.concatenate(panels,axis=1)
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        for i,label in enumerate(['OVERVIEW','GRIPPER VIEW (SEE-THROUGH)','HAIR MAP']):
                            x=i*512;y=frame.shape[0]-25
                            cv2.rectangle(frame,(x,y),(x+512,frame.shape[0]),(25,25,25),-1)
                            cv2.putText(frame,label,(x+10,y+18),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
                            if i:cv2.line(frame,(x,0),(x,frame.shape[0]),(130,130,130),2)
                        cv2.putText(frame, state or "WAITING", (12,84), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
                        paint_distance_badge(frame, dist_mm, tip_name == "Flange")
                        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        if ok:
                            stream.publish(jpeg.tobytes())

                if hasattr(v, "set_images"):
                    _show_insets(v, [map_img, grip_img], [obs_img])
                dist = "-" if dist_mm is None else f"{dist_mm:.0f} mm"
                cover = trail.text() if trail is not None else ""
                regions = trail.region_text() if trail is not None else ""
                view_txt = f"follow {tip_name.lower()}, {region} (F = free)" if follow["on"] else f"free (F = follow {tip_name.lower()})"
                v.set_texts([
                    (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                     f"Piper twin\nTeleop\nHair covered\nRegions\n{tip_name} to head\nView\nVideo",
                     f"{'live' if live else 'waiting'}\n{state or '-'}   (space = pause/resume)\n{cover}\n{regions}\n{dist}\n{view_txt}\n"
                     f"{'REC  (V = stop)' if rec.active else 'V = start recording'}"),
                ])
                v.sync()
                time.sleep(max(0.0, period - (time.perf_counter() - t0)))
    finally:
        if observer:observer.close()
        if trail is not None:trail.save(force=True)
        if stream is not None:
            stream.close()
        if stream_renderer is not None:
            stream_renderer.close()
        if views is not None:
            views.close()
        rec.close()


if __name__ == "__main__":
    main()
