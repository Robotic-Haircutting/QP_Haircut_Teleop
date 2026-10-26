"""Record the current view of the MuJoCo window to mp4 (H.264, plays directly in PowerPoint and on video sites).

Press V in the simulation window to start / stop recording; with --sim.record, teleop_sim starts recording as soon as the window opens.
It records the window's current view (wherever the mouse has turned it), including the trajectory, coverage points and coverage text.
Frames are duplicated/skipped by wall-clock time so the video length matches real time. Layout (passed in by viewer.py): left column (gripper view and haircut progress view, stacked),
MuJoCo main view in the middle, right column (real camera), all in one video. Files are saved as video_date_time.mp4 in mujoco_haircut/results (in this run's subfolder during teleoperation).
"""

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def _fit(img: np.ndarray | None, w: int, h: int) -> np.ndarray:
    """Fit an RGB image, keeping its aspect ratio, into a w×h cell with a black background; None draws an empty cell."""
    out = np.zeros((h, w, 3), dtype=np.uint8)
    if img is None:
        cv2.putText(out, "no image", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        return out
    s = min(w / img.shape[1], h / img.shape[0])
    nw, nh = max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))
    x, y = (w - nw) // 2, (h - nh) // 2
    out[y:y + nh, x:x + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return out


def _ffmpeg() -> str | None:
    # When python.exe is started without activating the conda environment, ffmpeg is not on PATH; look for the one bundled with the environment
    found = shutil.which("ffmpeg")
    if found:
        return found
    for cand in (Path(sys.prefix) / "Library" / "bin" / "ffmpeg.exe", Path(sys.prefix) / "bin" / "ffmpeg"):
        if cand.is_file():
            return str(cand)
    return None


class Recorder:
    def __init__(self, model, data, width=1280, height=720, fps=30, out_dir: Path = RESULTS, left_w: int = 0, right_w: int = 0,
                 segment_s: float = 0.0, prefix: str = "video"):
        self.model, self.data = model, data
        self.w, self.h, self.fps = width, height, fps
        # One column on each side of the main view (0 = none); images in a column split its height evenly; widths are even (required by H.264)
        self.left_w, self.right_w = int(left_w) // 2 * 2, int(right_w) // 2 * 2
        self.out_w = width + self.left_w + self.right_w
        # segment_s > 0: record continuously as mp4 segments of segment_s seconds (file names carry the start time; each segment is usable once written),
        # so other programs can cut out a time range (AprilTag_Precision_Lab cuts per experiment session). Requires ffmpeg
        self.segment_s, self.prefix = float(segment_s), prefix
        self.out_dir = Path(out_dir)
        self.renderer = None
        self.proc = None
        self.writer = None
        self.path = None
        self.t0 = 0.0
        self.frames = 0

    @property
    def active(self) -> bool:
        return self.path is not None

    def start(self) -> None:
        if self.active:
            return
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, self.h, self.w)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.out_dir / f"{self.prefix}_{datetime.now():%Y%m%d_%H%M%S}.mp4"
        ff = _ffmpeg()
        if ff and self.segment_s > 0:
            # Frame k corresponds to wall clock start_wall + k/fps (capture fills frames by wall clock); the segment list gives each segment's start/end seconds in the video;
            # together they allow cutting any range by wall-clock time
            stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
            self.path = self.out_dir / f"{self.prefix}_{stamp}_%04d.mp4"
            seg_list = self.out_dir / f"{self.prefix}_{stamp}_segments.csv"
            self._seg_meta = (self.out_dir / f"{self.prefix}_{stamp}.json",
                              {"fps": self.fps, "segment_list": seg_list.name, "width": self.out_w, "height": self.h})
            cmd = [ff, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{self.out_w}x{self.h}", "-r", str(self.fps),
                   "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                   "-force_key_frames", f"expr:gte(t,n_forced*{self.segment_s})", "-f", "segment", "-segment_time", str(self.segment_s),
                   "-reset_timestamps", "1", "-segment_list", str(seg_list), "-segment_list_type", "csv", "-segment_format", "mp4", str(self.path)]
            flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, creationflags=flags)
        elif ff:
            cmd = [ff, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{self.out_w}x{self.h}", "-r", str(self.fps),
                   "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(self.path)]
            flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0  # the window process has no console; do not pop up a console for ffmpeg
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, creationflags=flags)
        else:
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (self.out_w, self.h))
        if self.writer is not None and not self.writer.isOpened():
            self.writer.release()
            self.writer, self.path = None, None
            raise RuntimeError("Video encoder failed to open")
        self.t0, self.frames = time.perf_counter(), 0
        if getattr(self, "_seg_meta", None):
            meta_path, meta = self._seg_meta
            meta_path.write_text(json.dumps({"start_wall": time.time(), **meta}), encoding="utf-8")
            self._seg_meta = None
        print(f"Recording started: {self.path}", flush=True)

    def stop(self) -> None:
        if not self.active:
            return
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait(timeout=30)
            self.proc = None
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        print(f"Recording saved: {self.path} ({self.frames / self.fps:.1f} s)", flush=True)
        self.path = None

    def toggle(self) -> None:
        self.stop() if self.active else self.start()

    def _column(self, cells: list, w: int) -> np.ndarray:
        cells = list(cells) or [None]
        hs = [self.h // len(cells)] * len(cells)
        hs[-1] += self.h - sum(hs)
        return np.vstack([_fit(c, w, ch) for c, ch in zip(cells, hs)])

    def capture(self, cam, user_scn=None, lines: list[str] | None = None, left: list | None = None, right: list | None = None) -> None:
        """Render a frame when needed: work out from wall-clock time which frame is due, and repeat the same frame if behind."""
        if not self.active:
            return
        due = int((time.perf_counter() - self.t0) * self.fps) + 1
        if due <= self.frames:
            return
        self.renderer.update_scene(self.data, cam)
        s = self.renderer.scene
        if user_scn is not None:
            for i in range(user_scn.ngeom):
                if s.ngeom >= s.maxgeom:
                    break
                src, g = user_scn.geoms[i], s.geoms[s.ngeom]
                mujoco.mjv_initGeom(g, src.type, src.size, src.pos, src.mat.flatten(), src.rgba)
                g.size[:] = src.size
                s.ngeom += 1
        img = np.ascontiguousarray(self.renderer.render())
        if lines:
            for k, txt in enumerate(lines):
                org = (20, 40 + 34 * k)
                cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
                cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        parts = []
        if self.left_w:
            parts.append(self._column(left or [], self.left_w))
        parts.append(img)
        if self.right_w:
            parts.append(self._column(right or [], self.right_w))
        img = np.ascontiguousarray(np.hstack(parts))
        buf = img.tobytes() if self.proc is not None else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        while self.frames < due:
            if self.proc is not None:
                self.proc.stdin.write(buf)
            else:
                self.writer.write(buf)
            self.frames += 1

    def close(self) -> None:
        self.stop()
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
