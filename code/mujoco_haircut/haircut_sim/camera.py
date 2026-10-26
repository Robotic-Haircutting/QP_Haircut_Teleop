"""Real camera: a background thread keeps reading the latest frame for the small picture in the bottom-right of the MuJoCo window and for the composited recording.

  python -m haircut_sim.camera          # preview window only, for adjusting the angle; Q / Esc or close the window to quit (uses real_camera.index in scene_config.json)
  python -m haircut_sim.camera 1        # choose the camera index

Change the camera index under real_camera in scene_config.json (0 is usually the laptop's built-in camera). A camera can only be opened by one program at a time,
so close the preview window before running run_haircut.bat.
"""

import sys
import threading
import time

import cv2


class CameraStream:
    def __init__(self, index: int, width: int = 640, height: int = 480):
        self.index = index
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {index} (wrong index, or it is in use by another program)")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.frame = None  # BGR
        self.stamp = 0.0
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="camera")
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            ok, f = self.cap.read()
            if ok:
                with self._lock:
                    self.frame, self.stamp = f, time.perf_counter()
            else:
                time.sleep(0.01)

    def latest(self):
        """Latest frame (BGR), or None if nothing has been read yet."""
        with self._lock:
            return self.frame

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.cap.release()


def main() -> None:
    """Standalone preview window (Tk; the opencv in the lerobot environment is headless and has no imshow)."""
    import tkinter as tk

    from PIL import Image, ImageTk

    from .scene import load_config

    rc = load_config().get("real_camera", {})
    index = int(sys.argv[1]) if len(sys.argv) > 1 else int(rc.get("index", 1))
    cam = CameraStream(index, int(rc.get("width", 640)), int(rc.get("height", 480)))
    print(f"Previewing camera {index}; press Q / Esc or close the window to quit", flush=True)
    root = tk.Tk()
    root.title(f"Camera {index} preview (Q = close)")
    label = tk.Label(root)
    label.pack()

    def tick():
        f = cam.latest()
        if f is not None:
            label.img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
            label.configure(image=label.img)
        root.after(30, tick)

    for key in ("q", "Q", "<Escape>"):
        root.bind(key, lambda e: root.destroy())
    tick()
    try:
        root.mainloop()
    finally:
        cam.close()


if __name__ == "__main__":
    main()
