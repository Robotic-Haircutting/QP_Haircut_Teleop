"""Joint-angle channel from the teleoperation process to the simulation window process (local UDP loopback, fire-and-forget: never blocks, never raises).

Packet format (JSON):
  {"s": "piper", "q": [J1..J6 deg], "g": gripper opening in mm}
  {"s": "so101", "q": [pan, lift, elbow, wrist_flex, wrist_roll deg], "g": gripper.pos 0-100}
"""

import json
import socket
import subprocess
import sys
from pathlib import Path

DEFAULT_PORT = 50555


class SimPublisher:
    def __init__(self, port: int = DEFAULT_PORT, host: str = "127.0.0.1"):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)

    def _send(self, msg: dict) -> None:
        try:
            self.sock.sendto(json.dumps(msg, separators=(",", ":")).encode(), self.addr)
        except OSError:
            pass  # simulation window not open or already closed: drop the packet, teleoperation is unaffected

    def send_piper(self, joints_deg, gripper_mm=None) -> None:
        self._send({"s": "piper", "q": [round(float(v), 3) for v in joints_deg], "g": None if gripper_mm is None else float(gripper_mm)})

    def send_so101(self, joints_deg, gripper_pos=None) -> None:
        self._send({"s": "so101", "q": [round(float(v), 3) for v in joints_deg], "g": None if gripper_pos is None else float(gripper_pos)})

    def stop_recording(self) -> None:
        self._send({"s": "control", "command": "stop_recording"})


def spawn_viewer(config: str | None = None, port: int = DEFAULT_PORT, extra: list[str] | None = None) -> subprocess.Popen:
    """Start a separate Python process for the MuJoCo window so rendering does not use the teleoperation process's CPU/GIL."""
    root = Path(__file__).resolve().parent.parent
    cmd = [sys.executable, "-m", "haircut_sim.viewer", "--port", str(port)]
    if config:
        cmd += ["--config", config]
    cmd += extra or []
    # On Windows, detach from the teleoperation console: pressing Ctrl+C or closing that console does not kill the simulation window or the recording;
    # the recording is finalized and saved only when the MuJoCo window is closed
    flags = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS) if sys.platform == "win32" else 0
    return subprocess.Popen(cmd, cwd=str(root), creationflags=flags)
