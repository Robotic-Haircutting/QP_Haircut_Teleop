"""Offline self-test: the scene compiles, the Piper model matches the controller kinematics, and the UDP sync channel works.  python tests/test_scene.py"""

import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from haircut_sim.publisher import SimPublisher  # noqa: E402
from haircut_sim.scene import Scene, load_config  # noqa: E402

# Modified DH of piper_sdk C_PiperForwardKinematics(dh_is_offset=1), same as piper_teleop/piper_kinematics.py
_A = np.array([0.0, 0.0, 285.03, -21.98, 0.0, 0.0]) / 1000.0
_ALPHA = np.array([0.0, -np.pi / 2, 0.0, np.pi / 2, -np.pi / 2, np.pi / 2])
_THETA0 = np.array([0.0, -np.deg2rad(172.22), -np.deg2rad(102.78), 0.0, 0.0, 0.0])
_D = np.array([123.0, 0.0, 0.0, 250.75, 0.0, 91.0]) / 1000.0


def piper_fk(q):
    T = np.eye(4)
    for i in range(6):
        a, al, th, d = _A[i], _ALPHA[i], q[i] + _THETA0[i], _D[i]
        ca, sa, ct, st = np.cos(al), np.sin(al), np.cos(th), np.sin(th)
        T = T @ np.array([[ct, -st, 0, a], [st * ca, ct * ca, -sa, -sa * d], [st * sa, ct * sa, ca, ca * d], [0, 0, 0, 1]])
    return T


def test_alignment():
    s = Scene(load_config())
    b = s.model.body("piper/link6").id
    rng = np.random.default_rng(0)
    lim = np.array([[-150, 150], [0, 180], [-170, 0], [-100, 100], [-70, 70], [-100, 100]])
    pe, re = [], []
    for _ in range(300):
        q = rng.uniform(lim[:, 0], lim[:, 1])
        s.set_piper(q, 30)
        s.forward()
        T = piper_fk(np.deg2rad(q))
        pe.append(np.linalg.norm(s.data.xpos[b] - T[:3, 3]))
        R = T[:3, :3].T @ s.data.xmat[b].reshape(3, 3)
        re.append(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    print(f"Piper flange error: position max {max(pe) * 1e3:.2f} mm, orientation max {max(re):.3f}°")
    assert max(pe) < 0.003 and max(re) < 0.1


def test_udp_sync():
    port = 50599
    proc = subprocess.Popen([sys.executable, "-m", "haircut_sim.viewer", "--port", str(port), "--headless", "2.5"],
                            cwd=str(ROOT), stdout=subprocess.PIPE, text=True)
    time.sleep(1.0)
    pub = SimPublisher(port)
    piper_q = [10, 60, -70, 5, 20, -15]
    so101_q = [12, -20, 30, 40, -50]
    for _ in range(30):
        pub.send_piper(piper_q, 40)
        pub.send_so101(so101_q, 60)
        time.sleep(1 / 60)
    out = proc.communicate(timeout=20)[0].strip().splitlines()[-1]
    print("Received by the simulation process:", out)
    import json

    r = json.loads(out)
    assert r["count"]["piper"] > 10 and r["count"]["so101"] > 10
    assert np.allclose(r["piper_deg"], piper_q, atol=0.01) and np.allclose(r["so101_deg"], so101_q, atol=0.01)


if __name__ == "__main__":
    test_alignment()
    test_udp_sync()
    print("All passed")
