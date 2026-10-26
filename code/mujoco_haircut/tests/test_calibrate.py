"""Offline self-test: head calibration recovers the head position from noisy touch points (currently only the top of the head) and writes it back to the config correctly.  python tests/test_calibrate.py"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from haircut_sim.calibrate import POINTS, Calibrator  # noqa: E402
from haircut_sim.scene import Scene, load_config  # noqa: E402


def test_calibrate_roundtrip(fit_yaw: bool = False):
    tmp = Path(tempfile.mkdtemp()) / "scene_config.json"
    shutil.copyfile(ROOT / "scene_config.json", tmp)
    cfg = load_config(tmp)
    cfg["head"]["file"] = str((ROOT / cfg["head"]["file"]).resolve())  # so the mesh is found from the temp directory too
    raw = json.loads(tmp.read_text(encoding="utf-8"))
    raw["head"]["file"] = cfg["head"]["file"]
    tmp.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    cfg["calibration"] = {"fit_yaw": fit_yaw}
    scene = Scene(cfg)
    scene.forward()
    cal = Calibrator(scene, cfg, tmp)

    # Assumed real head: 3 cm forward, 2 cm left, 1 cm higher and rotated another 12° relative to the current position
    true_dyaw = np.deg2rad(12.0 if fit_yaw else 0.0)
    Rz = np.array([[np.cos(true_dyaw), -np.sin(true_dyaw), 0], [np.sin(true_dyaw), np.cos(true_dyaw), 0], [0, 0, 1]])
    R0 = cal._R_orig()
    t_true = cal.orig_pos + np.array([0.03, 0.02, 0.01])
    rng = np.random.default_rng(1)
    for k, _ in POINTS:
        p = Rz @ R0 @ cal.local[k] + t_true + rng.normal(0, 0.002, 3)  # 2 mm touch error
        cal.tip = lambda p=p: p  # stands in for the real clipper tip
        cal.record()
    R_new, t, res = cal.result
    yaw_err = np.degrees(np.arctan2(R_new[1, 0], R_new[0, 0]) - np.arctan2((Rz @ R0)[1, 0], (Rz @ R0)[0, 0]))
    print(f"position error {np.linalg.norm(t - t_true) * 1000:.1f} mm, heading error {yaw_err:.2f}°, per-point residuals {np.round(res * 1000, 1)} mm")
    assert np.linalg.norm(t - t_true) < 0.005 and abs(yaw_err) < 2.0

    scn = mujoco.MjvScene(scene.model, maxgeom=100)
    cal.draw(scn)
    assert scn.ngeom == 1 + 2 * len(POINTS)
    cal.save()
    saved = load_config(tmp)
    s2 = Scene(saved)
    s2.forward()
    bid = s2.model.body("head").id
    assert np.allclose(s2.data.xpos[bid], scene.data.xpos[bid], atol=1e-3)
    assert np.allclose(s2.data.xmat[bid], scene.data.xmat[bid], atol=1e-3)
    print("Written back to config:", saved["head"]["pos"], saved["head"]["euler_deg"])


if __name__ == "__main__":
    test_calibrate_roundtrip(fit_yaw=False)
    if len(POINTS) >= 3:
        test_calibrate_roundtrip(fit_yaw=True)
    print("All passed")
