"""Verify actual rendered video, report outputs and paper figure sizes."""
from pathlib import Path
import sys, time, tempfile
import cv2
import mujoco
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from haircut_sim.scene import Scene, load_config
from haircut_sim.recorder import Recorder
from haircut_sim.figures import make_figures
from haircut_sim.long_tracking_report import make_long_report
from haircut_sim.tracking_report import make_tracking_report
from haircut_sim.report import make_report

out = Path(tempfile.mkdtemp(prefix='haircut_outputs_'))
scene = Scene(load_config())
rec = Recorder(scene.model, scene.data, width=640, height=480, fps=10, out_dir=out)
cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)
cam.lookat[:] = [.25,.2,.2]
cam.distance, cam.azimuth, cam.elevation = 1.5, 200, -25
rec.start()
video = rec.path
for _ in range(4):
    rec.capture(cam, lines=['MuJoCo twin recording test'])
    time.sleep(.1)
rec.close()
cap = cv2.VideoCapture(str(video))
assert cap.isOpened()
ok, frame = cap.read()
assert ok and frame.shape == (480,640,3)
assert cap.get(cv2.CAP_PROP_FRAME_COUNT) >= 4
cap.release()
print('PASS video encode/decode:', video)

csv_path = Path(sys.argv[1])
folder = make_figures(csv_path)
make_long_report(csv_path)
make_tracking_report(csv_path, figures_ready=True)
make_report(csv_path, open_browser=False)
names = ['tracking_xyz','tracking_orientation','tracking_error','solver_performance',
         'joint_tracking_1_3','joint_tracking_4_6','coverage_views','coverage_ratio']
for name in names:
    assert (folder/f'{name}.pdf').stat().st_size > 1000
    with Image.open(folder/f'{name}.png') as im:
        assert im.width == (4296 if name == 'coverage_views' else 2100)
        assert im.height < 2600
        assert abs(im.info['dpi'][0]-600) < 1
print('PASS eight PDF/PNG figure pairs, 600 dpi and exact column widths:', folder)
