import sys,time,csv,tempfile,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from robot_bridge import RobotExperiment
from pnd_tracking import PNDTracker,fk_jacobian
robot=RobotExperiment(root=Path(__file__).parent/'robot_test_results')
data=dict(client_id='robot-test',moving='phone',tracking_mode='board',board={'tag_mm':50},camera_id='test',tag_mm=50,width=640,height=480,calibration={'calibrated':True})
def feed(p):
 p=np.asarray(p,float);robot.feed(dict(camera_t_mm=p.tolist(),camera_R=np.eye(3).tolist(),t_mm=(-p).tolist(),R=np.eye(3).tolist(),reprojection_px=.1,calibrated=True,ambiguous=False),data)
def action(name,**extra):return robot.action(dict(action=name,client_id='robot-test',**extra))
def hold(p,seconds):
 end=time.monotonic()+seconds
 while time.monotonic()<end:feed(p);time.sleep(.03)
try:
 action('connect',backend='mock');assert robot.backend.sent==0 and not robot.backend.enabled
 try:action('enable');raise AssertionError('unsafe enable accepted')
 except ValueError:pass
 action('enable',bench_clear=True);assert robot.backend.sent==0
 hold([0,0,-600],.12);action('sensitivity',step=.7);action('clutch',bench_clear=True);home=robot.feedback['p'].copy()
 hold([-10,-10,-590],2.5);actual=robot.backend.read()['p'];assert np.linalg.norm(actual-home-[7,7,7])<.2,(actual,home,robot.state())
 action('clutch');q=robot.backend.q.copy();hold([80,30,-500],.2);assert np.array_equal(q,robot.backend.q)
 action('clutch',bench_clear=True);hold([80,30,-500],.1);assert np.max(np.abs(q-robot.backend.q))<.01
 hold([78,28,-498],1.2);assert np.linalg.norm(robot.backend.read()['p']-actual-[1.4,1.4,1.4])<.15
 time.sleep(.9);assert robot.paused and not robot.fault
 action('enable',bench_clear=True);hold([78,28,-498],.12);action('clutch',bench_clear=True)
 [hold([x,28,-498],.03) for x in np.linspace(78,-100,30)];assert not robot.paused and not robot.fault,'obsolete 50mm limit is still active'
 action('stop');path=robot.log.path
 with path.open() as f:rows=list(csv.DictReader(f))
 assert len(rows)>50 and all(float(row['qp_iters'])>=1 for row in rows)
 assert any(abs(float(row['qc2'])-float(row['q2']))>1e-5 for row in rows),'command and measured conflated'
 print('MOCK PND PASS: 10:7 XYZ follow; explicit enable interlock; pause/recenter no jump; camera watchdog; no fixed travel envelope; independent command/actual CSV',path,flush=True)
finally:robot.close()
assert robot.plots and robot.plots.is_file(),robot.plot_error
print('FIGURES PASS:',robot.plots,flush=True)
