import sys,time,tempfile,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from robot_bridge import RobotExperiment
r=RobotExperiment(root=Path(tempfile.mkdtemp())/'results')
d=dict(client_id='resume',moving='phone',tracking_mode='board',board={'tag_mm':50},camera_id='test',tag_mm=50,width=640,height=480,calibration={'calibrated':True})
def feed(p,ambiguous=True):
 p=np.asarray(p,float)
 r.feed(dict(t_mm=(-p).tolist(),R=np.eye(3).tolist(),reprojection_px=.1,calibrated=True,ambiguous=ambiguous),d)
def action(a):return r.action(dict(action=a,client_id='resume',bench_clear=True,backend='mock'))
try:
 action('connect');action('enable');q=r.backend.q.copy()
 feed([0,0,-600]);action('clutch');assert not r.resume_pending and r.paused
 # Paused phone relocation must replace the old pose, including a 350mm shift.
 for _ in range(3):feed([350,0,-600]);time.sleep(.025)
 assert r.paused and not r.resume_pending,'stable frames auto-resumed the arm'
 action('clutch');assert not r.paused
 assert np.linalg.norm(r.ref-[350,0,-600])<1e-9
 assert np.max(np.abs(r.backend.q-q))<.01,'reference acquisition moved the arm'
 # Recorded phone intervals reach 450ms; they must not trigger the old 300ms timeout.
 for _ in range(4):feed([350,0,-600]);time.sleep(.45);assert not r.paused,r.reason
 time.sleep(.9);assert r.paused and not r.fault
 action('clutch');assert not r.resume_pending and r.paused
 for _ in range(3):feed([350,0,-600]);time.sleep(.025)
 assert r.paused,'camera recovery auto-resumed'
 action('clutch');assert not r.paused,'camera recovery required unnecessary motor re-enable'
 action('clutch');feed([-350,0,-600]);assert np.linalg.norm(r.latest[0]-[-350,0,-600])<1e-9
 action('clutch');assert not r.resume_pending and r.paused
 action('clutch');assert not r.resume_pending and r.paused
 held=r.backend.q.copy()
 for _ in range(4):feed([-350,0,-600]);time.sleep(.03)
 assert r.paused and np.array_equal(held,r.backend.q),'cancelled Resume moved arm'
 print('RESUME REACQUISITION PASS: paused 350mm relocation, 3-frame explicit Resume, no target jump, 450ms packets tolerated, no queued or automatic Resume',flush=True)
finally:r.close()
