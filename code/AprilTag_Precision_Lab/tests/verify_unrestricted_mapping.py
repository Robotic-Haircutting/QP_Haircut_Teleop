"""1:1 phone targets beyond the old envelope, retaining joint/rate limits."""
import sys,time,tempfile
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from robot_bridge import RobotExperiment
r=RobotExperiment(root=Path(tempfile.mkdtemp())/'results')
d=dict(client_id='mapping',moving='phone',tracking_mode='board',board={'tag_mm':50},camera_id='test',tag_mm=50,width=640,height=480,calibration={'calibrated':True})
def action(a):return r.action(dict(action=a,client_id='mapping',bench_clear=True,backend='mock'))
def feed(p,angle=0):
 matrix=Rotation.from_euler('z',angle,degrees=True).as_matrix();p=np.asarray(p,float)
 r.feed(dict(camera_t_mm=p.tolist(),camera_R=matrix.tolist(),t_mm=(-matrix@p).tolist(),R=matrix.T.tolist(),reprojection_px=.1,calibrated=True,ambiguous=False),d)
def hold(p,angle=0,duration=.15):
 end=time.monotonic()+duration
 while time.monotonic()<end:
  feed(p,angle);time.sleep(.025)
  if r.feedback:
   q=r.feedback['q'];limits=np.asarray(r.tracker.solver.p.joint_limits_deg)
   assert np.all(q>=limits[:,0]-1e-8) and np.all(q<=limits[:,1]+1e-8)
try:
 action('connect');action('enable');assert r.scale==3.
 r.action(dict(action='sensitivity',step=1.,client_id='mapping'))
 hold([0,0,-600]);action('clutch');q=r.backend.q.copy();home=r.home.copy()
 hold([0,0,-540],angle=25,duration=.5)
 assert not r.paused and not r.fault,r.reason
 assert np.linalg.norm(r.filter.p-home)>0,'target did not move'
 assert r.backend.sent>0 and np.max(np.abs(q-r.backend.q))>.01
 # Raw requested target is a full 60mm; speed-limited command remains slower.
 expected=home+r.axis@np.array([0.,0.,60.])
 target=r.home+r.axis@(r.latest[0]-r.ref)*r.scale
 assert np.allclose(target,expected) and abs(np.linalg.norm(target-home)-60)<1e-8
 assert np.rad2deg(Rotation.from_matrix((r.axis@(r.latest[1]@r.ref_r.T)@r.axis.T@r.rotation)@r.rotation.T).magnitude())>24.9
 assert np.linalg.norm(r.command-home)<=60*.65,'cartesian speed cap removed'
 path=r.log.path;action('clutch');held=r.backend.q.copy()
 hold([150,0,-600]);action('clutch');hold([150,0,-600])
 assert not r.paused and np.max(np.abs(held-r.backend.q))<.02,'clutch reanchor jumped'
 assert r.log.path==path and r.tracker.max_cartesian_speed==60
 action('clutch')
 r.action(dict(action='sensitivity',step=.7,client_id='mapping'))
 assert r.scale==.7
 r.action(dict(action='sensitivity',step=20.,client_id='mapping'))
 assert r.scale==20.
 for invalid in (20.01,float('nan'),.14):
  try:r.action(dict(action='sensitivity',step=invalid,client_id='mapping'));raise AssertionError('invalid gain accepted')
  except ValueError:pass
 assert r.scale==20.
 hold([150,0,-600]);action('clutch');hold([150,0,-590],duration=.3)
 assert not r.paused and not r.fault
 raw=r.home+r.axis@(r.latest[0]-r.ref)*r.scale
 assert abs(np.linalg.norm(raw-r.home)-200)<1e-6,'10:200 mapping incorrect'
 action('clutch')
 # Driver faults remain faults even though the artificial travel envelope is removed.
 hold([150,0,-600]);action('clutch')
 def driver_fault():raise RuntimeError('Driver fault test')
 r.backend.healthy=driver_fault;hold([150,0,-600])
 assert r.fault and r.paused
 print('UNRESTRICTED MAPPING PASS: default10:30; 10:200 accepted and maps 10mm to200mm; 60mm/25deg target accepted; actual moves; joint and speed caps preserved; clutch retains log and pose; adjustable gain; driver faults still stop',flush=True)
finally:r.close()
