"""Compare old/new response under identical phone input; Mock only."""
import sys,time,tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from robot_bridge import RobotExperiment
robots=[RobotExperiment(root=Path(tempfile.mkdtemp())/'results') for _ in range(2)]
data=dict(client_id='response',moving='phone',tracking_mode='board',board={'tag_mm':50},camera_id='test',tag_mm=50,width=640,height=480,calibration={'calibrated':True})
def feed(r,p):
 p=np.asarray(p,float);r.feed(dict(camera_t_mm=p.tolist(),camera_R=np.eye(3).tolist(),t_mm=(-p).tolist(),R=np.eye(3).tolist(),reprojection_px=.1,calibrated=True,ambiguous=False),data)
def action(r,a):return r.action(dict(action=a,client_id='response',bench_clear=True,backend='mock'))
try:
 old,new=robots;old.follow_speed=20.;old.follow_joint_speed=15.;old.follow_orientation_speed=10.
 old.tracker.max_cartesian_speed=20;old.tracker.max_orientation_speed=10;old.tracker.solver.p.max_joint_speed_deg_s=15
 homes=[]
 for r in robots:
  action(r,'connect');action(r,'enable');assert r.scale==3
  for _ in range(4):feed(r,[0,0,-600]);time.sleep(.025)
  action(r,'clutch');homes.append(r.backend.read()['p'].copy())
 until=time.monotonic()+.8
 while time.monotonic()<until:
  for r in robots:feed(r,[0,0,-580])
  time.sleep(.025)
 distances=[float(np.linalg.norm(r.backend.read()['p']-home)) for r,home in zip(robots,homes)]
 assert distances[1]>distances[0]*1.8 and distances[1]>20,distances
 assert all(not r.paused and not r.fault for r in robots)
 action(new,'clutch');q=new.backend.q.copy()
 for _ in range(4):feed(new,[100,100,-500]);time.sleep(.025)
 action(new,'clutch')
 for _ in range(4):feed(new,[100,100,-500]);time.sleep(.025)
 assert np.max(np.abs(q-new.backend.q))<.02 and not new.paused
 assert new.state()['following_limits']==dict(position_mm_s=60.,joint_deg_s=30.,orientation_deg_s=20.)
 print('FOLLOW RESPONSE PASS: old/new actual displacement after0.8s (mm)',distances,'; faster response, default10:30, clutch no jump, recorded limits',flush=True)
finally:
 for r in robots:r.close()
