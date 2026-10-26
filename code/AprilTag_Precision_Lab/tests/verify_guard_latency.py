import sys,time,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sim_bridge import Simulation,PROJECTS
from head_guard import HeadGuard
from piper_twin import fk_jacobian
from reachable_step import solve_tracking_ik
q=np.array(json.loads(Path(__file__).with_name('guard_boundary_pose.json').read_text())['joints_deg']);g=HeadGuard(PROJECTS/'mujoco_haircut')
assert 10<=g.clearance(q)['distance_mm']<10.1
rng=np.random.default_rng(1);durations=[]
for _ in range(25):
    start=time.monotonic();candidate,info=g.filter_step(q,q+rng.uniform(-.15,.15,6));durations.append(time.monotonic()-start)
    assert g.clearance(candidate)['distance_mm']>=10-1e-6
assert max(durations)<.12,max(durations)
t,_=fk_jacobian(np.deg2rad(q));choices=[]
for axis in np.eye(3):
 for sign in (-1,1):
    target=t[:3,3]+axis*sign*.0035
    nextq,_,_=solve_tracking_ik(target,t[:3,:3],q)
    choices.append((g.clearance(nextq)['distance_mm'],axis*sign,nextq))
distance,direction,candidate=max(choices,key=lambda x:x[0]);accepted,info=g.filter_step(q,candidate)
assert np.max(np.abs(accepted-q))>1e-8 and g.clearance(accepted)['distance_mm']>g.clearance(q)['distance_mm'],(direction,info)
print('RECORDED FREEZE PASS: actual 10.0003mm pose, max check',round(max(durations)*1000,2),'ms; certified retreat moves without a protection button',flush=True)
sim=Simulation();sim.recovery_q=q.copy();data=dict(client_id='freeze-test',moving='phone',camera_id='test',tag_id=0,tag_mm=80,width=640,height=480,calibration=dict(width=640,height=480,fx=800,fy=800,cx=320,cy=240,calibrated=True))
def feed(p):
 p=np.asarray(p,float);sim.update(dict(t_mm=(-p).tolist(),R=np.eye(3).tolist(),camera_t_mm=p.tolist(),camera_R=np.eye(3).tolist(),reprojection_px=.1,ambiguous=False),data)
try:
 sim.start('freeze-test');feed([0,0,-550]);sim.resume()
 from phone_pose import CAMERA_TO_BASE
 p=np.array([0,0,-550.])+CAMERA_TO_BASE.T@direction*20
 end=time.monotonic()+8
 while time.monotonic()<end:feed(p);time.sleep(.03)
 assert np.max(np.abs(sim.twin.q-q))>.05
 assert sim.head_safety['distance_mm']>10.5,sim.state()
 sim.pause();stamp=time.monotonic();s=sim.state();assert time.monotonic()-stamp<.1 and s['paused']
 print('REAL MUJOCO FREEZE RECOVERY PASS: recorded boundary pose retreats with phone input, state and clutch remain responsive',flush=True)
finally:sim.close()
