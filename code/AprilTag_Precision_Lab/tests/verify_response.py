import sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sim_bridge import Simulation,WORKING_JOINTS
from reachable_step import bounded_step
from piper_twin import fk_jacobian
q=WORKING_JOINTS.copy();t,_=fk_jacobian(np.deg2rad(q))
_,_,_,info=bounded_step(t[:3,3]*1000+[0,1,0],t[:3,:3],q,.0001)
assert info['limited'] and info['cause']=='joint_speed',info
sim=Simulation()
data=dict(client_id='response-test',moving='phone',camera_id='test',tag_id=0,tag_mm=80,width=640,height=480,calibration=dict(width=640,height=480,fx=800,fy=800,cx=320,cy=240,calibrated=True))
def feed(p):
    p=np.asarray(p,float);sim.update(dict(t_mm=(-p).tolist(),R=np.eye(3).tolist(),camera_t_mm=p.tolist(),camera_R=np.eye(3).tolist(),reprojection_px=.1,ambiguous=False),data)
def hold(p,seconds):
    end=time.monotonic()+seconds
    while time.monotonic()<end:feed(p);time.sleep(.03)
try:
    assert sim.scale==.35
    sim.start('response-test');feed([0,0,-550]);sim.resume();home=np.array(sim.state()['position_mm']);ref=sim.ref.copy()
    hold([0,0,-510],2.5)
    expected=home+np.array([14,0,0]);actual=np.array(sim.state()['position_mm'])
    assert np.linalg.norm(actual-expected)<.15,(actual,expected)
    assert np.max(np.abs(sim.ref-ref))<1e-9,'rate limit discarded phone target'
    hold([-20,0,-510],1.2)
    left=np.array(sim.state()['position_mm']);assert np.linalg.norm(left-(expected+[0,7,0]))<.1,left
    hold([20,0,-510],1.2)
    right=np.array(sim.state()['position_mm']);assert np.linalg.norm(right-(expected+[0,-7,0]))<.1,right
    print('PHONE LEFT/RIGHT PASS: phone -X/left -> robot +Y 7mm; phone +X/right -> robot -Y 7mm',flush=True)
    hold([0,0,-509.5],.8);actual=np.array(sim.state()['position_mm'])
    assert np.linalg.norm(actual-(expected+[.175,0,0]))<.05,actual
    before=sim.twin.q.copy();sim.action(dict(client_id='response-test',action='sensitivity',step=.5))
    assert np.max(np.abs(before-sim.twin.q))<1e-9
    hold([0,0,-507.5],.8);after=np.array(sim.state()['position_mm'])
    assert np.linalg.norm(after-actual-[1,0,0])<.05,(after,actual)
    before=sim.twin.q.copy();sim.action(dict(client_id='response-test',action='sensitivity',step=.7))
    assert np.max(np.abs(before-sim.twin.q))<1e-9
    hold([-10,-10,-497.5],1.5);high=np.array(sim.state()['position_mm'])
    assert np.linalg.norm(high-after-[7,7,7])<.15,(high,after)
    print('10:7 MUJOCO PASS: each 10mm phone translation -> 7mm robot on forward/left/up; gain change no jump',flush=True)
    before=sim.twin.q.copy();sim.action(dict(client_id='response-test',action='sensitivity',step=1.))
    assert np.max(np.abs(before-sim.twin.q))<1e-9
    hold([-12,-12,-495.5],1.);full=np.array(sim.state()['position_mm'])
    assert np.linalg.norm(full-high-[2,2,2])<.15,(full,high)
    try:sim.action(dict(client_id='response-test',action='sensitivity',step=1.05));raise AssertionError('out of range accepted')
    except ValueError:pass
    print('10:10 MUJOCO PASS: 1:1 translation on three axes, no jump, range checked',flush=True)
    print('RESPONSE PASS: speed limit classified separately; 40mm phone -> 14mm robot; 0.5mm input responds; live gain change has no jump',flush=True)
finally:sim.close()
