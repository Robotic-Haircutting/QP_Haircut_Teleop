import sys,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sim_bridge import Simulation

sim=Simulation()
data=dict(client_id='continuous-test',moving='phone',camera_id='test-camera',tag_id=0,tag_mm=80,width=640,height=480,calibration=dict(width=640,height=480,fx=800,fy=800,cx=320,cy=240,calibrated=True))
def feed(p,r=np.eye(3),error=.1):
    p=np.asarray(p,float);tag_r=r.T;tag_t=-tag_r@p
    sim.update(dict(t_mm=tag_t.tolist(),R=tag_r.tolist(),camera_t_mm=p.tolist(),camera_R=r.tolist(),reprojection_px=error,ambiguous=False),data)
def hold(p,r=np.eye(3),seconds=.3,error=.1):
    end=time.monotonic()+seconds
    while time.monotonic()<end:feed(p,r,error);time.sleep(.03)
try:
    sim.start('continuous-test');feed([0,0,-550]);sim.resume()
    # Both formerly triggered automatic clutch pauses.
    feed([70,0,-550]);assert not sim.paused
    r=Rotation.from_euler('x',30,degrees=True).as_matrix()
    feed([70,0,-550],r);assert not sim.paused
    hold([70,0,-550],r);assert not sim.paused
    print('JUMP GUARDS REMOVED PASS: 70mm detected position change and 30deg orientation change do not pause',flush=True)
    feed([70,0,-550],r,error=4);saved=sim.twin.q.copy();time.sleep(.2)
    assert not sim.paused and sim.tracking_hold
    assert np.max(np.abs(saved-sim.twin.q))<1e-9
    axes=sim.phone_axis.copy();feed([75,0,-545],r)
    assert not sim.paused and not sim.tracking_hold
    assert np.max(np.abs(saved-sim.twin.q))<1e-9
    assert np.max(np.abs(axes-sim.phone_axis))<1e-9
    hold([80,0,-545],r);assert np.max(np.abs(saved-sim.twin.q))>.01
    print('QUALITY HOLD PASS: bad input holds position without pause, valid input anchors without jump and continues',flush=True)
    time.sleep(.7);saved=sim.twin.q.copy()
    assert not sim.paused and sim.tracking_hold
    feed([90,0,-530],r);assert not sim.paused and not sim.tracking_hold
    assert np.max(np.abs(saved-sim.twin.q))<1e-9
    data['calibration']['fx']=810;saved=sim.twin.q.copy();feed([90,0,-530],r)
    assert not sim.paused and np.max(np.abs(saved-sim.twin.q))<1e-9
    print('LOSS/CONFIG PASS: no automatic clutch pause, recovery/new intrinsics anchor current FK',flush=True)
    sim.pause();hold([100,0,-520],r);assert sim.paused
    print('MANUAL PAUSE PASS: detections never override explicit pause',flush=True)
finally:sim.close()
