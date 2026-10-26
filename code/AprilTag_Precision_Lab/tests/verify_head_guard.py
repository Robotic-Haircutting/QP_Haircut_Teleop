import sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sim_bridge import Simulation,WORKING_JOINTS,PROJECTS
from head_guard import HeadGuard
from reachable_step import solve_tracking_ik
from piper_twin import fk_jacobian

g=HeadGuard(PROJECTS/'mujoco_haircut');home=WORKING_JOINTS.copy()
assert g.clearance(home)['distance_mm']>10
unsafe=np.array([17.2713102874882,64.18196467755197,-31.55357679455947,4.92094803882275,19.67000341749531,25.786481718484])
assert g.clearance(unsafe)['distance_mm']<0
assert not g.edge_safe(home,unsafe)
q,info=g.filter_step(home,unsafe);assert info['blocked'] and g.clearance(q)['distance_mm']>=10
# Both endpoints are safe; the old straight joint return crosses the head.
# 2026-10-07: pose re-picked for the current shared scene (head moved ~5.5 cm by the SO-101 single-point calibration,
# clipper tip 0.20 m). Old pose [71.997,55.093,-29.325,-25.150,40.041,-7.411] no longer crosses the head on the straight return.
start=np.array([67.42631935946443,60.49818001455333,-9.974497316549222,-29.19414347639257,37.60576312891074,2.612331075393266])
assert g.clearance(start)['distance_mm']>10 and not g.edge_safe(start,home)
route=g.plan_return(start,home);assert route and len(route)>1
minimum=1000
for a,b in zip([start]+route,route):
    assert g.edge_safe(a,b,time.monotonic()+3)
    for f in np.linspace(0,1,101):minimum=min(minimum,g.clearance(a+(b-a)*f)['distance_mm'])
assert minimum>=10
assert g.plan_return(unsafe,home) is None
print('SWEPT GUARD PASS: penetrating targets blocked; safe endpoints with unsafe midpoint rejected; detour minimum clearance',round(minimum,3),'mm; unsafe return holds',flush=True)

sim=Simulation();data=dict(client_id='head-test',moving='phone',camera_id='test',tag_id=0,tag_mm=80,width=640,height=480,calibration=dict(width=640,height=480,fx=800,fy=800,cx=320,cy=240,calibrated=True))
def feed(p):
    p=np.asarray(p,float);sim.update(dict(t_mm=(-p).tolist(),R=np.eye(3).tolist(),camera_t_mm=p.tolist(),camera_R=np.eye(3).tolist(),reprojection_px=.1,ambiguous=False),data)
def hold(p,seconds):
    end=time.monotonic()+seconds;minimum=1000;blocked=False
    while time.monotonic()<end:
        feed(p);time.sleep(.03)
        state=sim.state();minimum=min(minimum,state['head_safety']['distance_mm']);blocked=blocked or state['head_safety']['blocked'] or (state.get('motion_limit') or {}).get('cause')=='head_guard'
    assert minimum>=10-1e-6,minimum
    return blocked
try:
    sim.start('head-test');feed([0,0,-550]);sim.resume();session=sim.session_path;progress=sim.coverage_path
    blocked=hold([0,200,-550],3)
    assert blocked,'approaching head was not intercepted'
    before=sim.twin.q.copy();distance=sim.head_safety['distance_mm']
    hold([0,100,-550],1.3)
    assert np.max(np.abs(before-sim.twin.q))>.1
    assert sim.head_safety['distance_mm']>distance+2
    assert sim.session_path==session and sim.coverage_path==progress
    print('REAL MUJOCO FOLLOWING PASS: toward-head input stops outside 10mm; reverse phone input retreats; same data/coverage session',flush=True)
finally:sim.close()
