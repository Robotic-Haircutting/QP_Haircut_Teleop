import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pnd_tracking import PNDTracker,fk_jacobian
from pnd_ik import PNDIK,PNDIKParams
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'mujoco_haircut'))
from haircut_sim.control.qp_solver import solve_box_qp
rng=np.random.default_rng(19);solver=PNDIK(PNDIKParams(iterations=3000,tol=1e-9));worst=0
for _ in range(30):
 b=rng.normal(size=(6,6));h=b.T@b+.5*np.eye(6);g=rng.normal(size=6);lo=-rng.uniform(.1,1,6);hi=rng.uniform(.1,1,6)
 u,_=solver.solve(h,g,lo,hi);ref,_=solve_box_qp(h,g,lo,hi);worst=max(worst,np.max(np.abs(u-ref)))
 assert np.max(np.abs(u-ref))<2e-7
tracker=PNDTracker(speed=15);q=np.array([14.4,49.3,-57.9,0,38.6,0.]);t,_=fk_jacobian(np.deg2rad(q));goal=t[:3,3]*1000+[5,5,5]
for _ in range(200):
 old=q.copy();q,p,r,info=tracker.step(goal,t[:3,:3],q,.02)
 assert np.max(np.abs(q-old))<=.300001 and info['converged']
assert np.linalg.norm(p-goal)<.1
print('PND QP PASS: 30 problems match active-set reference <2e-7 rad/s; 15deg/s bound; XYZ tracking <0.1mm',worst)
import json
tracker=PNDTracker(speed=45,max_cartesian_speed=.1,max_orientation_speed=.1)
q=np.array([14.4,49.3,-57.9,0,38.6,0.]);t,_=fk_jacobian(np.deg2rad(q))
_,_,_,info=tracker.step(t[:3,3]*1000+[30,20,10],t[:3,:3],q,.02)
assert info['velocity_scale']<1 and type(info['limited']) is bool
json.dumps(info)
print('LIMITED STATE JSON PASS: Cartesian speed saturation telemetry serializes as native JSON')
