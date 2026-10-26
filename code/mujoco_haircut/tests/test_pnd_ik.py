from pathlib import Path
import sys, types, time
import numpy as np

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
from haircut_sim.pnd_ik import PNDIK, PNDIKParams
from haircut_sim.control.qp_solver import solve_box_qp
from haircut_sim.control.piper_kinematics import piper_fk
from haircut_sim.control.config_piper_follower import PiperFollowerConfig
from haircut_sim.control.piper_follower import PiperFollower
from haircut_sim.control.config_so101_leader_ee import SO101LeaderEEConfig
from haircut_sim.control.so101_leader_ee import SO101LeaderEE

rng = np.random.default_rng(5)
solver = PNDIK(PNDIKParams(iterations=3000, tol=1e-9))
worst = 0
for _ in range(50):
    B = rng.normal(size=(6,6))
    H = B.T @ B + .5*np.eye(6)
    f = rng.normal(size=6)
    lo, hi = -rng.uniform(.1,1,6), rng.uniform(.1,1,6)
    hi[0] = lo[0]  # locked joint
    u,_ = solver.solve(H,f,lo,hi)
    ref,_ = solve_box_qp(H,f,lo,hi)
    worst = max(worst, float(np.max(np.abs(u-ref))))
    assert np.max(np.abs(u-ref)) < 2e-7
print('50 QPs vs active-set reference, max speed difference:', worst)

solver = PNDIK()
q = np.deg2rad([149.5,50,-50,0,30,0])
T = piper_fk(q)
H,f,lo,hi,e = solver.build_qp(q,T[:3,3],T[:3,:3],1/60)
assert abs(np.rad2deg(hi[0])-30) < 1e-9
q[0] = np.deg2rad(150)
assert solver.build_qp(q,T[:3,3],T[:3,:3],1/60)[3][0] == 0
print('Angle-bound example: 149.5 deg -> max 30 deg/s; at 150 -> max 0')

q = np.deg2rad([0,50,-50,0,30,0])
T = piper_fk(q)
target = T[:3,3] + [.02,.01,.01]
start_err = np.linalg.norm(target-T[:3,3])
times=[]; not_converged=0
for _ in range(360):
    prev=q.copy(); t=time.perf_counter()
    q=solver.step(q,target,T[:3,:3],1/60)
    times.append((time.perf_counter()-t)*1000)
    not_converged += not solver.last_info['converged']
    assert np.max(np.abs(q-prev)) <= np.deg2rad(45)/60+1e-12
    lim=np.deg2rad(solver.p.joint_limits_deg)
    assert np.all(q>=lim[:,0]-1e-12) and np.all(q<=lim[:,1]+1e-12)
end_err=np.linalg.norm(target-piper_fk(q)[:3,3])
assert end_err < .002
print('Tracking position error mm:', start_err*1000, '->', end_err*1000)
print('Solve ms mean/p95/max:', np.mean(times),np.percentile(times,95),max(times),'unconverged:',not_converged)

leader=SO101LeaderEE(SO101LeaderEEConfig(port='none',mock=True,output_gripper=False))
robot=PiperFollower(PiperFollowerConfig(id='pnd_test',mock=True,control_mode='qp',ik_solver='pnd',start_joints_deg=[0,50,-50,0,30,0],return_on_disconnect=False))
leader.connect();robot.connect()
try:
    for i in range(100):
        action=leader.get_action()
        assert set(action)=={'ee.x','ee.y','ee.z','ee.wx','ee.wy','ee.wz'}
        robot.send_action(action)
        time.sleep(1/60)
finally:
    robot.disconnect();leader.disconnect()
print('Mock integration: pose-only action -> PND QP -> joint command passed')
