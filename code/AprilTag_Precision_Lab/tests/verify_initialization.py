import sys,time,tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from initialization_path import initialization_plan,initialization_step,WORKING_JOINTS
from robot_bridge import RobotExperiment
from piper_hardware import MockHardware

folded=np.array([34.697,-2.094,.299,-.946,29.269,14.319])
robot=RobotExperiment(root=Path(tempfile.mkdtemp())/'results')
try:
 robot.action(dict(action='connect',client_id='init',backend='mock'))
 robot.backend.q=folded.copy()
 limits=robot.tracker.solver.p.joint_limits_deg
 path,guard=initialization_plan(folded,limits);q=folded.copy();minimum=10000
 for waypoint in path:
  while np.max(np.abs(q-waypoint))>1e-6:
   nxt=initialization_step(q,waypoint,.02)
   assert np.max(np.abs(nxt-q))<=.300001
   assert guard.edge_safe(q,nxt)
   minimum=min(minimum,guard.clearance(nxt)['distance_mm']);q=nxt
 assert minimum>=10 and np.allclose(q,WORKING_JOINTS)
 for bad in [folded+[0,-5,0,0,0,0],folded+[0,0,5,0,0,0]]:
  try:initialization_plan(bad,limits);raise AssertionError('invalid start accepted')
  except ValueError:pass
 try:robot.action(dict(action='prepare',client_id='init'));raise AssertionError('missing bench interlock')
 except ValueError:pass
 robot.action(dict(action='prepare',client_id='init',bench_clear=True))
 time.sleep(.1);assert robot.preparing and robot.backend.sent>0
 robot.action(dict(action='clutch',client_id='init'));assert not robot.preparing and robot.paused and not robot.initial_ready
 try:robot.action(dict(action='clutch',client_id='init',bench_clear=True));raise AssertionError('follow before home accepted')
 except ValueError:pass
 held=robot.backend.q.copy();time.sleep(.1);assert np.array_equal(held,robot.backend.q)
 robot.action(dict(action='prepare',client_id='init',bench_clear=True))
 end=time.monotonic()+18
 while robot.preparing and time.monotonic()<end:time.sleep(.05)
 assert not robot.preparing and not robot.fault,robot.state()
 assert np.max(np.abs(robot.backend.q-WORKING_JOINTS))<.25
 assert robot.paused and robot.backend.enabled and robot.initial_ready
 print('INITIALIZATION PASS: recorded folded pose -> work pose, 15deg/s limit, model head >=10mm, bench interlock, pause holds, completion remains paused; minimum clearance',minimum,flush=True)
finally:robot.close()
