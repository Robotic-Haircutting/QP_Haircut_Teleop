import sys,time,tempfile,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sim_bridge import Simulation,PROJECTS,WORKING_JOINTS,fk_jacobian
from robot_bridge import RobotExperiment
from head_alignment import crown,apply_alignment,translate_coverage,flange_config
sys.path.insert(0,str(PROJECTS/'mujoco_haircut'))
from haircut_sim.scene import Scene,load_config
from hair_progress import PersistentTrailOverlay
import mujoco

scene=Scene(flange_config(load_config(None)));scene.set_piper(WORKING_JOINTS);scene.forward()
tf,_=fk_jacobian(np.deg2rad(WORKING_JOINTS))
actual=scene.data.xpos[scene.model.body('piper/link6').id]
assert np.linalg.norm(actual-tf[:3,3])<.002,(actual,tf[:3,3])
trail=PersistentTrailOverlay(scene,scene.cfg if hasattr(scene,'cfg') else flange_config(load_config(None)),Path(tempfile.mkdtemp())/'coverage.npz')
scn=mujoco.MjvScene(scene.model,maxgeom=10000);trail.attach(scn)
trail.cov.covered[:10]=True;covered=trail.cov.covered.copy();points=trail.cov.pts.copy()
trail.segments=[np.array([[.1,.2,.3],[.2,.3,.4]])];trail.colors=[np.ones(4)];segment=trail.segments[0].copy()
translate_coverage(trail,np.array([.01,.02,.03]))
assert np.allclose(trail.cov.pts,points+[.01,.02,.03]) and np.array_equal(covered,trail.cov.covered)
assert np.array_equal(segment,trail.segments[0])

sim=Simulation();robot=RobotExperiment(root=Path(tempfile.mkdtemp())/'experiment')
try:
 sim.observer_url='https://127.0.0.1:1/no-camera';sim.start('align-test')
 robot.action(dict(action='connect',backend='mock',client_id='align-test'))
 sent=robot.backend.sent;q=robot.backend.q.copy()
 result=robot.align_head(dict(client_id='align-test'),sim);record=result['alignment']
 assert robot.backend.sent==sent and np.array_equal(q,robot.backend.q),'alignment sent motion'
 assert not robot.state()['head_registered']
 assert np.allclose(crown(sim.head_guard.model,sim.head_guard.data),np.array(record.get('tool_tip_world_m',record['flange_world_m']))-[0,0,.01],atol=1e-7)  # clipper tip when head_alignment.TOOL_MOUNTED
 second=robot.align_head(dict(client_id='align-test'),sim)['alignment']
 assert np.allclose(second['head_body_pos_m'],record['head_body_pos_m'],atol=1e-7),'alignment accumulated'
 limit=time.monotonic()+12
 while time.monotonic()<limit and sim.state()['head_alignment_applied']!=second['revision']:time.sleep(.1)
 assert sim.state()['head_alignment_applied']==second['revision'],'viewer did not acknowledge alignment'
 assert sim.thread.is_alive()
 assert json.loads((Path(__file__).resolve().parents[1]/'results/calibration/head_alignment.json').read_text(encoding='utf-8'))['revision']==second['revision']
 robot.preparing=True
 try:robot.align_head(dict(client_id='align-test'),sim);raise AssertionError('homing arm accepted')
 except ValueError:pass
 robot.preparing=False;robot.paused=False;q=robot.backend.q.copy()
 robot.align_head(dict(client_id='align-test'),sim)
 assert robot.paused and np.array_equal(q,robot.backend.q),'alignment moved arm instead of holding'
 print('HEAD ALIGNMENT PASS: crown 10mm below the reference point ('+record['reference']+'); repeated align no drift; native MuJoCo ACK; paused capture sends no command; following capture holds current pose; preserved coverage and world paths; homing rejected; physical cutting remains unvalidated',flush=True)
finally:robot.close();sim.close()
