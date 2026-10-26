"""Regression from actual recorded J5=68deg failures, including clutch return."""
import json,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sim_bridge import Simulation,WORKING_JOINTS
from reachable_step import bounded_step,JOINT_MIN_DEG,JOINT_MAX_DEG
from piper_twin import fk_jacobian
from phone_pose import rotation_step

replays=json.loads(Path(__file__).with_name('reach_failure_replays.json').read_text(encoding='utf-8'))
for item in replays:
    q=np.array(item['sample']['joints_deg']);current,_=fk_jacobian(np.deg2rad(q))
    outward=q.copy();outward[4]+=5;target,_=fk_jacobian(np.deg2rad(outward))
    dp=(target[:3,3]-current[:3,3])*1000
    p=current[:3,3]*1000+dp*min(1,.8/max(np.linalg.norm(dp),1e-9))
    r=rotation_step(current[:3,:3],target[:3,:3],np.deg2rad(.4))
    candidate,_,_,info=bounded_step(p,r,q,.02)
    assert np.all(candidate>=JOINT_MIN_DEG) and np.all(candidate<=JOINT_MAX_DEG)
    assert np.max(np.abs(candidate-q))<=.9+1e-8
edge=np.array(replays[-1]['sample']['joints_deg'])
assert edge[4]==68
current,_=fk_jacobian(np.deg2rad(edge));inside=edge.copy();inside[4]-=5
target,_=fk_jacobian(np.deg2rad(inside));dp=(target[:3,3]-current[:3,3])*1000
p=current[:3,3]*1000+dp*min(1,.8/np.linalg.norm(dp))
r=rotation_step(current[:3,:3],target[:3,:3],np.deg2rad(.4))
candidate,_,_,info=bounded_step(p,r,edge,.02)
assert candidate[4]<edge[4]-.1 and not info['limited']
print('RECORDED BOUNDARY PASS: all 14 recorded poses respect joint/rate bounds; inward motion retreats from J5=68',flush=True)

sim=Simulation();sim.recovery_q=edge.copy()
sample=replays[-1]['sample'];camera_p=np.array(sample['control_position_mm']);camera_r=np.array(sample['control_R'])
data=dict(client_id='boundary-test',moving='phone',camera_id='boundary-camera',tag_id=0,tag_mm=80,width=640,height=480,calibration=dict(width=640,height=480,fx=800,fy=800,cx=320,cy=240,calibrated=True))
def feed(offset=0):
    p=camera_p+np.array([offset,0,0]);tag_r=camera_r.T;tag_t=-tag_r@p
    result=dict(t_mm=tag_t.tolist(),R=tag_r.tolist(),camera_t_mm=p.tolist(),camera_R=camera_r.tolist(),reprojection_px=.1,ambiguous=False)
    sim.update(result,data)
try:
    sim.start('boundary-test');original_session=sim.session_path;original_tracking=sim.tracking_path;original_coverage=sim.coverage_path;feed();state=sim.action(dict(client_id='boundary-test',action='clutch'))
    assert state['returning'] and state['paused']
    assert sim.coverage_path==original_coverage
    assert np.max(np.abs(sim.twin.q-edge))<1,'return teleported'
    last=sim.twin.q.copy();previous=time.monotonic();deadline=previous+25
    while sim.returning and time.monotonic()<deadline:
        feed(15);time.sleep(.04);now=time.monotonic();q=sim.twin.q.copy()
        assert np.max(np.abs(q-last))<=15*(now-previous)+.8,'return exceeded speed bound'
        assert np.all(q>=JOINT_MIN_DEG) and np.all(q<=JOINT_MAX_DEG)
        last=q;previous=now
    assert not sim.returning and not sim.paused,sim.reason
    assert sim.session_path==original_session and sim.tracking_path==original_tracking,'return reset data files'
    assert np.max(np.abs(sim.twin.q-WORKING_JOINTS))<1e-7
    assert np.linalg.norm(sim.ref-(camera_p+[15,0,0]))<1e-7
    print('AUTO RETURN PASS: clutch at actual J5 boundary slowly returns to working posture, then anchors current phone pose without jump',flush=True)
    # Cancel a second return and ensure further detections do not restart it.
    sim.pause();sim.twin.show_joints(edge);feed()
    sim.action(dict(client_id='boundary-test',action='clutch'));time.sleep(.12)
    sim.action(dict(client_id='boundary-test',action='clutch'));saved=sim.twin.q.copy()
    for _ in range(4):feed();time.sleep(.04)
    assert sim.paused and not sim.returning and not sim.resume_pending
    assert np.max(np.abs(sim.twin.q-saved))<1e-9
    print('RETURN CANCEL PASS: button stops return immediately; fresh detections do not resume',flush=True)
    # With no valid tag, return finishes into waiting rather than following.
    sim.twin.show_joints(WORKING_JOINTS+np.array([0,0,0,0,1,0]));sim.last=0
    sim.begin_return();deadline=time.monotonic()+2
    while sim.returning and time.monotonic()<deadline:time.sleep(.03)
    assert not sim.returning and sim.paused and sim.resume_pending
    feed();assert not sim.paused and not sim.resume_pending
    print('RETURN NO-TAG PASS: completion holds until fresh input establishes the new zero',flush=True)
finally:sim.close()
