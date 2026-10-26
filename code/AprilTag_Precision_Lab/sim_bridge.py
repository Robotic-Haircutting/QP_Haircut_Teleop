"""Laptop AprilTag -> simulation only. No hardware receiver or SDK is imported."""
import os
import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

PROJECTS = Path(os.environ.get('HHE_ROBOT_ROOT', str(Path(__file__).resolve().parent.parent)))
PIPER = PROJECTS/'Client_RHCR/piper'
sys.path.insert(0, str(PIPER))
from piper_twin import MujocoTwin, solve_ik, fk_jacobian
from follow_filter import FollowFilter
from head_guard import HeadGuard
from reachable_step import bounded_step, JOINT_MIN_DEG, JOINT_MAX_DEG
from tracking_csv import TrackingCSV,plot_tracking
from phone_pose import CAMERA_TO_BASE,PHONE_POSITION_SIGN,select_phone_pose,rotation_step
from scipy.spatial.transform import Rotation
from pnd_tracking import PNDTracker

# Camera +X right, +Y down, +Z away. Operator facing camera:
# approach -> robot +X; operator left -> robot +Y; up -> robot +Z.
DEFAULT_MAP = np.array([[0.,0.,-1.],[1.,0.,0.],[0.,-1.,0.]])
from initial_pose import INITIAL_JOINTS_DEG
WORKING_JOINTS=np.array(INITIAL_JOINTS_DEG,float)

def free_port(kind):
    with socket.socket(socket.AF_INET, kind) as s:
        s.bind(('127.0.0.1',0))
        return s.getsockname()[1]

class Simulation:
    def __init__(self):
        self.lock=threading.RLock()
        self.twin=None;self.viewer=None;self.stream_port=None;self.owner=None
        self.paused=True;self.reason='Simulation not started';self.latest=None;self.last=0
        self.axis=DEFAULT_MAP.copy();self.origin=None;self.samples={};self.config=None
        self.alive=True;self.logfile=None
        self.control=None;self.head_alignment=None;self.head_alignment_applied=None;self.last_alignment_send=0
        self.input_mode='tag';self.depth=None
        self.latest_r=None;self.phone_axis=None;self.phone_previous=None
        self.recovery_q=None
        self.pose_cache={}
        self.last_rejection=0
        self.anchored=False
        self.resume_pending=False;self.pending_until=0
        self.motion_limit=None;self.last_limit_log=0
        self.returning=False
        self.tracking_hold=False
        self.scale=.35
        self.solver_kind='DLS';self.pnd=PNDTracker();self.observer_url=None;self.mirror_only=False
        self.follow_filter=FollowFilter()
        self.head_guard=None;self.head_safety=None;self.return_path=[];self.return_blocked=False
        self.session_log=None;self.session_path=None;self.coverage_path=None;self.last_sample=0
        self.tracking=None;self.tracking_path=None
        self.tool_mm=np.zeros(3)
        self.plot_path=None
        self.plotting=False;self.plot_error=None
        self.plot_threads=[]
        self.thread=threading.Thread(target=self._loop,daemon=True);self.thread.start()

    def start(self,owner):
        if not isinstance(owner,str) or not owner:raise ValueError('Missing page session')
        with self.lock:
            if self.twin and self.viewer and self.viewer.poll() is not None:
                self._viewer_exited()
            if self.twin:
                if owner!=self.owner:raise ValueError('The simulation is controlled by another page; stop it from that page')
                return self.state()
            udp=free_port(socket.SOCK_DGRAM)
            self.control=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
            self.control.bind(('127.0.0.1',0));self.control.setblocking(False)
            control=self.control.getsockname()[1]
            self.stream_port=free_port(socket.SOCK_STREAM)
            logs=Path(__file__).parent/'results';logs.mkdir(exist_ok=True)
            session_id=f'session_{time.time_ns()}'
            self.coverage_path=logs/'coverage'/f'{session_id}.npz'
            self.logfile=open(logs/'simulation_viewer.log','w',encoding='utf-8')
            scene=PROJECTS/'mujoco_haircut'
            self.return_blocked=False
            self.head_guard=HeadGuard(scene,flange=bool(self.observer_url),alignment=self.head_alignment if self.observer_url else None)
            self.head_alignment_applied=None
            initial=self.recovery_q if self.recovery_q is not None else WORKING_JOINTS
            self.head_safety=dict(**self.head_guard.clearance(initial),blocked=False,retreat=False)
            env=os.environ.copy();env['PYTHONPATH']=str(scene)+os.pathsep+env.get('PYTHONPATH','')
            try:
                self.viewer=subprocess.Popen([sys.executable,str(Path(__file__).parent/'cutting_view.py'),
                    '--port',str(udp),'--control-port',str(control),'--stream-port',str(self.stream_port),'--coverage-path',str(self.coverage_path)]+(['--observer-url',self.observer_url,'--flange-reference'] if self.observer_url else []),
                    cwd=scene,env=env,stdout=self.logfile,stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                self.twin=MujocoTwin(port=udp,spawn=False,haircut_dir=str(scene),
                    start_deg=self.recovery_q if self.recovery_q is not None else WORKING_JOINTS,log=lambda text:None)
                self.twin.start()
                self.recovery_q=None
            except Exception:
                self.stop();raise
            self.owner=owner;self.latest=None;self.config=None;self.last=0
            self.phone_previous=None;self.anchored=False
            self.resume_pending=False
            self.motion_limit=None
            self.returning=False
            self.tracking_hold=False
            self.paused=True;self.reason='Simulation started; keep the Tag visible and press Resume to set the reference'
            data_dir=logs/'data';data_dir.mkdir(exist_ok=True)
            self.session_path=data_dir/f'{session_id}.jsonl'
            self.session_log=open(self.session_path,'a',encoding='utf-8',buffering=1)
            self.tracking_path=self.session_path.with_suffix('.csv');self.tracking=TrackingCSV(self.tracking_path)
            self.plot_path=None
            self.plotting=False;self.plot_error=None
            self.last_sample=0
            self.record('start',scale=self.scale,axis_map=self.axis.tolist(),mode='simulation',coverage_file=str(self.coverage_path),coverage_initialization='new_session')
            return self.state()

    def record(self,event,**data):
        if not self.session_log:return
        try:
            self.session_log.write(json.dumps(dict(t=round(time.time(),3),time_ms=round(time.time()*1000),event=event,**data),ensure_ascii=False)+'\n')
        except OSError:
            self.session_log.close();self.session_log=None

    def update(self,result,data):
        with self.lock:
            client=data.get('client_id')
            if isinstance(client,str) and client:
                self.pose_cache.pop(client,None)
                self.pose_cache[client]=(time.monotonic(),result,data.copy())
                while len(self.pose_cache)>10:self.pose_cache.pop(next(iter(self.pose_cache)))
            if not self.twin or data.get('client_id')!=self.owner:return
            mode=data.get('moving','tag')
            if mode not in ('tag','phone'):return
            config=json.dumps([mode,data.get('camera_id'),data.get('tag_id'),data['tag_mm'],data['width'],data['height'],data['calibration'],data.get('tracking_mode','single'),data.get('board')],sort_keys=True)
            same_config=config==self.config
            selected=0
            if not self.paused and self.latest is not None and not self.fresh():self.hold_tracking()
            if mode=='phone':
                previous=self.phone_previous if config==self.config and self.fresh() else None
                p,r,selected=select_phone_pose(result,previous)
                self.phone_previous=(p.copy(),r.copy());self.latest_r=r
            else:p=np.asarray(result['t_mm'],float)
            if p.shape!=(3,) or not np.isfinite(p).all() or result['reprojection_px']>3:
                error=float(result['reprojection_px'])
                self.last=0
                if not self.paused:self.hold_tracking()
                if not self.returning:self.reason=f'Pose temporarily unavailable: reprojection {error:.2f} px (must be ≤3 px); resumes automatically when valid detection returns'
                if time.monotonic()-self.last_rejection>1:
                    self.last_rejection=time.monotonic()
                    self.record('pose_rejected',reprojection_px=error,corners=data.get('corners'),calibration=data['calibration'],width=data['width'],height=data['height'],tag_mm=data['tag_mm'])
                return
            if config!=self.config:
                self.record('configuration',tracking_mode=data.get('tracking_mode','single'),board=data.get('board'),moving=mode,tag_id=data.get('tag_id'),tag_mm=data['tag_mm'],width=data['width'],height=data['height'],calibration=data['calibration'])
            if self.config is not None and self.config!=config:
                if not self.paused:self.hold_tracking()
                self.origin=None;self.samples={}
            self.latest=p;self.last=time.monotonic();self.config=config
            self.input_mode=mode;self.depth=float(result['t_mm'][2])
            if self.tracking_hold and not self.paused:self.anchor_tracking(preserve_axes=same_config)
            if self.last-self.last_sample>=.1:
                self.last_sample=self.last
                actual,r=self.twin.home_pose()
                with self.twin.lock:q=self.twin.q.copy()
                target=(self.home+self.delta()-r@self.tool_mm).tolist() if not self.paused else None  # flange-referenced like actual_mm
                self.record('sample',board_ids=result.get('board_ids'),visible_ids=result.get('visible_ids'),rejected_ids=result.get('rejected_ids'),board_geometry=result.get('board_geometry'),tag_camera_mm=result['t_mm'],camera_tag_mm=result['camera_t_mm'],control_position_mm=p.tolist(),tag_R=result['R'],camera_R=result['camera_R'],reprojection_px=result['reprojection_px'],
                    selected_candidate=selected,ambiguous=result['ambiguous'],raw_camera_tag_mm=result.get('raw_camera_t_mm'),candidate_errors_px=result.get('candidate_errors_px'),control_R=self.latest_r.tolist() if mode=='phone' else None,
                    moving=mode,position_frame=result.get('position_frame','camera_in_tag') if mode=='phone' else 'tag_in_camera',depth_mm=self.depth,
                    paused=self.paused,returning=self.returning,tracking_hold=self.tracking_hold,reason=self.reason,target_mm=target,actual_mm=actual.tolist(),actual_R=r.tolist(),joints_deg=q.tolist())
            if self.resume_pending and time.monotonic()<self.pending_until:self.resume()
            return True

    def delta(self):
        if self.input_mode=='phone':return PHONE_POSITION_SIGN@self.phone_axis@(self.latest-self.ref)*self.scale
        return self.axis@(self.latest-self.ref)*self.scale

    def target_rotation(self):
        if self.input_mode=='phone':
            relative=self.latest_r@self.ref_r.T
            return self.phone_axis@relative@self.phone_axis.T@self.rotation
        return self.rotation

    def pause(self,reason='Paused; re-center the Tag, then press Resume',cancel_pending=True,cancel_return=True):
        with self.lock:
            self.tracking_hold=False
            if cancel_pending:self.resume_pending=False
            if cancel_return and self.returning:
                self.returning=False;self.record('return_cancelled')
            if not self.paused or self.reason!=reason:self.record('pause',reason=reason)
            self.paused=True;self.reason=reason
            if self.twin:self.twin.status('PAUSED')

    def fresh(self):return self.latest is not None and time.monotonic()-self.last<.6

    def _tip(self,p_mm,r):
        """Flange pose (mm, R) -> clipper tip position (mm)."""
        return np.asarray(p_mm,float)+np.asarray(r)@getattr(self,'tool_mm',np.zeros(3))

    def hold_tracking(self):
        if not self.tracking_hold:self.record('tracking_hold')
        self.tracking_hold=True
        self.reason='Waiting for a valid pose; holding position. Tracking resumes automatically when detection recovers'

    def near_limits(self):
        if not self.twin:return False
        with self.twin.lock:q=self.twin.q.copy()
        return bool(np.any(np.minimum(q-JOINT_MIN_DEG,JOINT_MAX_DEG-q)<2))

    def begin_return(self):
        if self.mirror_only:self.pause('Measured joint mirror. Use Return Home in Arm Control.');return
        self.paused=True;self.resume_pending=False;self.returning=False;self.motion_limit=None
        with self.twin.lock:q=self.twin.q.copy()
        self.return_path=self.head_guard.plan_return(q,WORKING_JOINTS)
        if not self.return_path:
            self.return_blocked=True
            self.reason='Head guard: no homing path keeps 10 mm clearance, so the arm stopped; move outward first, then home'
            self.twin.status('PAUSED');self.record('return_blocked',reason=self.reason,**self.head_guard.clearance(q));return
        self.returning=True;self.return_blocked=False
        self.reason='Arm is returning to the initial working pose at up to 15°/s; the phone is not driving it; press the button to stop homing'
        self.twin.status('RETURNING')
        self.record('return_started',target_joints_deg=WORKING_JOINTS.tolist(),max_joint_speed_deg_s=15,safe_waypoints=[q.tolist() for q in self.return_path],head_clearance_mm=10)

    def return_step(self,dt):
        with self.twin.lock:q=self.twin.q.copy()
        difference=self.return_path[0]-q
        remaining=float(np.max(np.abs(difference)))
        fraction=min(1.,15*dt/max(remaining,1e-12))
        next_q=q+difference*fraction
        checked,info=self.head_guard.filter_step(q,next_q);self.head_safety=info
        if info['blocked']:
            self.returning=False;self.return_path=[];self.return_blocked=True;self.pause('Head 10 mm guard: homing path blocked, homing stopped')
            self.record('return_blocked',**info);return
        self.twin.show_joints(checked)
        p,r=self.twin.home_pose();self.command=self._tip(p,r);self.command_rotation=r.copy()
        if self.tracking:self.tracking.log(p,r,next_q,0)
        if fraction>=1:
            self.return_path.pop(0)
            if self.return_path:return
            self.returning=False;self.record('return_completed',joints_deg=next_q.tolist())
            if self.fresh():self.resume()
            else:
                self.resume_pending=True;self.pending_until=float('inf')
                self.reason='Arm is home; waiting for a valid phone pose to set a new reference (press the button to cancel)'

    def resume(self):
        if self.mirror_only:self.pause('Measured joint mirror. Use Resume in Arm Control.');return
        with self.lock:
            if not self.twin:raise ValueError('Start the simulation first')
            if self.viewer and self.viewer.poll() is not None:raise ValueError('MuJoCo window was closed; stop and start again')
            if self.near_limits() and not self.return_blocked:self.begin_return();return
            if not self.fresh():raise ValueError('No recent Tag detection; keep the Tag in view')
            self.anchor_tracking()

    def anchor_tracking(self,preserve_axes=False):
        with self.lock:
            from head_alignment import tool_offset_mm
            self.tool_mm=tool_offset_mm()  # control point = clipper tip (user 2026-10-07, same as SO-101)
            self.ref=self.latest.copy();self.home,self.rotation=self.twin.home_pose();self.home=self._tip(self.home,self.rotation)
            self.ref_depth=self.depth
            if self.input_mode=='phone':
                self.ref_r=self.latest_r.copy()
                if not preserve_axes or self.phone_axis is None:self.phone_axis=CAMERA_TO_BASE@self.ref_r.T
            self.command_rotation=self.rotation.copy()
            self.follow_filter.reset(self.home,self.rotation)
            self.pnd.reset()
            self.anchored=True
            self.resume_pending=False
            self.motion_limit=None
            self.tracking_hold=False
            self.command=self.home.copy();self.paused=False;self.reason=f'Following · translation 10:{self.scale*10:g}'+(' · orientation 1:1' if self.input_mode=='phone' else ' · tool orientation fixed')
            self.twin.status('FOLLOWING')
            self.record('resume_anchor',tag_reference_mm=self.ref.tolist(),actual_anchor_mm=self.home.tolist(),fixed_R=self.rotation.tolist(),axis_map=(PHONE_POSITION_SIGN@self.phone_axis).tolist() if self.input_mode=='phone' else self.axis.tolist(),orientation_axis_map=self.phone_axis.tolist() if self.input_mode=='phone' else None,moving=self.input_mode,phone_reference_R=self.ref_r.tolist() if self.input_mode=='phone' else None,scale=self.scale)

    def align(self,step):
        with self.lock:
            if not self.twin or not self.fresh():raise ValueError('Start the simulation and keep the Tag visible first')
            self.pause('Calibrating directions; return to the start point before each sample, then move along the requested direction')
            if step=='origin':
                self.origin=self.latest.copy();self.samples={};return
            if step not in ('forward','left','up') or self.origin is None:raise ValueError('Sample the start point first')
            v=self.latest-self.origin
            if np.linalg.norm(v)<40:raise ValueError('Move at least 40 mm before sampling')
            samples=dict(self.samples);samples[step]=v/np.linalg.norm(v)
            if len(samples)==3:
                basis=np.column_stack([samples[k] for k in ('forward','left','up')])
                if np.linalg.det(basis)<.4:raise ValueError('Directions inconsistent: move forward, left and up separately, each time from the same start point')
                u,_,vt=np.linalg.svd(basis);rotation=u@vt
                errors=np.degrees(np.arccos(np.clip(np.sum(rotation*basis,axis=0),-1,1)))
                if np.max(errors)>20:raise ValueError('The three directions are not perpendicular enough; sample again')
                self.axis=rotation.T;self.reason='Direction alignment done; return to a comfortable position and press Resume'
            self.samples=samples
            self.record('axis_alignment',step=step,axis_map=self.axis.tolist(),samples=list(self.samples))

    def align_head(self,owner,flange_mm,joints_deg):
        from head_alignment import make_alignment,apply_alignment
        with self.lock:
            if not self.twin or self.owner!=owner or not self.observer_url:raise ValueError('Start the Piper MuJoCo view before Align Head.')
            if not self.paused:raise ValueError('Pause simulation control before alignment.')
            guard=self.head_guard;guard.clearance(joints_deg)
            record=make_alignment(guard.cfg,guard.model,guard.data,flange_mm,joints_deg)
            root=Path(__file__).parent/'results/calibration';root.mkdir(parents=True,exist_ok=True)
            payload=json.dumps(record,indent=2)
            (root/f'head_alignment_{record["revision"]}.json').write_text(payload,encoding='utf-8')
            temporary=root/'head_alignment.tmp';temporary.write_text(payload,encoding='utf-8');temporary.replace(root/'head_alignment.json')
            apply_alignment(guard.model,guard.data,record);guard.last=None
            self.head_alignment=record;self.head_alignment_applied=None;self.last_alignment_send=0
            self.record('head_alignment',**record)
            return record

    def _loop(self):
        previous=time.monotonic()
        while self.alive:
            now=time.monotonic();dt=min(now-previous,.05);previous=now
            with self.lock:
                if self.control:
                    try:
                        message=json.loads(self.control.recv(2048))
                        if message.get('type')=='head_alignment_applied':self.head_alignment_applied=message.get('revision')
                        if message.get('type')=='clutch' and message.get('action')=='toggle':
                            if self.returning:self.pause('Arm homing stopped')
                            elif self.paused:self.resume()
                            else:self.pause()
                    except (BlockingIOError,ValueError):pass
                if self.twin and self.observer_url and self.head_alignment and now-self.last_alignment_send>.2:
                    self.twin.sock.sendto(json.dumps(dict(s='head_alignment',alignment=self.head_alignment)).encode(),self.twin.addr)
                    self.last_alignment_send=now
                if self.twin and self.viewer and self.viewer.poll() is not None:
                    self._viewer_exited()
                if self.twin and self.returning:self.return_step(dt)
                if self.twin and not self.paused:
                    if not self.fresh():self.hold_tracking()
                    else:
                        delta=self.delta()
                        if np.linalg.norm(delta)>150:self.pause('More than 150 mm from the current reference; re-center, then resume')
                        else:
                            goal=self.home+delta
                            if self.input_mode=='phone':goal=self.follow_filter.position(goal,dt)
                            step=goal-self.command
                            n=np.linalg.norm(step)
                            command=self.command+step*min(1.,40*dt/max(n,1e-12))
                            with self.twin.lock:q0=self.twin.q.copy()
                            target_r=self.target_rotation()
                            if np.degrees(Rotation.from_matrix(target_r@self.rotation.T).magnitude())>60:
                                self.pause('More than 60° from the current reference orientation; re-center, then resume');continue
                            if self.input_mode=='phone':target_r=self.follow_filter.orientation(target_r,dt)
                            command_r=rotation_step(self.command_rotation,target_r,np.deg2rad(20)*dt)
                            solve_started=time.perf_counter()
                            # tip -> flange with the current (q0) flange rotation; solvers stay flange-level
                            tool_world=fk_jacobian(np.deg2rad(q0))[0][:3,:3]@self.tool_mm
                            q,actual_p,actual_r,limit=self.pnd.step(goal-tool_world,target_r,q0,dt) if self.solver_kind=='PND' else bounded_step(command-tool_world,command_r,q0,dt)
                            solve_ms=(time.perf_counter()-solve_started)*1000
                            safe_q,safety=self.head_guard.filter_step(q0,q);self.head_safety=safety
                            if safety['blocked'] and np.max(np.abs(safe_q-q0))<1e-6:
                                retreat_q,_,_,_=bounded_step(command-tool_world,self.command_rotation,q0,dt)
                                retreat_q,retreat_safety=self.head_guard.filter_step(q0,retreat_q)
                                if np.max(np.abs(retreat_q-q0))>1e-6:
                                    safe_q=retreat_q;safety=retreat_safety;self.head_safety=safety
                            if safety['blocked'] or np.max(np.abs(safe_q-q))>1e-6:
                                q=safe_q;actual,_=fk_jacobian(np.deg2rad(q))
                                actual_p=actual[:3,3]*1000;actual_r=actual[:3,:3]
                                limit.update(limited=True,cause='head_guard',accepted_target_mm=actual_p.tolist(),accepted_target_R=actual_r.tolist())
                            self.twin.show_joints(q)
                            actual_tip=self._tip(actual_p,actual_r)
                            self.command=actual_tip;self.command_rotation=actual_r
                            if limit['limited']:
                                self.motion_limit=limit
                                joints=' / '.join('J'+str(i) for i in limit['joints'])
                                self.reason=('Head 10 mm guard: approach blocked; you can move outward' if limit['cause']=='head_guard' else joints+' at joint limit; move back or pause and home' if limit['cause']=='joint_limit' else 'Speed-limited while catching up; the arm keeps following after the phone stops')
                                # Discard blocked displacement/rotation instead of accumulating
                                # it behind the boundary. Keep the user's original axis map.
                                if limit['cause']=='joint_limit' or (limit['cause']=='head_guard' and not safety['retreat']):
                                    self.ref=self.latest.copy();self.home=actual_tip.copy();self.rotation=actual_r.copy()
                                    self.follow_filter.reset(actual_tip,actual_r)
                                    if self.input_mode=='phone':self.ref_r=self.latest_r.copy()
                                if now-self.last_limit_log>.5:
                                    self.last_limit_log=now
                                    self.record('motion_limited',**limit,joints_deg=q.tolist(),requested_mm=command.tolist(),requested_R=command_r.tolist(),actual_mm=actual_p.tolist(),actual_R=actual_r.tolist())
                            elif self.motion_limit is not None:
                                if np.max(np.abs(q-q0))>1e-5:
                                    self.motion_limit=None
                                    self.reason=f'Following · translation 10:{self.scale*10:g}'+(' · orientation 1:1' if self.input_mode=='phone' else ' · tool orientation fixed')
                            if self.tracking:self.tracking.log(np.asarray(limit['accepted_target_mm']),np.asarray(limit['accepted_target_R']),q,solve_ms,limit.get('qp_iters',float('nan')))
                # Real-arm mirror: the window counts hair coverage only in FOLLOWING, so report the ARM's state
                # (server.mirror sets robot_following) instead of the paused simulation's.
                if self.twin:self.twin.status(('FOLLOWING' if getattr(self,'robot_following',False) else 'PAUSED') if self.mirror_only and self.paused else 'RETURNING' if self.returning else 'PAUSED' if self.paused else 'FOLLOWING')
            time.sleep(max(.001,.02-(time.monotonic()-now)))

    def state(self):
        with self.lock:
            p=self.twin.home_pose()[0].tolist() if self.twin else None
            q=None;margins=None
            if self.twin:
                with self.twin.lock:q=self.twin.q.copy()
                margins=np.minimum(q-JOINT_MIN_DEG,JOINT_MAX_DEG-q).tolist()
            cached=self.pose_cache.get(self.owner)
            quality=None
            if cached:
                stamp,result,data=cached
                quality=dict(age_seconds=round(time.monotonic()-stamp,2),reprojection_px=result['reprojection_px'],calibrated=bool(data['calibration'].get('calibrated',False)))
            return dict(running=self.twin is not None,paused=self.paused,reason=self.reason,
                tracking_quality=quality,
                controller_id=self.owner,anchored=self.anchored,
                resume_pending=self.resume_pending,
                motion_limit=self.motion_limit,head_safety=self.head_safety,head_alignment_applied=self.head_alignment_applied,
                tracking_hold=self.tracking_hold,
                returning=self.returning,needs_return=self.near_limits(),return_blocked=self.return_blocked,
                joints_deg=q.tolist() if q is not None else None,joint_margins_deg=margins,
                visible=self.fresh(),position_mm=p,axis_map=self.axis.tolist(),
                moving=self.input_mode,orientation_tracking=self.input_mode=='phone',solver=self.solver_kind,
                solver_info=self.pnd.solver.last_info if self.solver_kind=='PND' else None,
                alignment_samples=list(self.samples),scale=self.scale,
                data_log=str(self.session_path) if self.session_path else None,data_recording=self.session_log is not None,
                tracking_log=str(self.tracking_path) if self.tracking_path else None,
                coverage_file=str(self.coverage_path) if self.coverage_path else None,
                plot_file=str(self.plot_path) if self.plot_path else None,plotting=self.plotting,plot_error=self.plot_error)

    def _generate_plot(self,path):
        output=None;error=None
        try:output=plot_tracking(path)
        except Exception as exc:error=str(exc)
        with self.lock:
            if self.tracking_path==path:
                self.plot_path=output;self.plotting=False;self.plot_error=error

    def _viewer_exited(self):
        code=self.viewer.poll()
        with self.twin.lock:q=self.twin.q.copy()
        self.stop()
        self.recovery_q=q
        self.reason=f'MuJoCo window exited (code {code}); press Start simulation to recover, then Resume to set the reference'

    def action(self,data):
        with self.lock:
            action=data.get('action');owner=data.get('client_id')
            self.record('action_request',action=action,client_id=owner)
            if action=='start':return self.start(owner)
            if action=='clutch':
                if not isinstance(owner,str) or not owner:raise ValueError('Missing page session')
                if self.twin and self.viewer and self.viewer.poll() is not None:self._viewer_exited()
                if self.returning:
                    self.pause('Arm homing stopped; holding position');return self.state()
                if self.resume_pending:
                    self.pause('Waiting cancelled; still paused');return self.state()
                if self.twin and not self.paused:
                    self.pause('Paused; re-center the phone or Tag, then press the same button to resume')
                    return self.state()
                cached=self.pose_cache.get(owner)
                fresh=cached is not None and time.monotonic()-cached[0]<.6
                if not self.twin:self.start(owner)
                if owner!=self.owner:
                    self.owner=owner;self.latest=None;self.last=0;self.config=None;self.phone_previous=None
                    self.record('controller_changed')
                if self.near_limits() and not self.return_blocked:self.begin_return();return self.state()
                accepted=self.update(cached[1],cached[2]) if fresh else False
                if accepted and self.fresh():self.resume()
                if not accepted:
                    self.resume_pending=True;self.pending_until=float('inf')
                    self.reason='Resume received, waiting for a valid pose; '+(self.reason if fresh else 'open the camera on this page and keep the whole Tag visible')
                    self.record('resume_waiting',timeout_seconds=None)
                return self.state()
            if self.twin and owner!=self.owner:raise ValueError('Use the page that started the simulation')
            if action=='sensitivity':
                scale=float(data.get('step'))
                if not np.isfinite(scale) or not .15<=scale<=1.:raise ValueError('Sensitivity range is 10:1.5–10:10')
                self.scale=scale
                if self.twin and not self.paused and self.fresh():self.anchor_tracking(preserve_axes=True)
                self.record('sensitivity_changed',scale=scale)
            elif action=='pause':self.pause()
            elif action=='resume':self.resume()
            elif action=='stop':self.stop()
            elif action=='align':self.align(data.get('step'))
            elif action=='axis':
                raise ValueError('Phone mode now controls position and orientation together; refresh the page (single-direction switching is no longer supported)')
            elif action=='default':
                self.pause();self.axis=DEFAULT_MAP.copy();self.samples={};self.origin=None
            else:raise ValueError('Unknown simulation action')
            return self.state()

    def stop(self):
        with self.lock:
            self.record('stop')
            self.pause('Simulation stopped')
            if self.twin:self.twin.close();self.twin.sock.close();self.twin=None
            if self.viewer and self.viewer.poll() is None:
                self.viewer.terminate()
                try:self.viewer.wait(timeout=5)
                except subprocess.TimeoutExpired:self.viewer.kill();self.viewer.wait(timeout=5)
            self.viewer=None
            if self.control:self.control.close();self.control=None
            if self.logfile:self.logfile.close();self.logfile=None
            self.owner=None;self.latest=None
            if self.session_log:self.session_log.close();self.session_log=None
            if self.tracking:
                self.tracking.close();self.tracking=None
                self.plotting=True;self.plot_error=None
                worker=threading.Thread(target=self._generate_plot,args=(self.tracking_path,),daemon=True)
                self.plot_threads=[w for w in self.plot_threads if w.is_alive()]+[worker];worker.start()

    def close(self):
        self.alive=False;self.stop();self.thread.join(timeout=2)
        for worker in self.plot_threads:worker.join(timeout=30)
