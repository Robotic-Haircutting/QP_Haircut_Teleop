"""Explicit Piper bench experiment with the hair clipper mounted (same tool as SO-101, user 2026-10-07; earlier: tool-free). Never enables on service startup."""
import json,time,threading
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from pnd_tracking import PNDTracker,fk_jacobian
from piper_hardware import PiperHardware,MockHardware
from phone_pose import CAMERA_TO_BASE,select_phone_pose,rotation_step
from follow_filter import FollowFilter
from experiment_log import ExperimentLog
from phone_motion_filter import PhoneMotionFilter

class PhoneTrackingLost(RuntimeError):pass
class PhoneTrackingGap(RuntimeError):pass

# Piper firmware joint ranges (piper_sdk JointCtrl table, degrees). Commands outside them are not executed,
# e.g. the folded rest pose reads J2≈-1.8°, J3≈+1.4°, so homing commands must be clamped into range.
FIRMWARE_LIMITS_DEG=np.array([[-150.,150.],[0.,180.],[-170.,0.],[-100.,100.],[-70.,70.],[-120.,120.]])
HOMING_LEASH_DEG=4.0      # homing command may run at most this far ahead of the measured joints
HOMING_STALL_S=3.0        # fault only if the arm makes no measurable progress for this long
# Start and Finish share the validated initial pose; keep motors holding.
from initial_pose import INITIAL_JOINTS_DEG
REST_JOINTS=np.array(INITIAL_JOINTS_DEG,float)
REST_TOLERANCE_DEG=0.5
# Stop Ending (user decision 2026-10-07): initial pose -> folded pose (mechanical zero) -> release motors.
FOLD_JOINTS=np.zeros(6)
FOLD_TOLERANCE_DEG=2.0   # folded pose sits on mechanical stops; accept within 2 deg (or after a 5 s stall check)

def clamp_firmware(q):
    return np.clip(np.asarray(q,float),FIRMWARE_LIMITS_DEG[:,0],FIRMWARE_LIMITS_DEG[:,1])

class RobotExperiment:
    def __init__(self,root=None,mirror=None):
        self.lock=threading.RLock();self.root=Path(root or Path(__file__).parent/'results/experiments');self.mirror=mirror
        self.backend=None;self.owner=None;self.mode=None;self.paused=True;self.fault=False;self.reason='Disconnected. Clipper-mounted bench control only.';self.latest=None;self.last=0;self.config=None
        self.ref=None;self.ref_r=None;self.axis=None;self.home=None;self.rotation=None;self.command=None;self.command_r=None;self.initial_home=None;self.initial_r=None
        self.last_pose_log=0;self.head_alignment=None;self.phone_filter=PhoneMotionFilter();self.filtered_latest=None;self.filter_info=None
        self.scale=3.0;self.log=None;self.feedback=None;self.info=None;self.plots=None;self.report=None;self.plot_error=None;self.plotting=False;self.plot_threads=[]
        self.preparing=False;self.initial_ready=False;self.prepare_path=[];self.prepare_guard=None;self.prepare_until=0
        # Guided workflow: idle → connected → homing → head_calibration → haircut → resting → rested (rest_failed on alert)
        self.workflow='idle';self.rest_mode=False
        self.pose_timeout=.8;self.tracking_pause_timeout=2.;self.tracking_hold=False;self.last_input=0;self.input_issue=None;self.stable_frames=0;self.resume_pending=False;self.resume_request=None
        limits=[[-148,148],[0,178],[-168,0],[-98,98],[-68,68],[-118,118]]
        self.follow_speed=100.;self.follow_joint_speed=45.;self.follow_orientation_speed=30.  # user 2026-10-07: faster bench following (was 60 mm/s, 30 deg/s, 20 deg/s)
        # Leash (user request 2026-10-07): the phone target may lead the arm by at most this much (robot mm).
        # Faster phone motion is dropped instead of queued, so the arm never chases an old backlog.
        self.follow_leash_mm=float('inf');  # user 2026-10-07: leash off for now (set 40. to re-enable)
        self.leash_slips=0;self.last_leash_log=0.
        self.tracker=PNDTracker(speed=self.follow_joint_speed,limits=limits,max_cartesian_speed=self.follow_speed,max_orientation_speed=self.follow_orientation_speed);self.filter=FollowFilter();self.alive=True
        self.thread=threading.Thread(target=self._loop,daemon=True);self.thread.start()
    def feed(self,result,data):
        with self.lock:
            if self.owner!=data.get('client_id'):return
            if self.preparing:return
            self.last_input=time.monotonic();self.input_issue=None
            config=json.dumps([data.get(k) for k in ('camera_id','tracking_mode','board','tag_mm','calibration','width','height')],sort_keys=True)
            if self.config and config!=self.config:
                self.pause('Camera or calibration changed. Resume to set a new reference.');self.latest=None;self.last=0;self.stable_frames=0
            self.config=config
            if data.get('moving')!='phone' or not result.get('calibrated') or result['reprojection_px']>3:
                self.stable_frames=0;self.input_issue='Uncalibrated, wrong mode or reprojection error above 3 px';return
            # Four-tag board (user 2026-10-07): with fewer than 3 tags the planar pose flips by 100-700 mm. Do not
            # steer with such frames; treat them like a short tracking gap (arm holds) instead of counting as jumps.
            used=len(result.get('board_ids') or []);board=data.get('tracking_mode')=='board'
            if board and used<3:
                self.stable_frames=0;self.input_issue=f'Only {used} board tag(s) usable; keep at least 3 tags in view';return
            p,r,_=select_phone_pose(result)
            now=time.monotonic()
            if not self.paused and self.last and now-self.last>self.tracking_pause_timeout:
                self.pause('Phone tracking lost for 2 s. Arm held; Resume manually after tracking returns.')
            continuous=self.latest is not None and now-self.last<=self.pose_timeout and np.linalg.norm(p-self.latest[0])<=20 and Rotation.from_matrix(r@self.latest[1].T).magnitude()<=np.deg2rad(5)
            if self.paused or not self.last or now-self.last>self.pose_timeout:self.phone_filter.reset()
            filtered_p,filtered_r,info=self.phone_filter.update(p,r,now,self.scale,bool(result.get('ambiguous')) and not (board and used>=3))
            self.filter_info=info
            if not info['accepted']:
                self.stable_frames=0;self.input_issue='Pose outlier rejected'
                if self.log and not self.log.closed:self.log.event('phone_pose_rejected',raw_camera_mm=p.tolist(),raw_camera_R=r.tolist(),ambiguous=bool(result.get('ambiguous')),board_ids=result.get('board_ids'),**info)
                if info['rejected_frames']>=3:self.pause('Repeated pose jumps. Arm held; restore stable tracking, then press Resume.')
                return
            self.stable_frames=self.stable_frames+1 if continuous else 1
            if self.last and 0<now-self.last<1.:self.sample_interval=.8*getattr(self,'sample_interval',.1)+.2*(now-self.last)
            self.latest=(p,r);self.filtered_latest=(filtered_p,filtered_r);self.last=now
            if self.log and not self.log.closed and self.last-self.last_pose_log>.1:
                self.last_pose_log=self.last;self.log.event('phone_pose',camera_mm=p.tolist(),camera_R=r.tolist(),filtered_camera_mm=filtered_p.tolist(),filtered_camera_R=filtered_r.tolist(),filter=info,raw_camera_mm=result.get('raw_camera_t_mm'),board_ids=result.get('board_ids'),reprojection_px=result['reprojection_px'],ambiguous=result.get('ambiguous'))
    def pause(self,reason='Paused. Reposition phone, then Resume to set a new reference.'):
        self.preparing=False
        self.resume_pending=False;self.resume_request=None
        self.paused=True;self.tracking_hold=False;self.reason=reason;self.tracker.reset()
        self.phone_filter.reset();self.filtered_latest=None
        if self.backend:self.backend.hold()
        if self.log and not self.log.closed:self.log.event('pause',reason=reason)
    def action(self,data):
        with self.lock:
            action=data['action'];owner=data.get('client_id')
            if not isinstance(owner,str) or not owner:raise ValueError('Missing control session.')
            if self.backend and self.owner!=owner:
                if action in ('stop','emergency'):
                    pass # Any page may stop; never allow it to initiate motion.
                elif action=='connect' and self.paused and not self.preparing:
                    previous=self.owner;self.owner=owner;self.latest=None;self.filtered_latest=None;self.last=0;self.config=None;self.stable_frames=0
                    self.resume_pending=False;self.resume_request=None;self.tracking_hold=False;self.phone_filter.reset();self.tracker.reset()
                    self.reason='Paused connection transferred to this page. Start or Resume explicitly.'
                    if self.log and not self.log.closed:self.log.event('control_session_transferred',previous_client=previous,client_id=owner)
                else:raise ValueError('Another page controls the moving arm. Pause it before connecting from this page.')
            if action=='connect':
                if self.backend:return self.state()
                mode=data.get('backend','mock')
                if mode not in ('mock','hardware'):raise ValueError('Select Mock or Piper.')
                backend=MockHardware() if mode=='mock' else PiperHardware('0')
                feedback=backend.connect();self.backend=backend;self.feedback=feedback;self.mode=mode;self.owner=owner
                from initialization_path import WORKING_JOINTS
                self.initial_ready=bool(np.max(np.abs(feedback['q']-WORKING_JOINTS))<.5)
                self.initial_home=feedback['p'].copy();self.initial_r=feedback['r'].copy();self.paused=True;self.fault=False;self.latest=None;self.last=0
                self.reason='Connected. Confirm only the clipper is mounted and the workspace is clear, then Start.';self.workflow='connected';self.rest_mode=False
            elif action in ('prepare','return_initial'):
                if not self.backend:raise ValueError('Connect the arm first.')
                if data.get('bench_clear') is not True:raise ValueError('Confirm only the clipper is mounted and no people are in the workspace before homing.')
                if self.preparing:return self.state()
                self.pause('Checking home path.')
                from initialization_path import initialization_plan
                f=self.backend.read();self.backend.healthy()
                tf,_=fk_jacobian(np.deg2rad(f['q']))
                if np.linalg.norm(tf[:3,3]*1000-f['p'])>15 or Rotation.from_matrix(tf[:3,:3]@f['r'].T).magnitude()>np.deg2rad(10):raise ValueError('SDK and model pose disagree. Homing refused.')
                try:path,guard=initialization_plan(f['q'],FIRMWARE_LIMITS_DEG if self.mode=='hardware' else self.tracker.solver.p.joint_limits_deg,self.head_alignment,enforce_head=self.mode!='hardware')
                except ValueError as e:self.reason=str(e);raise
                try:
                    if action=='prepare' or not self.backend.enabled:self.backend.enable()
                    elif self.mode=='hardware':self.backend.healthy(require_enabled=True)
                except Exception as e:
                    self.reason=str(e);self.fault=True;self.paused=True;self.preparing=False
                    raise
                if self.log is None or self.log.closed:self.log=ExperimentLog(self.root,self.mode);self.plots=None;self.plot_error=None
                self.prepare_path=path;self.prepare_guard=guard;self.prepare_until=time.monotonic()+60
                self.workflow='homing';self.rest_mode=False
                # Homing integrates its own command trajectory (not measured feedback + tiny step, which never
                # advances when the arm does not respond), starting from the measured pose clamped into firmware range.
                self.prepare_cmd=clamp_firmware(f['q']);self.prepare_progress=None
                self.preparing=True;self.initial_ready=False;self.fault=False;self.latest=None;self.last=0
                self.stable_frames=0
                self.reason='Returning to saved initial pose at up to 15 deg/s. Phone control locked; arrival leaves the arm paused.'
                self.log.event('initialization_start',trigger=action,initial_joints_deg=list(INITIAL_JOINTS_DEG),actual_joints_deg=f['q'].tolist(),waypoints=[v.tolist() for v in path],speed_deg_s=15,guard_scope='physical workspace checked by operator; joint limits retained' if self.mode=='hardware' else 'simulation head clearance')
            elif action=='rest':
                if not self.backend or not self.backend.enabled:raise ValueError('Connect and enable the arm first.')
                if data.get('bench_clear') is not True:raise ValueError('Confirm only the clipper is mounted and no people are in the workspace before moving to rest.')
                if self.preparing:raise ValueError('Homing in progress. Pause first, then Finish & Rest.')
                self.pause('Finishing: returning to the initial pose.')
                from initialization_path import initialization_plan
                f=self.backend.read();self.backend.healthy()
                limits=FIRMWARE_LIMITS_DEG if self.mode=='hardware' else self.tracker.solver.p.joint_limits_deg
                path,guard=initialization_plan(f['q'],limits,self.head_alignment,enforce_head=self.mode!='hardware')
                if self.log is None or self.log.closed:self.log=ExperimentLog(self.root,self.mode);self.plots=None;self.plot_error=None
                self.prepare_path=[np.asarray(w,float) for w in path];self.prepare_guard=guard
                self.prepare_until=time.monotonic()+120;self.prepare_cmd=clamp_firmware(f['q']);self.prepare_progress=None
                self.rest_target=REST_JOINTS.copy();self.rest_tol=REST_TOLERANCE_DEG
                self.preparing=True;self.rest_mode=True;self.rest_save=True;self.initial_ready=False;self.workflow='resting';self.latest=None;self.last=0;self.stable_frames=0
                self.reason='Finishing: returning to the initial pose at up to 15 deg/s. Phone control locked.'
                self.log.event('rest_start',actual_joints_deg=f['q'].tolist(),waypoints=[np.asarray(v).tolist() for v in self.prepare_path],rest_joints_deg=REST_JOINTS.tolist(),speed_deg_s=15)
            elif action=='enable':
                if self.preparing:raise ValueError('Homing in progress. Pause or Stop to interrupt.')
                if not self.backend:raise ValueError('Connect the arm first.')
                if data.get('bench_clear') is not True:raise ValueError('Confirm only the clipper is mounted and the workspace is clear.')
                f=self.backend.read();limits=np.asarray(self.tracker.solver.p.joint_limits_deg)
                outside=np.where((f['q']<limits[:,0])|(f['q']>limits[:,1]))[0]
                if len(outside):raise ValueError('Prepare a valid working pose first:'+ '; '.join(f'J{i+1}={f["q"][i]:.2f}°, allowed range {limits[i,0]:g} to {limits[i,1]:g}°' for i in outside)+'. Mechanical zero is unchanged.')
                self.backend.enable();self.initial_home=f['p'].copy();self.initial_r=f['r'].copy();self.fault=False;self.reason='Enabled and paused. Keep Tags visible, then Resume.'
            elif action=='clutch':
                if self.preparing:self.pause('Homing interrupted. Holding position; press Start to complete homing.');return self.state()
                if self.resume_pending:self.pause('Resume cancelled. Arm held.');return self.state()
                if not self.backend or not self.backend.enabled:raise ValueError('Enable the arm first.')
                if not self.initial_ready:raise ValueError('Complete homing before phone control.')
                if not self.paused:self.pause();return self.state()
                if self.fault:raise ValueError('Resolve the fault, then Enable. Motion will not resume automatically.')
                if data.get('bench_clear') is not True:raise ValueError('Clipper-mounted bench control only; keep people outside the workspace.')
                if self.latest is None or time.monotonic()-self.last>self.pose_timeout or self.stable_frames<3:
                    self.resume_pending=False;self.resume_request=None
                    self.reason='Tracking not stable yet. Hold phone still with Tags visible, then press Resume again.'
                    return self.state()
                f=self.backend.read();self.backend.healthy()
                q=f['q'];limits=np.asarray(self.tracker.solver.p.joint_limits_deg)
                if np.any(q<limits[:,0]) or np.any(q>limits[:,1]):raise ValueError('Measured joints outside follow limits. Check model and zero.')
                tf,_=fk_jacobian(np.deg2rad(q))
                if np.linalg.norm(tf[:3,3]*1000-f['p'])>15 or Rotation.from_matrix(tf[:3,:3]@f['r'].T).magnitude()>np.deg2rad(10):raise ValueError('SDK end pose and local FK disagree. Check model and zero.')
                if self.log is None or self.log.closed:
                    self.log=ExperimentLog(self.root,self.mode);self.plots=None;self.plot_error=None
                # Control point = clipper tip (user 2026-10-07, same as SO-101): home/command/target are tip positions.
                from head_alignment import tool_offset_mm
                self.tool_mm=tool_offset_mm()
                self.home=f['p']+f['r']@self.tool_mm;self.rotation=f['r'].copy();self.ref=self.latest[0].copy();self.ref_r=self.latest[1].copy();self.axis=CAMERA_TO_BASE@self.ref_r.T
                self.phone_filter.reset();fp,fr,_=self.phone_filter.update(*self.latest,self.last,self.scale)
                self.filtered_latest=(fp,fr)
                self.command=self.home.copy();self.command_r=self.rotation.copy();self.filter.reset(self.home,self.rotation);self.tracker.reset();self.paused=False
                self.reason='PND following. No fixed travel envelope; joint limits and speed limits remain active.'
                self.log.event('resume_anchor',phone_mm=self.ref.tolist(),actual_mm=self.home.tolist(),axis_map=self.axis.tolist(),scale=self.scale,control_point='clipper tip' if np.any(self.tool_mm) else 'flange',tool_offset_mm=self.tool_mm.tolist())
                self.log.event('configuration',configuration=json.loads(self.config),mode=self.mode,speed_deg_s=self.follow_joint_speed,cartesian_speed_mm_s=self.follow_speed,orientation_speed_deg_s=self.follow_orientation_speed,hardware_speed_percent=30,input_filter='adaptive low-pass plus continuous position/SO3 hysteresis before gain; max 1 mm robot translation band and 0.35 deg rotation band; outlier gate; manual Resume only',head_registered=False)
            elif action=='sensitivity':
                value=float(data['step'])
                if not np.isfinite(value) or not .15<=value<=20:raise ValueError('Scale range: 10:1.5–10:200.')
                if not self.paused:self.pause('Scale changed. Resume to set a new reference.')
                self.scale=value
            elif action=='stop':
                # User request 2026-10-07: Stop & Save returns the arm to the initial pose (same checked, 15 deg/s
                # path as Finish & Rest) and saves when it arrives, instead of stopping wherever it is. Only when the
                # arm cannot move safely (not connected/enabled, fault, homing in progress) does it stop in place.
                if self.backend and self.backend.enabled and not self.fault and not self.preparing and self.owner==owner and data.get('bench_clear') is True:
                    try:
                        state=self.action(dict(data,action='rest'));self.rest_save=False  # user 2026-10-07: Stop does not save
                        # then fold to the mechanical-zero pose before releasing the motors
                        self.prepare_path.append(FOLD_JOINTS.copy());self.rest_target=FOLD_JOINTS.copy();self.rest_tol=FOLD_TOLERANCE_DEG
                        self.reason='Stop Ending: returning to the initial pose, then folding; motors are released at the folded pose.'
                        state=self.state()
                        return state
                    except Exception as e:
                        if self.log and not self.log.closed:self.log.event('stop_return_failed',error=str(e))
                self.pause('Stopped. Data kept in this session; press Save Data to write figures.')
            elif action=='save':
                # Separate Save (user 2026-10-07): pause following if needed, close this session's log and generate
                # figures. The arm keeps holding; a new session log starts automatically on the next motion.
                if not self.paused:self.pause('Paused to save data.')
                if self.log and not self.log.closed:self.finish_log();self.reason='Data saved; figures are being generated. Arm still holding.'
                else:self.reason='Nothing new to save.'
            elif action=='emergency':
                if self.backend:self.backend.emergency()
                self.paused=True;self.fault=True;self.reason='Software stop requested. Physical E-stop remains available.'
                if self.log:self.log.event('emergency')
            elif action=='disconnect':
                self.pause('Disconnected.');self.finish_log()
                if self.backend:self.backend.disconnect()
                self.backend=None;self.owner=None;self.latest=None;self.workflow='idle';self.rest_mode=False
            else:raise ValueError('Unknown arm action.')
            return self.state()
    def _finish_rest(self,f):
        """Called from the control loop when the rest waypoint is reached."""
        self.preparing=False;self.rest_mode=False;self.paused=True
        target=getattr(self,'rest_target',REST_JOINTS);tol=getattr(self,'rest_tol',REST_TOLERANCE_DEG)
        err=float(np.max(np.abs(f['q']-target)))
        if err<=tol:
            self.backend.hold()
            self.initial_ready=True;self.fault=False
            self.initial_home=f['p'].copy();self.initial_r=f['r'].copy()
            saving=getattr(self,'rest_save',True)
            self.workflow='rested' if saving else 'stopped';self.reason=f'Target pose reached (max joint error {err:.2f} deg). Motors enabled and holding.'+(' Session saved.' if saving else '')
            if self.log and not self.log.closed:self.log.event('rest_complete',actual_joints_deg=f['q'].tolist(),max_error_deg=err,motors_disabled=False,initial_joints_deg=REST_JOINTS.tolist())
            if not saving:
                # Stop Ending (user decision 2026-10-07): back at the initial pose, release the motors because the
                # operator is about to power the arm off. Without holding torque the arm sinks under gravity.
                try:
                    self.backend.disable();self.initial_ready=False;self.workflow='ended'
                    self.reason='Stop Ending: folded pose reached, motors released. Ready to power off.'
                    if self.log and not self.log.closed:self.log.event('stop_ending_released',actual_joints_deg=f['q'].tolist())
                except Exception as e:
                    self.workflow='rest_failed';self.reason=f'Stop Ending: motor release not confirmed ({e}). Motors may still be holding.'
                    if self.log and not self.log.closed:self.log.event('stop_ending_release_failed',error=str(e))
        else:
            self.workflow='rest_failed';self.reason=f'ALERT: rest pose not reached (max joint error {err:.2f} deg). Motors kept enabled and holding.'
            if self.log and not self.log.closed:self.log.event('rest_not_reached',actual_joints_deg=f['q'].tolist(),max_error_deg=err)
        if getattr(self,'rest_save',True):self.finish_log()
    def finish_log(self):
        if not self.log or self.log.closed:return
        self.log.close();self.plotting=True;self.report=None;path=self.log.path;mode=self.mode;started=self.log.wall_started;alignment=self.head_alignment
        def run():
            # Run figures/report in a separate process (figure_worker.py): in-process matplotlib starved the CAN
            # receive thread (GIL), feedback looked stale and the arm was e-stopped right after Save.
            import subprocess,sys,tempfile,os
            try:
                with tempfile.NamedTemporaryFile('w',suffix='.json',delete=False,encoding='utf-8') as fh:
                    json.dump(dict(path=str(path),mode=mode,started=started,alignment=alignment),fh,default=str);args_file=fh.name
                try:
                    worker=Path(__file__).resolve().parent/'figure_worker.py'
                    flags=getattr(subprocess,'CREATE_NO_WINDOW',0)|getattr(subprocess,'BELOW_NORMAL_PRIORITY_CLASS',0)
                    proc=subprocess.run([sys.executable,'-B',str(worker),args_file],capture_output=True,text=True,timeout=900,creationflags=flags)
                    lines=[l for l in proc.stdout.splitlines() if l.startswith('{')]
                    if not lines:raise RuntimeError('figure worker failed: '+(proc.stderr or '')[-500:])
                    out=json.loads(lines[-1])
                    if out.get('error'):raise RuntimeError(out['error'])
                    self.plots=out.get('plots');self.report=out.get('report')
                finally:os.unlink(args_file)
            except Exception as e:self.plot_error=str(e)
            finally:self.plotting=False
        t=threading.Thread(target=run,daemon=True);self.plot_threads.append(t);t.start()
    def _loop(self):
        last=time.monotonic()
        while self.alive:
            started=time.monotonic();dt=min(started-last,.04);last=started
            with self.lock:
                try:
                    if self.backend:
                        f=self.backend.read();self.feedback=f
                        if self.mirror:self.mirror(f['q'])
                        if self.preparing:
                            if time.monotonic()>self.prepare_until:
                                if self.rest_mode:self.workflow='rest_failed'
                                raise RuntimeError('Homing timed out. Holding position.')
                            self.backend.healthy();target_q=self.prepare_path[0]
                            cmd=getattr(self,'prepare_cmd',None)
                            if cmd is None:cmd=self.prepare_cmd=clamp_firmware(f['q'])
                            lag=float(np.max(np.abs(f['q']-cmd)))
                            ref=getattr(self,'prepare_progress',None)
                            if ref is None or np.max(np.abs(f['q']-ref[0]))>.3:self.prepare_progress=ref=(f['q'].copy(),time.monotonic())
                            elif lag>1. and time.monotonic()-ref[1]>HOMING_STALL_S:raise RuntimeError(f'Arm is not moving: no joint progress for {HOMING_STALL_S:.0f} s while commanded {lag:.1f} deg away. Holding; check enable state, E-stop, teach mode and CAN.')
                            if self.rest_mode and len(self.prepare_path)==1 and np.max(np.abs(cmd-target_q))<1e-6:
                                # Rest command fully issued: if the arm stops short for 5 s, finish with an alert (motors stay on)
                                if getattr(self,'rest_cmd_done_at',None) is None:self.rest_cmd_done_at=time.monotonic()
                                if np.max(np.abs(f['q']-target_q))>=.25 and time.monotonic()-self.rest_cmd_done_at>5:
                                    self.prepare_path=[];self._finish_rest(f);continue
                            else:self.rest_cmd_done_at=None
                            if np.max(np.abs(f['q']-target_q))<.25:
                                self.prepare_path.pop(0)
                                if not self.prepare_path and self.rest_mode:
                                    self._finish_rest(f)
                                elif not self.prepare_path:
                                    self.preparing=False;self.initial_ready=True;self.initial_home=f['p'].copy();self.initial_r=f['r'].copy()
                                    if self.head_alignment is None:
                                        self.workflow='head_calibration'
                                        self.reason='Home reached. Head calibration: Resume, guide the clipper tip 10 mm above the crown center, then press Align Head.'
                                    else:
                                        self.workflow='haircut';self.reason='Home reached. Keep Tags visible, then Resume.'
                                    self.log.event('initialization_complete',actual_joints_deg=f['q'].tolist())
                            else:
                                from initialization_path import initialization_step
                                # Leash: never command more than HOMING_LEASH_DEG ahead of the measured joints
                                next_q=clamp_firmware(np.clip(initialization_step(cmd,target_q,dt),f['q']-HOMING_LEASH_DEG,f['q']+HOMING_LEASH_DEG))
                                # 25 ms was too tight: a busy CPU (e.g. MuJoCo window starting) made the first check run out of time,
                                # which counts as unsafe. Homing is slow (15 deg/s), so allow 200 ms; a real collision still fails.
                                check_deadline=time.monotonic()+.2
                                if self.prepare_guard is not None and not self.prepare_guard.edge_safe(f['q'],next_q,deadline=check_deadline):
                                    # A sender-side controller fault must not be hidden by
                                    # a simultaneous planner timeout or clearance failure.
                                    tx_fault=getattr(self.backend,'tx_fault',None)
                                    if tx_fault:raise RuntimeError(tx_fault)
                                    if self.mode=='hardware':self.backend.healthy(require_enabled=True)
                                    before=self.prepare_guard.clearance(f['q']);after=self.prepare_guard.clearance(next_q)
                                    timed_out=time.monotonic()>check_deadline
                                    reason=('Home path verification timed out; no motion issued.' if timed_out else
                                            f"Home path clearance rejected: measured {before['distance_mm']:.1f} mm, commanded {after['distance_mm']:.1f} mm; required 10 mm.")
                                    if self.log:self.log.event('home_path_rejected',reason=reason,actual_joints_deg=f['q'].tolist(),command_joints_deg=next_q.tolist(),start_clearance=before,end_clearance=after)
                                    raise RuntimeError(reason+' Holding position.')
                                self.backend.send(next_q);self.prepare_cmd=next_q
                                tf,_=fk_jacobian(np.deg2rad(target_q));self.log.log(tf[:3,3]*1000,tf[:3,:3],next_q,f,dict(solve_ms=0,qp_iters=0,phase='joint_initialization'))
                        elif not self.paused:
                            age=time.monotonic()-self.last
                            if age>self.tracking_pause_timeout:raise PhoneTrackingLost('Phone tracking lost for 2 s. Arm held; Resume manually after tracking returns.')
                            if age>self.pose_timeout or self.tracking_hold and self.stable_frames<3:
                                if not self.tracking_hold:
                                    self.backend.hold();self.tracking_hold=True;self.stable_frames=0
                                    self.command=f['p']+f['r']@getattr(self,'tool_mm',np.zeros(3));self.command_r=f['r'].copy();self.filter.reset(self.command,self.command_r);self.tracker.reset()
                                    self.reason='Waiting for fresh phone tracking. Arm held; short-gap recovery needs 3 stable frames.'
                                    if self.log and not self.log.closed:self.log.event('tracking_gap_hold',pose_age_s=age)
                                # Do not advance the old target or drain queued motion.
                                raise PhoneTrackingGap()
                            if self.tracking_hold:
                                self.tracking_hold=False;self.reason='PND following. Tracking recovered after a brief gap.'
                                if self.log and not self.log.closed:self.log.event('tracking_gap_recovered',stable_frames=self.stable_frames)

                            self.backend.healthy();p,r=self.filtered_latest or self.latest;target=self.home+self.axis@(p-self.ref)*self.scale
                            target_r=self.axis@(r@self.ref_r.T)@self.axis.T@self.rotation
                            self.filter.tau=float(np.clip(.6*getattr(self,'sample_interval',.1),.035,.3));goal=self.filter.position(target,dt)
                            lead=goal-self.command;lead_n=float(np.linalg.norm(lead))
                            if lead_n>self.follow_leash_mm:
                                # drop the excess: shift the phone->robot mapping so the target sits exactly on the leash
                                excess=lead*(1-self.follow_leash_mm/lead_n);self.home=self.home-excess;target=target-excess
                                self.filter.p=self.filter.p-excess
                                if getattr(self.filter,'p1',None) is not None:self.filter.p1=self.filter.p1-excess
                                goal=goal-excess;self.leash_slips+=1
                                if time.monotonic()-self.last_leash_log>.5 and self.log and not self.log.closed:
                                    self.last_leash_log=time.monotonic();self.log.event('leash_slip',dropped_mm=float(np.linalg.norm(excess)),leash_mm=self.follow_leash_mm,total=self.leash_slips)
                            prev_command,prev_command_r=self.command.copy(),self.command_r.copy()
                            step=goal-self.command;n=np.linalg.norm(step)
                            self.command+=step*min(1,self.follow_speed*dt/max(n,1e-12));self.command_r=rotation_step(self.command_r,self.filter.orientation(target_r,dt),np.deg2rad(self.follow_orientation_speed)*dt)
                            # Tip command -> flange command for the flange-level PND. Offset rotated by the MEASURED flange
                            # orientation (as SO-101): orientation is a soft task, so this keeps the tip on target even
                            # while orientation lags.
                            tool_world=f['r']@getattr(self,'tool_mm',np.zeros(3))
                            next_q,_,_,info=self.tracker.step(self.command-tool_world,self.command_r,f['q'],dt);self.info=info
                            if not info['converged']:raise RuntimeError('PND did not converge. Arm held; check solver settings.')
                            # Head guard while following (user 2026-10-07): check the commanded joints against the MuJoCo
                            # head (aligned by Align Head) with gripper + clipper; a step that would bring them within
                            # 10 mm is not sent (or is cut short at 10 mm). The arm waits; the next phone frame with a
                            # safe target is executed normally. Moving away from the head is always allowed.
                            guard=self._follow_guard()
                            if guard is not None:
                                safe_q,safety=guard.filter_step(f['q'],next_q);self.head_safety=safety
                                if safety['blocked']:
                                    if np.max(np.abs(safe_q-f['q']))<1e-6:
                                        # skip this frame: no motion, and do not let the command run on behind the head
                                        self.command,self.command_r=prev_command,prev_command_r;self.tracker.reset()
                                        self.reason=f"Head guard: target {safety['distance_mm']:.0f} mm from head ({safety['part']}); waiting for a safe phone target."
                                        if time.monotonic()-getattr(self,'last_guard_log',0)>.5 and self.log and not self.log.closed:
                                            self.last_guard_log=time.monotonic();self.log.event('head_guard_block',**safety,command_joints_deg=next_q.tolist(),actual_joints_deg=f['q'].tolist())
                                        raise PhoneTrackingGap()
                                    next_q=safe_q;self.tracker.reset()
                                    self.reason='Head guard: stopped 10 mm from the head; move the phone away to continue.'
                                elif self.reason.startswith('Head guard'):
                                    self.reason='PND following. No fixed travel envelope; joint limits and speed limits remain active.'
                            self.backend.send(next_q,speed_rate=30)
                            self.log.log(target-tool_world,target_r,next_q,f,info)  # CSV stays flange-referenced
                except PhoneTrackingGap:
                    pass
                except PhoneTrackingLost as e:
                    self.pause(str(e))
                except Exception as e:
                    if not self.paused or self.preparing:
                        self.fault=True
                        if self.log and getattr(self.backend,'controller_fault',None):self.log.event('controller_fault',snapshot=self.backend.controller_fault,transmission=self.backend.transmission_state() if hasattr(self.backend,'transmission_state') else None)
                        self.pause(str(e))
                    self.reason=str(e)
            time.sleep(max(.001,.02-(time.monotonic()-started)))
    def _follow_guard(self):
        """HeadGuard for phone following, rebuilt when the head alignment changes. None if it cannot be built."""
        revision=(self.head_alignment or {}).get('revision')
        if getattr(self,'_guard_revision','unset')!=revision:
            self._guard_revision=revision
            try:
                from head_guard import HeadGuard
                from sim_bridge import PROJECTS
                self._guard=HeadGuard(PROJECTS/'mujoco_haircut',flange=True,alignment=self.head_alignment)
            except Exception as e:
                self._guard=None
                if self.log and not self.log.closed:self.log.event('head_guard_unavailable',error=str(e))
        return getattr(self,'_guard',None)
    def align_head(self,data,simulation):
        with self.lock:
            if not self.backend or self.owner!=data.get('client_id'):raise ValueError('Connect from this page before alignment.')
            if self.preparing:raise ValueError('Wait for homing to finish before Align Head.')
            if not self.paused or self.resume_pending:self.pause('Head alignment requested. Arm held; Resume continues from this position.')
            feedback=self.backend.read();self.backend.healthy()
            time.sleep(.1)
            current=self.backend.read();self.backend.healthy()
            if np.max(np.abs(current['q']-feedback['q']))>.2:raise ValueError('Hold the arm still at the crown reference, then Align Head.')
            feedback=current
            tf,_=fk_jacobian(np.deg2rad(feedback['q']))
            if np.linalg.norm(tf[:3,3]*1000-feedback['p'])>15:raise ValueError('SDK and model flange positions disagree.')
            record=simulation.align_head(self.owner,feedback['p'],feedback['q'])
            self.head_alignment=record;self.feedback=feedback
            if self.workflow in ('head_calibration','homing','connected'):self.workflow='haircut'
            self.reason='Head aligned. Resume to start the haircut; Finish & Rest when done.'
            if self.log and not self.log.closed:self.log.event('head_alignment',**record)
            return dict(alignment=record,robot=self.state())

    def state(self):
        with self.lock:
            f=self.feedback
            return dict(workflow=self.workflow,controller_id=self.owner,initial_joints_deg=list(INITIAL_JOINTS_DEG),rest_joints_deg=REST_JOINTS.tolist(),connected=self.backend is not None,backend=self.mode,enabled=bool(self.backend and self.backend.enabled),paused=self.paused,preparing=self.preparing,initial_ready=self.initial_ready,resume_pending=self.resume_pending,stable_frames=self.stable_frames,pose_timeout_s=self.pose_timeout,tracking_pause_timeout_s=self.tracking_pause_timeout,tracking_hold=self.tracking_hold,input_age_s=time.monotonic()-self.last_input if self.last_input else None,input_issue=self.input_issue,pose_age_s=time.monotonic()-self.last if self.last else None,fault=self.fault,reason=self.reason,scale=self.scale,
                        joints_deg=f['q'].tolist() if f else None,position_mm=f['p'].tolist() if f else None,feedback_age_s=f['age'] if f else None,solver='PND box QP',solver_info=self.info,following_limits=dict(position_mm_s=self.follow_speed,joint_deg_s=self.follow_joint_speed,orientation_deg_s=self.follow_orientation_speed),
                        session_dir=str(self.log.root) if self.log else None,plotting=self.plotting,plot_error=self.plot_error,plots=str(self.plots) if self.plots else None,report=str(self.report) if self.report else None,head_registered=False,head_alignment=self.head_alignment,input_filter=self.filter_info,controller=f.get('controller') if f else None,controller_fault=getattr(self.backend,'controller_fault',None),transmission=self.backend.transmission_state() if hasattr(self.backend,'transmission_state') else None)
    def close(self):
        with self.lock:
            self.pause('Service stopped.');self.finish_log()
            if self.backend:self.backend.disconnect();self.backend=None
            self.alive=False
        self.thread.join(timeout=2)
        for t in self.plot_threads:t.join(timeout=30)
