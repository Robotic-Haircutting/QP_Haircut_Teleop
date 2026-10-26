"""Read-first gs_usb adapter. No enable, recovery or gripper command in connect."""
import os,time,threading
import numpy as np
from scipy.spatial.transform import Rotation

TX_PERIOD_S=0.02  # CAN command stream period (50 Hz)
COMM_FAULT_DEBOUNCE_S=0.3  # joint-communication flag must persist this long before motion is stopped

class PiperHardware:
    mode='hardware'
    def __init__(self,channel='0'):
        self.channel=channel;self.arm=None;self.enabled=False;self.sent=0;self.received={};self.estopped=False;self.controller_fault=None;self.comm_fault_since=None
        self.tx_lock=threading.RLock();self.tx_stop=threading.Event();self.tx_thread=None;self.tx_command=None;self.tx_stamp=0.;self.tx_speed=10;self.tx_fault=None;self.tx_last=0.;self.tx_max_gap=0.;self.tx_stale_holds=0
    def connect(self):
        import can,libusb_package
        from piper_sdk import C_PiperInterface_V2
        from piper_sdk.hardware_port import C_STD_CAN
        directory=os.path.dirname(str(libusb_package.get_library_path()))
        os.environ['PATH']=directory+os.pathsep+os.environ.get('PATH','')
        index=int(self.channel)
        from can.interfaces.gs_usb import GsUsbBus
        class ReleasableBus(GsUsbBus):
            def shutdown(self):
                # Avoid python-can's shutdown rescan, which opens a second USB
                # handle and leaves it claimed on Windows.
                self._index=None
                try:super().shutdown()
                finally:
                    device=getattr(getattr(self,'gs_usb',None),'gs_usb',None)
                    if device is not None:
                        import usb.util
                        usb.util.dispose_resources(device)
        class USBPort(C_STD_CAN):
            def _create_bus(self):return ReleasableBus(channel=index,index=index,bitrate=1000000)
        self.arm=C_PiperInterface_V2(self.channel,judge_flag=False,can_auto_init=False,start_sdk_joint_limit=False,start_sdk_gripper_limit=False)
        def receive(msg):
            self.arm.ParseCANFrame(msg)
            if msg.arbitration_id in (0x2A1,0x2A2,0x2A3,0x2A4,0x2A5,0x2A6,0x2A7,0x261,0x262,0x263,0x264,0x265,0x266):self.received[msg.arbitration_id]=time.monotonic()
        self.arm._C_PiperInterface_V2__arm_can=USBPort(self.channel,'gs_usb',1000000,False,False,receive)
        try:
            self.arm.ConnectPort(piper_init=False)
            deadline=time.monotonic()+5
            while time.monotonic()<deadline:
                try:return self.read()
                except RuntimeError as e:self.last_read_error=str(e);time.sleep(.03)
            raise RuntimeError('No complete fresh CAN feedback within 5 seconds:'+self.last_read_error)
        except Exception:self.disconnect();raise
    def read(self):
        if self.arm is None:raise RuntimeError('Piper disconnected.')
        # Copy SDK values; Get... returns shared mutable feedback objects.
        joint=self.arm.GetArmJointMsgs();stamp=float(joint.time_stamp)
        q=np.array([getattr(joint.joint_state,f'joint_{i}') for i in range(1,7)],float)/1000
        pose=self.arm.GetArmEndPoseMsgs();ps=float(pose.time_stamp);v=pose.end_pose
        p=np.array([v.X_axis,v.Y_axis,v.Z_axis],float)/1000
        r=Rotation.from_euler('xyz',np.array([v.RX_axis,v.RY_axis,v.RZ_axis])/1000,degrees=True).as_matrix()
        # gs_usb timestamps can be adapter uptime, not UTC. Freshness must use
        # our monotonic receive time for every one of the six feedback frames.
        age=max(time.monotonic()-self.received.get(i,0) for i in (0x2A2,0x2A3,0x2A4,0x2A5,0x2A6,0x2A7))
        if min(stamp,ps)<=0 or not np.isfinite(q).all() or not np.isfinite(p).all() or age>.25 or age<-.1:raise RuntimeError(f'Piper feedback stale or invalid; joint stamp {stamp:g}, pose stamp {ps:g}, frame groups{list(map(hex,self.received))}')
        return dict(q=q,p=p,r=r,age=age,stamp=stamp,pose_stamp=ps,controller=self.controller_state())
    def controller_state(self):
        status=self.arm.GetArmStatus().arm_status
        fields={name:int(getattr(status,name)) for name in ('ctrl_mode','arm_status','mode_feed','teach_status','motion_status','err_code')}
        fields['communication_fault_joints']=[i for i in range(1,7) if getattr(status.err_status,f'communication_status_joint_{i}')]
        fields['angle_limit_joints']=[i for i in range(1,7) if getattr(status.err_status,f'joint_{i}_angle_limit')]
        fields['age_s']=float(time.monotonic()-self.received.get(0x2A1,0))
        low=self.arm.GetArmLowSpdInfoMsgs();drivers=[]
        for i in range(1,7):
            foc=getattr(low,f'motor_{i}').foc_status
            driver={name:bool(getattr(foc,name)) for name in ('driver_enable_status','driver_error_status','stall_status','voltage_too_low','motor_overheating','driver_overcurrent','driver_overheating','collision_status','motor_brake_status') if hasattr(foc,name)}
            driver['voltage_v']=float(getattr(low,f'motor_{i}').vol)/10
            driver['age_s']=float(time.monotonic()-self.received.get(0x260+i,0))
            drivers.append(driver)
        fields['drivers']=drivers
        return fields
    def controller_error(self,controller):
        self.controller_fault=controller
        names={1:'Emergency stop',2:'No solution',3:'Singularity',4:'Target limit',5:'Joint communication fault',6:'Brake not released',7:'Collision',9:'Joint fault'}
        detail=names.get(controller['arm_status'],'Controller fault')
        joints=controller.get('communication_fault_joints',[]) if controller['arm_status']==5 else controller.get('angle_limit_joints',[])
        if joints:detail+=' ('+', '.join(f'J{i}' for i in joints)+')'
        return RuntimeError(f"{detail}: controller status {controller['arm_status']}, error {controller['err_code']}. Motion stopped.")
    def healthy(self,require_enabled=False):
        if require_enabled:
            controller=self.controller_state()
            if controller['age_s']>.5:raise RuntimeError('Controller status is stale; following stopped.')
            if controller['arm_status']==5:
                # User decision 2026-10-07 (option B): match the working SO-101 Piper backend. The joint-communication
                # flag (status 5) appears briefly when motion starts on this arm; stop only if it persists longer than
                # COMM_FAULT_DEBOUNCE_S. Driver enable/error/stall and feedback-freshness checks still run every cycle.
                if self.comm_fault_since is None:self.comm_fault_since=time.monotonic()
                if time.monotonic()-self.comm_fault_since>COMM_FAULT_DEBOUNCE_S:raise self.controller_error(controller)
            elif controller['arm_status']!=0:raise self.controller_error(controller)
            else:self.comm_fault_since=None
        low=self.arm.GetArmLowSpdInfoMsgs()
        for i in range(1,7):
            if time.monotonic()-self.received.get(0x260+i,0)>.5:raise RuntimeError(f'J{i} driver status is stale.')
            status=getattr(low,f'motor_{i}').foc_status
            if status.driver_error_status or status.stall_status:raise RuntimeError(f'J{i} driver fault; automatic fault clearing disabled.')
            if require_enabled and not status.driver_enable_status:raise RuntimeError(f'J{i} disabled; following stopped.')
    def enable(self):
        feedback=self.read();self.healthy();deadline=time.monotonic()+3
        controller=feedback['controller']
        if controller['age_s']>.5:raise RuntimeError('Controller status is stale; Start refused.')
        if controller['teach_status']!=0 or controller['ctrl_mode'] in (2,6):raise RuntimeError('Exit teach mode before Start.')
        if controller['arm_status'] not in (0,1,5):raise self.controller_error(controller)  # 5 is cleared below (clear_err)
        # Start is the explicit recovery action. The controller stop latch survives
        # disconnect/restart, so the Python estopped flag alone is insufficient.
        if controller['arm_status']==1 or self.estopped:
            self.arm.MotionCtrl_1(0x02,0,0)
            while time.monotonic()<deadline:
                current=self.controller_state()
                if current['age_s']<=.5 and current['arm_status']==0:break
                time.sleep(.02)
            else:raise RuntimeError('Controller remains stopped; check physical E-stop.')
            self.estopped=False
        # SDK joint demo enables all drives before selecting control mode.
        lo=np.array([-150.,0.,-170.,-100.,-70.,-120.]);hi=np.array([150.,180.,0.,100.,70.,120.])
        # Option B (user decision 2026-10-07): like the working SO-101 backend, clear joint error flags before enabling.
        self.arm.JointConfig(joint_num=7,set_zero=0x00,acc_param_is_effective=0x00,clear_err=0xAE);time.sleep(.1);self.comm_fault_since=None
        # EnablePiper returns cached feedback from BEFORE its send. Require fresh
        # replies and a stable CAN-control acknowledgement before releasing homing.
        started=time.monotonic();stable_since=None;last_mode=0.;target=None
        try:
            while time.monotonic()<deadline:
                now=time.monotonic();enabled=self.arm.EnablePiper()
                controller=self.controller_state()
                if controller['age_s']<=.5 and controller['arm_status'] not in (0,1,5):raise self.controller_error(controller)
                fresh=all(self.received.get(i,0)>started for i in (0x2A1,0x261,0x262,0x263,0x264,0x265,0x266))
                if enabled and fresh:
                    self.healthy()
                    if target is None:target=np.clip(self.read()['q'],lo,hi)
                    if now-last_mode>=.05:
                        self.arm.MotionCtrl_2(0x01,0x01,10,0x00)
                        # Enabling can invalidate the preload. Supply the bounded
                        # measured target after enable, as in the SDK joint demo.
                        self.arm.JointCtrl(*[int(round(x*1000)) for x in target])
                        last_mode=now
                    ready=controller['age_s']<=.5 and controller['arm_status']==0 and controller['err_code']==0 and controller['ctrl_mode']==1 and controller['teach_status']==0
                    if ready:
                        if stable_since is None:stable_since=now
                        if now-stable_since>=.25:
                            self.enabled=True;self.controller_fault=None;self._start_sender(target);return
                    else:stable_since=None
                else:stable_since=None
                time.sleep(.02)
            self.controller_fault=controller
            raise RuntimeError(f"Start not ready: enabled={enabled}, fresh={fresh}, mode={controller['ctrl_mode']}, status={controller['arm_status']}, error={controller['err_code']}.")
        except Exception:
            self.emergency();raise
    def _transmit(self,q,speed_rate):
        # Every SDK TX uses this lock: the SDK encoder has a shared message.
        with self.tx_lock:
            if not self.enabled or self.tx_stop.is_set():return
            self.arm.MotionCtrl_2(0x01,0x01,speed_rate,0x00)
            self.arm.JointCtrl(*[int(round(x*1000)) for x in q]);self.sent+=1
            now=time.monotonic()
            if self.tx_last:self.tx_max_gap=max(self.tx_max_gap,now-self.tx_last)
            self.tx_last=now
    def _start_sender(self,q):
        self._stop_sender();self.tx_stop.clear();self.tx_fault=None;self.tx_last=0.;self.tx_max_gap=0.;self.tx_stale_holds=0
        with self.tx_lock:self.tx_command=np.asarray(q,float).copy();self.tx_stamp=time.monotonic();self.tx_speed=10
        def run():
            while not self.tx_stop.is_set():
                started=time.monotonic()
                try:
                    feedback=self.read();self.healthy(require_enabled=True)
                    with self.tx_lock:
                        if not self.enabled or self.tx_stop.is_set():break
                        if time.monotonic()-self.tx_stamp>.3:
                            # Hold measured pose; never keep pursuing an old
                            # target if the planner has stopped publishing.
                            measured=feedback['q']
                            lo=np.array([-150.,0.,-170.,-100.,-70.,-120.]);hi=np.array([150.,180.,0.,100.,70.,120.])
                            if np.any(measured<lo) or np.any(measured>hi):raise RuntimeError('Planner update expired at an out-of-range measured pose; stopped.')
                            self.tx_command=measured.copy();self.tx_stamp=time.monotonic();self.tx_stale_holds+=1
                        self._transmit(self.tx_command,self.tx_speed)
                except Exception as e:
                    self.tx_fault=str(e);self.emergency();break
                # 50 Hz (was 200 Hz = ~800 TX frames/s with MotionCtrl_2+JointCtrl each cycle). The working SO-101
                # Piper backend streams at 60 Hz; 200 Hz coincided with joint communication faults / stale feedback.
                self.tx_stop.wait(max(.001,TX_PERIOD_S-(time.monotonic()-started)))
        self.tx_thread=threading.Thread(target=run,name='piper-can-sender',daemon=True);self.tx_thread.start()
    def _stop_sender(self):
        self.tx_stop.set()
        if self.tx_thread and self.tx_thread is not threading.current_thread():
            self.tx_thread.join(timeout=1.)
            if self.tx_thread.is_alive():raise RuntimeError('CAN sender did not stop; USB release refused.')
        self.tx_thread=None
    def send(self,q,speed_rate=10):
        if self.tx_fault:raise RuntimeError(self.tx_fault)
        if not self.enabled:raise RuntimeError('Piper is not enabled.')
        if not isinstance(speed_rate,int) or not 1<=speed_rate<=100:raise ValueError('Invalid hardware speed percentage.')
        q=np.asarray(q,float)
        lo=np.array([-150.,0.,-170.,-100.,-70.,-120.]);hi=np.array([150.,180.,0.,100.,70.,120.])
        if q.shape!=(6,) or not np.isfinite(q).all() or np.any(q<lo) or np.any(q>hi):raise ValueError('CAN target outside firmware joint limits.')
        self.read();self.healthy(require_enabled=True)
        with self.tx_lock:
            if not self.enabled:raise RuntimeError(self.tx_fault or 'Piper stopped.')
            self.tx_command=q.copy();self.tx_speed=speed_rate;self.tx_stamp=time.monotonic()
    def transmission_state(self):
        return dict(sent=self.sent,max_interval_ms=self.tx_max_gap*1000,planner_stale_holds=self.tx_stale_holds,fault=self.tx_fault,sender_running=bool(self.tx_thread and self.tx_thread.is_alive()))
    def hold(self):
        if self.arm and self.enabled:
            try:
                feedback=self.read();self.healthy(require_enabled=True)
                q=feedback['q']
                lo=np.array([-150.,0.,-170.,-100.,-70.,-120.]);hi=np.array([150.,180.,0.,100.,70.,120.])
                # A measured folded encoder pose can be outside firmware command
                # limits. Stop there rather than submit an illegal hold target.
                if np.any(q<lo) or np.any(q>hi):self.emergency();return
                with self.tx_lock:
                    self.tx_command=q.copy();self.tx_stamp=time.monotonic();self.tx_speed=10
                    self._transmit(q,10)
            except Exception:self.emergency()
    def disable(self):
        """Power off the motors (used only at the verified folded rest pose)."""
        if self.arm is None:return
        self.read();self._stop_sender();deadline=time.monotonic()+3
        # piper_sdk DisablePiper() sends the disable command and returns any(enable status read *before* sending),
        # i.e. True while some motor is still enabled. Repeat until it reports all motors disabled.
        while time.monotonic()<deadline:
            if not self.arm.DisablePiper():self.enabled=False;return
            time.sleep(.05)
        raise RuntimeError('Piper disable not confirmed (motors still report enabled).')
    def emergency(self):
        self.tx_stop.set()
        with self.tx_lock:
            if self.arm:self.arm.MotionCtrl_1(0x01,0,0)
            self.estopped=True;self.enabled=False;self.tx_command=None
    def disconnect(self):
        if self.arm:
            try:self.hold()
            finally:
                self._stop_sender()
                port=getattr(self.arm,'_C_PiperInterface_V2__arm_can',None)
                bus=getattr(port,'bus',None)
                for suffix in ('__read_can_stop_event','__can_monitor_stop_event'):
                    event=getattr(self.arm,'_C_PiperInterface_V2'+suffix,None)
                    if event is not None:event.set()
                for suffix in ('__can_deal_th','__can_monitor_th'):
                    thread=getattr(self.arm,'_C_PiperInterface_V2'+suffix,None)
                    if thread is not None:thread.join(timeout=1.)
                try:self.arm.DisconnectPort()
                finally:
                    # python-can stops gs_usb but retains its libusb handle.
                    # Explicitly release it so a subsequent connection can open it.
                    device=getattr(getattr(bus,'gs_usb',None),'gs_usb',None)
                    if device is not None:
                        import usb.util
                        usb.util.dispose_resources(device)
                    self.arm=None;self.enabled=False

class MockHardware:
    mode='mock'
    def __init__(self):self.q=np.array([14.4,49.3,-57.9,0,38.6,0.]);self.enabled=False;self.sent=0;self.connected=False
    def connect(self):self.connected=True;return self.read()
    def read(self):
        if not self.connected:raise RuntimeError('Mock disconnected.')
        from pnd_tracking import fk_jacobian
        tf,_=fk_jacobian(np.deg2rad(self.q));return dict(q=self.q.copy(),p=tf[:3,3]*1000,r=tf[:3,:3],age=0,stamp=time.time(),pose_stamp=time.time())
    def healthy(self):pass
    def enable(self):self.enabled=True
    def send(self,q,speed_rate=10):
        if not self.enabled:raise RuntimeError('Mock is not enabled.')
        self.q=np.asarray(q).copy();self.sent+=1
    def hold(self):pass
    def disable(self):self.enabled=False
    def emergency(self):self.enabled=False
    def disconnect(self):self.connected=False;self.enabled=False
