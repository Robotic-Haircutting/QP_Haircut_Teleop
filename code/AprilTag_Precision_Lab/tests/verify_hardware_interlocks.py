"""Exercise SDK protocol calls with a fake CAN/SDK; never opens actual hardware."""
import sys,time,types
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pnd_tracking import fk_jacobian
from scipy.spatial.transform import Rotation
q=np.array([14.4,49.3,-57.9,0,38.6,0.]);tf,_=fk_jacobian(np.deg2rad(q));rpy=Rotation.from_matrix(tf[:3,:3]).as_euler('xyz',degrees=True)
class Port:
 def __init__(self,*args):self.callback=args[-1]
class Arm:
 def __init__(self,*args,**kwargs):self.calls=[];self.driver_enabled=False
 def ParseCANFrame(self,msg):pass
 def ConnectPort(self,**kwargs):
  assert kwargs=={'piper_init':False};self.calls.append('connect')
  for i in [0x2A2,0x2A3,0x2A4,0x2A5,0x2A6,0x2A7,0x261,0x262,0x263,0x264,0x265,0x266]:self._C_PiperInterface_V2__arm_can.callback(types.SimpleNamespace(arbitration_id=i))
 def GetArmJointMsgs(self):return types.SimpleNamespace(time_stamp=80.,joint_state=types.SimpleNamespace(**{f'joint_{i+1}':int(round(v*1000)) for i,v in enumerate(q)}))
 def GetArmEndPoseMsgs(self):return types.SimpleNamespace(time_stamp=80.,end_pose=types.SimpleNamespace(**dict(zip(['X_axis','Y_axis','Z_axis','RX_axis','RY_axis','RZ_axis'],[*(tf[:3,3]*1e6),*(rpy*1000)]))))
 def GetArmLowSpdInfoMsgs(self):return types.SimpleNamespace(**{f'motor_{i}':types.SimpleNamespace(foc_status=types.SimpleNamespace(driver_error_status=False,stall_status=False,driver_enable_status=self.driver_enabled)) for i in range(1,7)})
 def EnablePiper(self):self.calls.append('enable');self.driver_enabled=True;return True
 def MotionCtrl_2(self,*args):self.calls.append(('mode',args))
 def JointCtrl(self,*args):self.calls.append(('joint',args))
 def MotionCtrl_1(self,*args):self.calls.append(('stop',args))
 def DisconnectPort(self):self.calls.append('disconnect')
sys.modules['can']=types.SimpleNamespace(Bus=lambda **kwargs:None)
sys.modules['libusb_package']=types.SimpleNamespace(get_library_path=lambda:str(Path(__file__).parent/'fake.dll'))
sys.modules['piper_sdk']=types.SimpleNamespace(C_PiperInterface_V2=Arm)
sys.modules['piper_sdk.hardware_port']=types.SimpleNamespace(C_STD_CAN=Port)
from piper_hardware import PiperHardware
hardware=PiperHardware();hardware.connect();arm=hardware.arm
assert arm.calls==['connect'] and not hardware.enabled and hardware.sent==0
assert hardware.read()['age']<.1,'adapter uptime incorrectly treated as UTC'
hardware.enable();assert arm.calls[1][0]=='mode' and arm.calls[2][0]=='joint' and arm.calls[3]=='enable'
assert arm.calls[2][1]==tuple(int(round(v*1000)) for v in q),'enable preload jumps'
hardware.send(q,speed_rate=30);assert arm.calls[-2]==('mode',(0x01,0x01,30,0x00))
hardware.send(q);assert arm.calls[-2]==('mode',(0x01,0x01,10,0x00))
before=len(arm.calls)
try:hardware.send(q,speed_rate=101);raise AssertionError('bad SDK speed accepted')
except ValueError:pass
assert len(arm.calls)==before
hardware.received[0x2A5]=time.monotonic()-1
try:hardware.send(q);raise AssertionError('stale pair accepted')
except RuntimeError:pass
hardware.hold();assert hardware.estopped and not hardware.enabled
hardware.disconnect()
print('HARDWARE FAKE SDK PASS: read-only connect sends no motion/enable; adapter uptime valid; enable preloads measured joints; stale joint pair blocks send; emergency hold')
