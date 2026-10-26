import sys,time,tempfile
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from phone_motion_filter import PhoneMotionFilter
from robot_bridge import RobotExperiment

f=PhoneMotionFilter();raw=[];smooth=[];angles=[];filtered_angles=[]
for i in range(240):
 p=np.array([1.5*(-1)**i,0,-600.]);angle=.6*(-1)**i
 r=Rotation.from_euler('z',angle,degrees=True).as_matrix()
 fp,fr,info=f.update(p,r,i/30,3)
 assert info['accepted'];raw.append(p[0]);smooth.append(fp[0]);angles.append(angle);filtered_angles.append(Rotation.from_matrix(fr).as_rotvec()[2]*180/np.pi)
position_ratio=np.std(smooth[60:])/np.std(raw[60:]);angle_ratio=np.std(filtered_angles[60:])/np.std(angles[60:])
assert position_ratio<.25 and angle_ratio<.25,(position_ratio,angle_ratio)
# Genuine motion raises the cutoff; a 20mm move should respond promptly.
f.reset();f.update([0,0,-600],np.eye(3),0)
for i in range(1,7):fp,fr,info=f.update([20,0,-600],np.eye(3),i/30)
assert fp[0]>18,'adaptive filter made the 200ms response sluggish'
accepted=fp.copy();fp,fr,info=f.update([-290,0,-600],np.eye(3),7/30,ambiguous=True)
assert not info['accepted'] and np.array_equal(fp,accepted),'branch jump was smoothed into motion'

robot=RobotExperiment(root=Path(tempfile.mkdtemp())/'results')
d=dict(client_id='filter',moving='phone',tracking_mode='board',board={'tag_mm':50},camera_id='test',tag_mm=50,width=640,height=480,calibration={'calibrated':True})
def action(a):return robot.action(dict(action=a,client_id='filter',bench_clear=True,backend='mock'))
def feed(p,ambiguous=False):
 p=np.array(p,float);robot.feed(dict(camera_t_mm=p.tolist(),camera_R=np.eye(3).tolist(),t_mm=(-p).tolist(),R=np.eye(3).tolist(),reprojection_px=.1,calibrated=True,ambiguous=ambiguous),d)
try:
 action('connect');action('enable');q=robot.backend.q.copy()
 feed([0,0,-600]);action('clutch');assert robot.paused and not robot.resume_pending
 for _ in range(4):feed([0,0,-600]);time.sleep(.025)
 assert robot.paused and np.array_equal(q,robot.backend.q),'tracking recovery automatically resumed'
 action('clutch');assert not robot.paused
 feed([-310,0,-600],True);assert not robot.paused and not robot.filter_info['accepted']
 assert np.array_equal(robot.latest[0],[0,0,-600])
 feed([0,0,-600]);time.sleep(.03);assert not robot.paused
 for _ in range(3):feed([-310,0,-600],True);time.sleep(.025)
 assert robot.paused and not robot.fault
 held=robot.backend.q.copy()
 for _ in range(4):feed([0,0,-600]);time.sleep(.025)
 assert robot.paused and np.array_equal(held,robot.backend.q),'repeated jump recovery automatically resumed'
 action('clutch');assert not robot.paused
 for _ in range(4):feed([0,0,-600]);time.sleep(.025)
 assert np.max(np.abs(held-robot.backend.q))<.01,'manual Resume jumped'
 print('PHONE FILTER PASS: position/rotation jitter ratios',position_ratio,angle_ratio,'; >90% response in200ms; 310mm branch rejected; one spike keeps following; repeated spikes hold; no autoResume; explicit Resume no jump',flush=True)
finally:robot.close()
