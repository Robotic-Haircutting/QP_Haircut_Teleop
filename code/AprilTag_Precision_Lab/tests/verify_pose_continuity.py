import sys
from pathlib import Path
import numpy as np,cv2
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pose_continuity import continuous_pose
from phone_pose import select_phone_pose
data=dict(tag_mm=50,width=960,height=1280,calibration=dict(width=960,height=1280,fx=1100,fy=1100,cx=480,cy=640,calibrated=True))
good=np.array([230.,5.,-900.]);bad=np.array([-80.,35.,-900.])
good_r=Rotation.from_euler('y',-9,degrees=True).as_matrix();bad_r=Rotation.from_euler('y',11,degrees=True).as_matrix()
def sample(first_good,offset=0):
 poses=[]
 for p,r in [(good+[offset,0,0],good_r),(bad+[offset,0,0],bad_r)]:poses.append(dict(t_mm=(-r@p).tolist(),R=r.tolist()))
 if not first_good:poses.reverse()
 raw=poses[0];r=np.array(raw['R']);p=-r.T@raw['t_mm']
 return dict(t_mm=raw['t_mm'],R=raw['R'],camera_t_mm=p.tolist(),camera_R=r.T.tolist(),reprojection_px=.8,candidate_errors_px=[.8,.86],candidate_poses=poses,ambiguous=True)
previous=None;raw=[];selected=[]
for i in range(40):
 result,previous=continuous_pose(sample(i%2==0),data,previous)
 raw.append(result['raw_camera_t_mm']);selected.append(result['camera_t_mm'])
 control,_,index=select_phone_pose(result)
 assert np.allclose(control,result['camera_t_mm']) and index==result['selected_candidate']
 assert result['ambiguous'] and result['cube_pixels'] is not None
assert np.max(np.linalg.norm(np.diff(raw,axis=0),axis=1))>300
assert np.max(np.linalg.norm(np.diff(selected,axis=0),axis=1))<1e-6
result,previous=continuous_pose(sample(False,10),data,previous)
assert np.allclose(result['camera_t_mm'],good+[10,0,0])
print('PASS: 310mm alternating planar branches stay continuous; readout/control agree; overlays and raw ambiguity retained; genuine 10mm movement responds')
