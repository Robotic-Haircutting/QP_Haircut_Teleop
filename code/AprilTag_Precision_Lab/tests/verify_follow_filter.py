import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from follow_filter import FollowFilter
for dt in (.01,.02,.04):
 f=FollowFilter();f.reset(np.zeros(3),np.eye(3))
 for _ in range(round(.12/dt)):p=f.position([.175,0,0],dt)
 assert p[0]>.169 and p[0]<.175
 assert np.linalg.norm(p[1:])==0
 for _ in range(round(.3/dt)):r=f.orientation(Rotation.from_euler('z',5,degrees=True).as_matrix(),dt)
 assert abs(Rotation.from_matrix(r).magnitude()*180/np.pi-5)<.01
 assert abs(np.linalg.det(r)-1)<1e-9
 f.reset([100,20,30],r);assert np.allclose(f.position([100,20,30],dt),[100,20,30])
f=FollowFilter();f.reset(np.zeros(3),np.eye(3));raw=[];filtered=[]
for i in range(150):
 p=.175*(-1)**i;raw.append(p);filtered.append(f.position([p,0,0],.02)[0])
ratio=np.std(filtered[20:])/np.std(raw[20:]);assert ratio<.4,ratio
print('FOLLOW FILTER PASS: no small-motion deadband, >96% step response in 120ms, sample-rate independent, proper rotations and zero-jump reset; fast jitter ratio',round(ratio,3))
