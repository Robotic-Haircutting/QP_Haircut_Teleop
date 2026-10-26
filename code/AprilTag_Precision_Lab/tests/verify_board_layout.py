import sys
from pathlib import Path
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from board_layout import calibrate_layout
from pose_core import object_corners
from tag_board import solve_board
cal=dict(width=960,height=1280,fx=1160,fy=1140,cx=480,cy=640,distortion=[.04,-.02,.001,-.002,0],calibrated=True)
k=np.array([[1160.,0,480],[0,1140,640],[0,0,1]])
points=[]
for x,y,a in [(0,0,0),(.152,.006,.03),(-.004,.146,-.02),(.150,.155,np.pi/2)]:
 c,s=np.cos(a),np.sin(a);xy=object_corners(.05)[:,:2]@np.array([[c,s],[-s,c]])+[x,y]
 points.append(np.column_stack([xy,np.zeros(4)]))
rng=np.random.default_rng(11);frames=[]
for n in range(16):
 rv=np.array([.18*np.sin(n*.4),.15*np.cos(n*.3),.03]);tv=np.array([-.07,-.08,.6+n*.004])
 frames.append([dict(id=i,corners=(cv2.projectPoints(p,rv,tv,k,np.array(cal['distortion']))[0].reshape(4,2)+rng.normal(0,.12,(4,2))).tolist()) for i,p in enumerate(points)])
frames[0].append(dict(frames[0][0]))
data=dict(frames=frames,tag_mm=50,calibration=cal,width=960,height=1280)
fit=calibrate_layout(data)
expected=np.asarray(points)*1000;expected-=expected.mean(axis=(0,1))
actual=np.array([fit['geometry']['corners_mm'][str(i)] for i in range(4)])
assert np.max(np.linalg.norm(actual-expected,axis=2))<1,(actual,expected)
for frame in frames:
 p=solve_board(frame,fit['geometry'],cal,960,1280)
 assert p['reprojection_px']<.6 and p['board_ids']==[0,1,2,3],p
try:calibrate_layout(dict(data,frames=frames[:2]));raise AssertionError('too few samples accepted')
except ValueError:pass
print('PASS: recovered unequal spacing and rotated Tag, <1 mm synthetic corner layout error; all frames and duplicate quad track; insufficient samples rejected',fit['reprojection_px'])
