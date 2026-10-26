import json,sys
from pathlib import Path
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tag_intrinsics import estimate_focal
from pose_core import object_corners,solve_pose

frames=json.loads(Path(__file__).with_name('phone_rejections.json').read_text(encoding='utf-8'))
data=dict(frames=[x['corners'] for x in frames],width=720,height=1280,tag_mm=80,calibration=frames[0]['calibration'])
estimated=estimate_focal(data)
old=[solve_pose(x['corners'],80,data['calibration'],720,1280)['reprojection_px'] for x in frames]
new=[solve_pose(x['corners'],80,estimated,720,1280)['reprojection_px'] for x in frames]
assert not estimated['calibrated'] and estimated['estimated']
assert sum(x<=3 for x in old)==0 and all(x<3 for x in new)
print('ACTUAL PHONE REPLAY PASS: 39/39 rejected with fx612 -> 39/39 accepted with fitted fx%.1f; median %.2f -> %.2f px'%(estimated['fx'],np.median(old),np.median(new)),flush=True)
assert all(solve_pose(x['corners'],80,dict(data['calibration'],fx=1088,fy=1088),720,1280)['reprojection_px']<3 for x in frames)
print('PORTRAIT INITIALIZATION PASS: long-side-based provisional focal admits all captured views',flush=True)
for bad in [dict(data,frames=data['frames'][:4]),dict(data,frames=[data['frames'][0]]*20),dict(data,calibration=dict(data['calibration'],calibrated=True))]:
    try:estimate_focal(bad)
    except ValueError:pass
    else:raise AssertionError('invalid/duplicate/fully calibrated input accepted')
k=np.array([[1400.,0,360],[0,1400.,640],[0,0,1]])
known=[]
for i in range(20):
    r=np.array([.12+i*.025,.18*np.sin(i),.03*i]);t=np.array([.04*np.sin(i),.03*np.cos(i),.55+.015*(i%5)])
    known.append(cv2.projectPoints(object_corners(.08),r,t,k,None)[0].reshape(4,2).tolist())
fitted=estimate_focal(dict(data,frames=known))
assert abs(fitted['fx']-1400)<1,fitted
print('FOCAL VALIDATION PASS: independent view validation, known focal recovered, duplicate views and formal-calibration overwrite refused',flush=True)
