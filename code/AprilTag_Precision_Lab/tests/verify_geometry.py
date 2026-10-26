import sys,json
from pathlib import Path
import cv2,numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pose_core import solve_pose,object_corners,camera_matrix,metrics
from server import calibrate_frames
cal={'width':1280,'height':720,'fx':1000.,'fy':990.,'cx':640.,'cy':360.,'distortion':[-.1,.02,.001,-.001,0],'calibrated':True}
k,d=camera_matrix(cal,1280,720);r=Rotation.from_euler('xyz',[165,12,8],degrees=True);t=np.array([.02,-.01,.55]);obj=object_corners(.08)
image=cv2.projectPoints(obj,r.as_rotvec(),t,k,d)[0].reshape(-1,2);out=solve_pose(image,80,cal,1280,720)
assert np.linalg.norm(np.array(out['t_mm'])-t*1000)<1e-5
assert (Rotation.from_matrix(out['R'])*r.inv()).magnitude()<1e-6
assert out['reprojection_px']<1e-5
half=solve_pose(image/2,80,cal,640,360);assert np.linalg.norm(np.array(half['t_mm'])-t*1000)<1e-5
try:solve_pose(image,80,cal,720,1280);raise AssertionError('resolution mismatch accepted')
except ValueError:pass
rng=np.random.default_rng(7);samples=[]
for _ in range(100):
 p=solve_pose(image+rng.normal(0,.05,(4,2)),80,cal,1280,720);samples.append(p)
m=metrics(samples);assert m['position_p95_radius_mm']<3
# A realistic tilted chessboard calibration, including distortion.
board=np.zeros((9*6,3),np.float32);board[:,:2]=np.mgrid[0:9,0:6].T.reshape(-1,2)*20;frames=[]
for i in range(20):
 rv=Rotation.from_euler('xyz',[-25+i*2,10*np.sin(i),i*3],degrees=True).as_rotvec();tv=np.array([-100+10*(i%5),-60+12*(i%4),500+20*(i%6)],float)
 corners=cv2.projectPoints(board,rv,tv,k,d)[0].astype(np.float32)
 frames.append((board,corners,(1280,720),(9,6,20)))
computed=calibrate_frames(frames);assert computed['rms_px']<.001;assert abs(computed['fx']-1000)<.1
# Generate a known tag to validate the actual WASM detector in Node.
dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11);tag=cv2.aruco.generateImageMarker(dictionary,0,160)
canvas=np.full((480,640),255,np.uint8);canvas[160:320,240:400]=tag
root=Path(__file__).resolve().parents[1];(root/'tests/synthetic_tag.raw').write_bytes(canvas.tobytes())
print(json.dumps({'pose_error_mm':float(np.linalg.norm(np.array(out['t_mm'])-t*1000)),'pose_rotation_error_deg':float((Rotation.from_matrix(out['R'])*r.inv()).magnitude()*180/np.pi),'calibration_rms_px':computed['rms_px'],'synthetic_jitter_p95_mm':m['position_p95_radius_mm'],'tests':'pose, distortion, scale, resolution rejection, metrics, calibration PASS'},indent=2))

# Portrait phone / imperfect corners: refinement must improve the fit,
# while keeping the 3px control quality threshold unchanged.
phone_cal=dict(width=720,height=1280,fx=612.,fy=612.,cx=360.,cy=640.)
phone_k,_=camera_matrix(phone_cal,720,1280);rng=np.random.default_rng(31)
for _ in range(2):
 noisy=cv2.projectPoints(obj,np.array([.12,.15,1.9]),np.array([-.06,.04,.26]),phone_k,None)[0].reshape(4,2)+rng.normal(0,4,(4,2))
initial=cv2.solvePnPGeneric(obj,noisy,phone_k,None,flags=cv2.SOLVEPNP_IPPE_SQUARE)
initial_errors=[float(np.sqrt(np.mean(np.sum((cv2.projectPoints(obj,rv,tv,phone_k,None)[0].reshape(4,2)-noisy)**2,axis=1)))) for rv,tv in zip(initial[1],initial[2])]
refined=solve_pose(noisy,80,phone_cal,720,1280)
assert min(initial_errors)>3 and refined['reprojection_px']<3,(initial_errors,refined['reprojection_px'])
assert all(x['t_mm'][2]>0 for x in refined['candidate_poses'])
print('PHONE FIT PASS: raw IPPE %.3fpx -> refined %.3fpx; quality gate remains 3px'%(min(initial_errors),refined['reprojection_px']))
