"""Provisional focal estimate from diverse square-tag views, for simulation."""
import cv2
import numpy as np
from pose_core import object_corners, camera_matrix, solve_pose


def estimate_focal(data):
    width,height=int(data['width']),int(data['height'])
    base=data['calibration']
    if base.get('calibrated'):raise ValueError('A proper calibration already exists; not overwriting')
    k,_=camera_matrix(base,width,height)
    frames=[]
    for points in data['frames'][:40]:
        p=np.asarray(points,np.float32)
        if p.shape!=(4,2) or not np.isfinite(p).all():continue
        if np.any(p<0) or np.any(p[:,0]>=width) or np.any(p[:,1]>=height):continue
        if abs(cv2.contourArea(p))<400:continue
        if frames and min(np.mean(np.linalg.norm(p-f,axis=1)) for f in frames)<8:continue
        frames.append(p)
    if len(frames)<12:raise ValueError('At least 12 different views are required; keep the Tag fixed and slowly change the phone angle and distance')
    obj=object_corners(float(data['tag_mm'])/1000).astype(np.float32)
    flags=(cv2.CALIB_USE_INTRINSIC_GUESS|cv2.CALIB_FIX_PRINCIPAL_POINT|
           cv2.CALIB_FIX_ASPECT_RATIO|cv2.CALIB_ZERO_TANGENT_DIST|
           cv2.CALIB_FIX_K1|cv2.CALIB_FIX_K2|cv2.CALIB_FIX_K3)
    def fit(views):
        _,matrix,_,_,_=cv2.calibrateCamera([obj]*len(views),views,(width,height),k.copy(),np.zeros(5),flags=flags)
        focal=float(matrix[0,0])
        if not np.isfinite(matrix).all() or not min(width,height)*.5<focal<max(width,height)*3:
            raise ValueError('Focal length estimate is unstable; add more full-Tag frames at different tilts')
        return matrix
    train=frames[::2];validation=frames[1::2]
    matrix=fit(train);check=fit(validation)
    if abs(matrix[0,0]-check[0,0])/matrix[0,0]>.25:
        raise ValueError('Focal length estimates disagree across views; add more tilt variation or use checkerboard calibration')
    def profile(matrix):
        return dict(width=width,height=height,fx=float(matrix[0,0]),fy=float(matrix[1,1]),
                    cx=float(matrix[0,2]),cy=float(matrix[1,2]),distortion=[0,0,0,0,0],
                    calibrated=False,estimated=True,source='tag_focal_fit',views=len(frames))
    provisional=profile(matrix)
    errors=[solve_pose(p,data['tag_mm'],provisional,width,height)['reprojection_px'] for p in validation]
    old=[solve_pose(p,data['tag_mm'],base,width,height)['reprojection_px'] for p in validation]
    if np.median(errors)>2 or np.percentile(errors,90)>3 or np.median(errors)>.7*np.median(old):
        raise ValueError('Validation on held-out frames failed; change the viewing angles or use checkerboard calibration')
    result=profile(fit(frames));result['validation_median_px']=float(np.median(errors))
    result['validation_p90_px']=float(np.percentile(errors,90))
    return result
