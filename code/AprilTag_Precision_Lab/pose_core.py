"""Raw square-tag pose estimation and precision metrics. No robot control."""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def camera_matrix(calibration, width, height):
    base_w, base_h = int(calibration['width']), int(calibration['height'])
    if base_w <= 0 or base_h <= 0 or width <= 0 or height <= 0:
        raise ValueError('Invalid image dimensions')
    if abs(width / height - base_w / base_h) > .001:
        raise ValueError('Calibration aspect ratio differs from the current frame; recalibrate (cropping and rotation are not guessed)')
    sx, sy = width/base_w, height/base_h
    fx, fy, cx, cy = [float(calibration[k]) for k in ('fx','fy','cx','cy')]
    if not np.isfinite([fx,fy,cx,cy]).all() or min(fx,fy) <= 0:
        raise ValueError('Invalid intrinsics')
    k = np.array([[fx*sx,0,cx*sx],[0,fy*sy,cy*sy],[0,0,1]],float)
    d = np.asarray(calibration.get('distortion',[0,0,0,0,0]),float).ravel()
    if len(d) not in (4,5,8,12,14) or not np.isfinite(d).all():
        raise ValueError('Invalid distortion coefficients')
    return k,d


def object_corners(size_m):
    if not .005 <= size_m <= 1:
        raise ValueError('Tag black-border size must be 5-1000 mm')
    h=size_m/2
    # Local tag corner indices; preserve native decoded order (never sort by image position).
    return np.array([[-h,h,0],[h,h,0],[h,-h,0],[-h,-h,0]],float)


def solve_pose(corners, size_mm, calibration, width, height):
    image=np.asarray(corners,float)
    if image.shape != (4,2) or not np.isfinite(image).all() or abs(cv2.contourArea(image.astype(np.float32))) < 4:
        raise ValueError('Invalid tag corners')
    obj=object_corners(float(size_mm)/1000)
    k,d=camera_matrix(calibration,int(width),int(height))
    result=cv2.solvePnPGeneric(obj,image,k,d,flags=cv2.SOLVEPNP_IPPE_SQUARE)
    candidates=[]
    for rv,tv in zip(result[1],result[2]):
        # IPPE provides two initial estimates. Refine each independently against
        # the measured corners; never accept a worse fit or negative depth.
        initial=cv2.projectPoints(obj,rv,tv,k,d)[0].reshape(-1,2)
        initial_error=float(np.sqrt(np.mean(np.sum((initial-image)**2,axis=1))))
        refined_rv,refined_tv=cv2.solvePnPRefineLM(obj,image,k,d,rv.copy(),tv.copy())
        refined_r=cv2.Rodrigues(refined_rv)[0]
        refined_image=cv2.projectPoints(obj,refined_rv,refined_tv,k,d)[0].reshape(-1,2)
        refined_error=float(np.sqrt(np.mean(np.sum((refined_image-image)**2,axis=1))))
        if np.isfinite(refined_error) and refined_error<=initial_error and np.all((obj@refined_r.T+refined_tv.ravel())[:,2]>0):
            rv,tv=refined_rv,refined_tv
        r=cv2.Rodrigues(rv)[0];t=tv.ravel()
        if not np.isfinite(r).all() or not np.isfinite(t).all() or np.any((obj@r.T+t)[:,2] <= 0):continue
        projected=cv2.projectPoints(obj,rv,tv,k,d)[0].reshape(-1,2)
        error=float(np.sqrt(np.mean(np.sum((projected-image)**2,axis=1))))
        candidates.append((error,rv,tv,r,t))
    if not candidates:raise ValueError('No positive-depth pose solution')
    candidates.sort(key=lambda x:x[0])
    error,rv,tv,r,t=candidates[0]
    camera_r=r.T;camera_t=-camera_r@t
    axes=np.array([[0,0,0],[size_mm/2000,0,0],[0,size_mm/2000,0],[0,0,size_mm/2000]],float)
    pixels=cv2.projectPoints(axes,rv,tv,k,d)[0].reshape(-1,2)
    cube=np.vstack([obj,obj+np.array([0,0,-float(size_mm)/1000])])
    cube_depths=(cube@r.T+t)[:,2]
    cube_pixels=cv2.projectPoints(cube,rv,tv,k,d)[0].reshape(-1,2) if np.all(cube_depths>0.01) else None
    ambiguity=(len(candidates)>1 and candidates[1][0]-error < .2)
    return {'pose_method':'IPPE + LM','t_mm':(t*1000).tolist(),'R':r.tolist(),'camera_t_mm':(camera_t*1000).tolist(),
        'camera_R':camera_r.tolist(),'rpy_deg':Rotation.from_matrix(camera_r).as_euler('xyz',degrees=True).tolist(),
        'reprojection_px':error,'candidate_errors_px':[x[0] for x in candidates],
        'candidate_poses':[{'t_mm':(x[4]*1000).tolist(),'R':x[3].tolist()} for x in candidates],
        'cube_pixels':cube_pixels.tolist() if cube_pixels is not None else None,'cube_depths':cube_depths.tolist(),
        'ambiguous':ambiguity,'axes_pixels':pixels.tolist(),'calibrated':bool(calibration.get('calibrated',False))}


def metrics(samples):
    if len(samples)<5:raise ValueError('Record at least 5 valid poses')
    p=np.asarray([s['camera_t_mm'] for s in samples],float)
    rots=Rotation.from_matrix(np.asarray([s['camera_R'] for s in samples],float))
    mean_r=rots.mean();angles=(rots*mean_r.inv()).magnitude()*180/np.pi
    radius=np.linalg.norm(p-p.mean(axis=0),axis=1)
    jumps=(rots[1:]*rots[:-1].inv()).magnitude()*180/np.pi
    return {'samples':len(samples),'mean_camera_t_mm':p.mean(axis=0).tolist(),'mean_camera_R':mean_r.as_matrix().tolist(),
        'position_std_mm':p.std(axis=0,ddof=1).tolist(),'position_p95_radius_mm':float(np.percentile(radius,95)),
        'rotation_rms_deg':float(np.sqrt(np.mean(angles**2))),'rotation_p95_deg':float(np.percentile(angles,95)),
        'max_adjacent_rotation_deg':float(jumps.max()),'ambiguous_fraction':float(np.mean([s.get('ambiguous',False) for s in samples])),
        'mean_reprojection_px':float(np.mean([s['reprojection_px'] for s in samples]))}
