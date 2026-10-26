"""Known coplanar AprilTag 0..3 board, with a common origin at board center."""
import itertools
import cv2,numpy as np
from scipy.spatial.transform import Rotation
from pose_core import object_corners,camera_matrix

def board_points(tag_id,geometry):
    if 'corners_mm' in geometry:
        points=np.asarray(geometry['corners_mm'][str(tag_id)],float)
        if points.shape!=(4,3) or not np.isfinite(points).all():raise ValueError('Invalid calibrated board layout')
        return points/1000
    size=float(geometry.get('tag_mm',50));gx=float(geometry.get('gap_x_mm',100));gy=float(geometry.get('gap_y_mm',100))
    if not (5<=size<=300 and 0<=gx<=1000 and 0<=gy<=1000):raise ValueError('Invalid board size or black-border edge spacing')
    x=(-1 if tag_id%2==0 else 1)*(size+gx)/2000
    y=(-1 if tag_id<2 else 1)*(size+gy)/2000
    return object_corners(size/1000)+[x,y,0]

def candidates(obj,img,k,dist):
    result=cv2.solvePnPGeneric(obj,img,k,dist,flags=cv2.SOLVEPNP_IPPE);poses=[]
    for rv,tv in zip(result[1],result[2]):
        refined=cv2.solvePnPRefineLM(obj,img,k,dist,rv.copy(),tv.copy())
        options=[]
        for v,t in ((rv,tv),refined):
            r=cv2.Rodrigues(v)[0]
            if np.any((obj@r.T+t.ravel())[:,2]<=0):continue
            p=cv2.projectPoints(obj,v,t,k,dist)[0].reshape(-1,2)
            err=float(np.sqrt(np.mean(np.sum((p-img)**2,axis=1))))
            if np.isfinite(err):options.append((err,v,t,r))
        if options:poses.append(min(options,key=lambda x:x[0]))
    return sorted(poses,key=lambda x:x[0])

def unique_detections(detections):
    tags={}
    for detection in detections:
        tid=int(detection['id']);corners=np.asarray(detection['corners'],float)
        if tid not in range(4):continue
        if corners.shape!=(4,2) or not np.isfinite(corners).all() or abs(cv2.contourArea(corners.astype(np.float32)))<4:raise ValueError('Invalid Tag corners')
        if tid in tags:
            previous=tags[tid]
            # Native output may repeat one physical quad; keep the stronger result.
            distance=min(np.sqrt(np.mean(np.sum((corners-np.roll(np.asarray(previous['corners']),n,axis=0))**2,axis=1))) for n in range(4))
            if distance>4:raise ValueError(f'ID {tid} appears at different places in the frame: check the printed IDs or remove Tags with the same ID from the background')
            if detection.get('decision_margin',0)>previous.get('decision_margin',0):tags[tid]=detection
        else:tags[tid]=detection
    return [tags[i] for i in sorted(tags)]

def solve_board(detections,geometry,calibration,width,height):
    k,dist=camera_matrix(calibration,width,height);tags={}
    for detection in unique_detections(detections):
        tid=int(detection['id']);corners=np.asarray(detection['corners'],float)
        if tid not in range(4):continue
        if corners.shape!=(4,2) or not np.isfinite(corners).all() or abs(cv2.contourArea(corners.astype(np.float32)))<4:raise ValueError('Invalid Tag corners')
        tags[tid]=corners
    ids=sorted(tags)
    if not ids:raise ValueError('Board IDs 0–3 not found')
    def points(subset):return np.concatenate([board_points(i,geometry) for i in subset]),np.concatenate([tags[i] for i in subset])
    obj,img=points(ids);poses=candidates(obj,img,k,dist)
    if not poses:raise ValueError('No valid board pose')
    def errors(pose):
        _,rv,tv,_=pose
        projected=cv2.projectPoints(obj,rv,tv,k,dist)[0].reshape(len(ids),4,2)
        return np.sqrt(np.mean(np.sum((projected-img.reshape(len(ids),4,2))**2,axis=2),axis=1))
    # Reject an incorrectly localized whole tag, without averaging four poses.
    if len(ids)>=3 and max(errors(poses[0]))>3:
        trials=poses[:]
        for count in range(2,len(ids)):
            for subset in itertools.combinations(ids,count):
                so,si=points(subset);trials.extend(candidates(so,si,k,dist))
        best=max(trials,key=lambda p:(int(np.sum(errors(p)<=3)),-float(np.mean(np.minimum(errors(p),3)))))
        used=[i for i,e in zip(ids,errors(best)) if e<=3]
        if len(used)<2:raise ValueError('Tag sizes, positions or corners are inconsistent; check the board and the intrinsics')
        obj,img=points(used);poses=candidates(obj,img,k,dist)
    else:used=ids
    if not poses:raise ValueError('No valid board pose')
    err,rv,tv,r=poses[0];t=tv.ravel();cr=r.T;cp=-cr@t
    s=float(geometry['tag_mm'])/1000;axes=np.array([[0,0,0],[s/2,0,0],[0,s/2,0],[0,0,s/2]])
    base=object_corners(s);cube=np.vstack([base,base+[0,0,-s]]);depth=(cube@r.T+t)[:,2]
    return dict(pose_method='board IPPE + LM',t_mm=(t*1000).tolist(),R=r.tolist(),camera_t_mm=(cp*1000).tolist(),camera_R=cr.tolist(),rpy_deg=Rotation.from_matrix(cr).as_euler('xyz',degrees=True).tolist(),reprojection_px=err,candidate_errors_px=[p[0] for p in poses],candidate_poses=[dict(t_mm=(p[2].ravel()*1000).tolist(),R=p[3].tolist()) for p in poses],ambiguous=len(poses)>1 and poses[1][0]-err<.2,axes_pixels=cv2.projectPoints(axes,rv,tv,k,dist)[0].reshape(-1,2).tolist(),cube_pixels=cv2.projectPoints(cube,rv,tv,k,dist)[0].reshape(-1,2).tolist() if np.min(depth)>.01 else None,cube_depths=depth.tolist(),calibrated=bool(calibration.get('calibrated',False)),board_ids=used,visible_ids=ids,rejected_ids=[i for i in ids if i not in used],position_frame='camera_in_board',board_geometry=geometry)
