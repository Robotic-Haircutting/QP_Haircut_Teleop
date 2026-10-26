"""Use one continuous planar pose for numeric readout, overlays and control."""
import cv2,numpy as np
from scipy.spatial.transform import Rotation
from phone_pose import select_phone_pose
from pose_core import camera_matrix,object_corners

def continuous_pose(result,data,previous=None):
    p,cr,index=select_phone_pose(result,previous)
    r=cr.T;t=-r@p
    out=dict(result,raw_camera_t_mm=result['camera_t_mm'],raw_camera_R=result['camera_R'],raw_reprojection_px=result['reprojection_px'],selected_candidate=index,
             camera_t_mm=p.tolist(),camera_R=cr.tolist(),t_mm=t.tolist(),R=r.tolist(),rpy_deg=Rotation.from_matrix(cr).as_euler('xyz',degrees=True).tolist(),
             reprojection_px=(result.get('candidate_errors_px') or [result['reprojection_px']])[index],pose_selection='temporal_continuity')
    k,dist=camera_matrix(data['calibration'],data['width'],data['height']);rv=cv2.Rodrigues(r)[0];tv=t/1000
    s=float(data['tag_mm'])/1000;axes=np.array([[0,0,0],[s/2,0,0],[0,s/2,0],[0,0,s/2]])
    cube=np.vstack([object_corners(s),object_corners(s)+[0,0,-s]]);depth=(cube@r.T+tv)[:,2]
    out['axes_pixels']=cv2.projectPoints(axes,rv,tv,k,dist)[0].reshape(-1,2).tolist()
    out['cube_pixels']=cv2.projectPoints(cube,rv,tv,k,dist)[0].reshape(-1,2).tolist() if np.min(depth)>.01 else None
    out['cube_depths']=depth.tolist()
    return out,(p,cr)
