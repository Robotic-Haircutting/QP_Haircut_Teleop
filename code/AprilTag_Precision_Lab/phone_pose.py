"""Continuity selection among near-equal IPPE candidates; raw ambiguity remains visible."""
import numpy as np
from scipy.spatial.transform import Rotation

CAMERA_TO_BASE=np.array([[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]])
# Rear camera: phone-left is camera -X, mapped to robot +Y.
# Keep position and orientation on the same proper camera basis.
PHONE_POSITION_SIGN=np.eye(3)

def select_phone_pose(result,previous=None):
    poses=result.get('candidate_poses') or [dict(t_mm=result['t_mm'],R=result['R'])]
    errors=result.get('candidate_errors_px') or [result['reprojection_px']]
    allowed=[i for i,e in enumerate(errors) if e<=min(errors)+.2]
    candidates=[]
    for pose in poses:
        r=np.asarray(pose['R'],float).T
        candidates.append((-r@np.asarray(pose['t_mm'],float),r))
    index=int(result.get('selected_candidate',0))
    if index not in allowed:index=allowed[0]
    if previous is not None:
        def score(i):
            p,r=candidates[i];old_p,old_r=previous
            angle=Rotation.from_matrix(r@old_r.T).magnitude()
            return angle+np.linalg.norm(p-old_p)/500
        index=min(allowed,key=score)
    return (*candidates[index],index)

def rotation_step(current,target,max_angle):
    vector=Rotation.from_matrix(target@current.T).as_rotvec()
    angle=np.linalg.norm(vector)
    if angle>max_angle:vector*=max_angle/angle
    return Rotation.from_rotvec(vector).as_matrix()@current
