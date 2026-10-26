"""Local DLS steps with joint speed bounds and task-space backtracking."""
import numpy as np
from scipy.spatial.transform import Rotation
from piper_twin import fk_jacobian,rot_log,JOINT_MIN_DEG,JOINT_MAX_DEG


def solve_tracking_ik(p_m,r,q0_deg):
    # The shared legacy solver stops at 0.1mm: a backtracked small step can
    # therefore return unchanged joints. Use a finer tolerance locally.
    q=np.deg2rad(np.asarray(q0_deg,float));lo=np.deg2rad(JOINT_MIN_DEG);hi=np.deg2rad(JOINT_MAX_DEG)
    weights=np.array([1.,1.,1.,.3,.3,.3])
    for _ in range(40):
        transform,jacobian=fk_jacobian(q)
        error=np.r_[p_m-transform[:3,3],rot_log(r@transform[:3,:3].T)]
        if np.linalg.norm(error[:3])<1e-5 and np.linalg.norm(error[3:])<1e-4:break
        weighted=jacobian*weights[:,None]
        dq=weighted.T@np.linalg.solve(weighted@weighted.T+1e-3*np.eye(6),error*weights)
        dq*=min(1.,.2/max(np.max(np.abs(dq)),1e-12))
        q=np.clip(q+dq,lo,hi)
    transform,_=fk_jacobian(q)
    return np.rad2deg(q),float(np.linalg.norm(p_m-transform[:3,3])*1000),float(np.rad2deg(np.linalg.norm(rot_log(r@transform[:3,:3].T))))


def bounded_step(position_mm, rotation, current_deg, dt):
    q0=np.asarray(current_deg,float)
    transform,_=fk_jacobian(np.deg2rad(q0))
    p0=transform[:3,3]*1000;r0=transform[:3,:3]
    vector=Rotation.from_matrix(rotation@r0.T).as_rotvec()
    max_joint_step=45*min(max(dt,0),.05)
    last_error=(0.,0.)
    speed_limited=False
    for fraction in (1.,.5,.25,.125,.0625):
        p=p0+fraction*(np.asarray(position_mm)-p0)
        r=Rotation.from_rotvec(vector*fraction).as_matrix()@r0
        q,pe,re=solve_tracking_ik(p/1000,r,q0)
        last_error=(float(pe),float(re))
        if not np.isfinite(q).all() or not np.isfinite([pe,re]).all():continue
        if np.any(q<JOINT_MIN_DEG-1e-8) or np.any(q>JOINT_MAX_DEG+1e-8):continue
        if np.max(np.abs(q-q0))>max_joint_step+1e-8:
            speed_limited=True;continue
        if pe>.2 or re>.2:continue
        actual,_=fk_jacobian(np.deg2rad(q))
        margin=np.minimum(q-JOINT_MIN_DEG,JOINT_MAX_DEG-q)
        boundary=np.where(margin<.05)[0]
        limited=fraction<1 or (len(boundary)>0 and (pe>.02 or re>.02))
        return q,actual[:3,3]*1000,actual[:3,:3],dict(limited=bool(limited),fraction=fraction,
            cause='joint_limit' if len(boundary)>0 and limited else 'joint_speed' if speed_limited else 'solver' if limited else None,
            position_error_mm=float(pe),rotation_error_deg=float(re),
            accepted_target_mm=p.tolist(),accepted_target_R=r.tolist(),
            joints=[int(i+1) for i in np.where(margin<1)[0]])
    margin=np.minimum(q0-JOINT_MIN_DEG,JOINT_MAX_DEG-q0)
    return q0.copy(),p0,r0,dict(limited=True,fraction=0.,position_error_mm=last_error[0],
        cause='joint_limit' if np.any(margin<1) else 'joint_speed' if speed_limited else 'solver',
        accepted_target_mm=p0.tolist(),accepted_target_R=r0.tolist(),
        rotation_error_deg=last_error[1],joints=[int(i+1) for i in np.where(margin<1)[0]])
