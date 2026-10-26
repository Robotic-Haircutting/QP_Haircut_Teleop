"""Validate a slow joint-space entry to the shared MuJoCo working pose."""
import numpy as np
from head_guard import HeadGuard
from sim_bridge import PROJECTS
from initial_pose import INITIAL_JOINTS_DEG
WORKING_JOINTS=np.array(INITIAL_JOINTS_DEG,float)

def initialization_plan(q,limits,alignment=None,enforce_head=True):
    q=np.asarray(q,float);limits=np.asarray(limits,float)
    if q.shape!=(6,) or not np.isfinite(q).all():raise ValueError('Invalid initial joint feedback.')
    # Encoder/servo rounding can put a stopped joint just outside the
    # conservative follow limit. Accept at most 0.05 deg for homing entry;
    # every planned waypoint still has to be inside the original limits.
    # Folded J2/J3 retain their validated 3-degree entry allowance.
    allowance=np.array([.05,3,3,.05,.05,.05])
    if np.any(q<limits[:,0]-allowance) or np.any(q>limits[:,1]+allowance):raise ValueError('Start pose outside validated homing envelope. Use teach mode to prepare it.')
    if not enforce_head:
        if np.any(WORKING_JOINTS<limits[:,0]) or np.any(WORKING_JOINTS>limits[:,1]):raise ValueError('Saved initial pose exceeds joint limits.')
        return [WORKING_JOINTS.copy()],None
    guard=HeadGuard(PROJECTS/'mujoco_haircut',flange=True,alignment=alignment)
    start_clearance=guard.clearance(q)
    initial_clearance=guard.clearance(WORKING_JOINTS)
    if initial_clearance['distance_mm']<10:
        raise ValueError(f"Saved initial pose conflicts with the head model ({initial_clearance['part']}, clearance {initial_clearance['distance_mm']:.1f} mm; required 10 mm). Adjust the physical head and Align Head again, or select a safe initial pose. Arm held.")
    if start_clearance['distance_mm']<10:
        raise ValueError(f"Current pose is inside the head protection margin ({start_clearance['distance_mm']:.1f} mm). A checked retreat is required before returning to initial. Arm held.")
    path=guard.plan_return(q,WORKING_JOINTS)
    if not path:raise ValueError('No home path found with model head clearance.')
    for waypoint in path:
        if np.any(waypoint<limits[:,0]) or np.any(waypoint>limits[:,1]):raise ValueError('Home path exceeds joint limits.')
    return path,guard

def initialization_step(q,target,dt,speed=15):
    q=np.asarray(q,float);delta=np.asarray(target)-q
    return q+delta*min(1,speed*min(max(dt,.001),.04)/max(np.max(np.abs(delta)),1e-12))
