"""PND box QP tracking, with explicit convergence and telemetry."""
import os,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(os.environ.get('HHE_ROBOT_ROOT',str(Path(__file__).resolve().parent.parent)))/'Client_RHCR/piper'))
from piper_twin import fk_jacobian,JOINT_MIN_DEG,JOINT_MAX_DEG
from pnd_ik import PNDIK,PNDIKParams

class SmoothPNDIK(PNDIK):
    """PND solves the same constrained QP with a velocity-continuity cost.

    Acceleration bounds are inside the QP, so smoothing never bypasses angle
    or velocity limits. Emergency stopping remains outside this ramp.
    """
    def __init__(self,params):
        super().__init__(params);self.previous_velocity=None;self.acceleration=np.deg2rad(240.);self.continuity_weight=.003
    def build_qp(self,q,p_d,R_d,dt):
        H,f,lo,hi,e=super().build_qp(q,p_d,R_d,dt)
        previous=np.zeros(len(q)) if self.previous_velocity is None else self.previous_velocity
        H=H+self.continuity_weight*np.eye(len(q));f=f-self.continuity_weight*previous
        ramp_lo=previous-self.acceleration*dt;ramp_hi=previous+self.acceleration*dt
        # Angle/speed limits take priority if noisy feedback makes the ramp
        # infeasible; project the previous velocity back into the hard box.
        anchor=np.clip(previous,lo,hi)
        lo=np.maximum(lo,np.minimum(ramp_lo,anchor));hi=np.minimum(hi,np.maximum(ramp_hi,anchor))
        return H,f,lo,hi,e

class PNDTracker:
    def __init__(self,speed=45,limits=None,gain=6,max_cartesian_speed=40,max_orientation_speed=20):
        limits=np.column_stack([JOINT_MIN_DEG,JOINT_MAX_DEG]) if limits is None else limits
        self.solver=SmoothPNDIK(PNDIKParams(gain=gain,w_rot=.3,damping=.001,iterations=1500,tol=1e-5,max_joint_speed_deg_s=speed,joint_limits_deg=np.asarray(limits).tolist()))
        self.max_cartesian_speed=max_cartesian_speed;self.max_orientation_speed=max_orientation_speed
    def reset(self):self.solver._u=None;self.solver.previous_velocity=None
    def step(self,p_mm,r,q_deg,dt):
        started=time.perf_counter();dt=min(max(float(dt),.001),.04)
        q=np.asarray(q_deg,float);next_q=np.rad2deg(self.solver.step(np.deg2rad(q),np.asarray(p_mm)/1000,r,dt))
        info=self.solver.last_info.copy();info['solve_ms']=(time.perf_counter()-started)*1000
        if not info['converged']:next_q=q.copy();self.reset()
        _,jacobian=fk_jacobian(np.deg2rad(q));velocity=jacobian@np.deg2rad(next_q-q)/dt
        factor=float(min(1.,self.max_cartesian_speed/max(np.linalg.norm(velocity[:3])*1000,1e-12),np.deg2rad(self.max_orientation_speed)/max(np.linalg.norm(velocity[3:]),1e-12)))
        next_q=q+factor*(next_q-q);info['velocity_scale']=factor
        self.solver.previous_velocity=np.deg2rad(next_q-q)/dt
        info.update(velocity_continuity_weight=self.solver.continuity_weight,joint_acceleration_deg_s2=240.)
        tf,_=fk_jacobian(np.deg2rad(next_q));limits=np.asarray(self.solver.p.joint_limits_deg)
        margin=np.min(np.column_stack([next_q-limits[:,0],limits[:,1]-next_q]),axis=1)
        info.update(limited=bool(not info['converged'] or factor<.999999 or np.any(margin<.05)),cause='qp_not_converged' if not info['converged'] else 'joint_limit' if np.any(margin<.05) else 'cartesian_speed' if factor<.999999 else None,
                    joints=[int(i+1) for i in np.where(margin<1)[0]],accepted_target_mm=np.asarray(p_mm).tolist(),accepted_target_R=np.asarray(r).tolist())
        return next_q,tf[:3,3]*1000,tf[:3,:3],info
