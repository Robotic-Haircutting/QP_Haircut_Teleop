"""Input outlier gate and adaptive low-pass in phone coordinates, before gain."""
import math
import numpy as np
from scipy.spatial.transform import Rotation

class PhoneMotionFilter:
    def __init__(self):self.reset()
    def reset(self):
        self.raw_p=None;self.raw_r=None;self.p=None;self.r=None;self.stamp=None
        self.velocity=np.zeros(3);self.angular_velocity=np.zeros(3);self.rejections=0;self.total_rejections=0
        self.output_p=None;self.output_r=None
        self.last_info=dict(accepted=False)
    @staticmethod
    def alpha(cutoff,dt):return 1-math.exp(-2*math.pi*cutoff*dt)
    def update(self,p,r,stamp,gain=3.,ambiguous=False):
        p=np.asarray(p,float);r=np.asarray(r,float)
        if p.shape!=(3,) or r.shape!=(3,3) or not np.isfinite(p).all() or not np.isfinite(r).all():raise ValueError('Invalid phone pose.')
        if self.raw_p is None:
            self.raw_p=p.copy();self.raw_r=r.copy();self.p=p.copy();self.r=r.copy();self.stamp=stamp
            self.output_p=p.copy();self.output_r=r.copy()
            self.last_info=dict(accepted=True,position_cutoff_hz=1.,rotation_cutoff_hz=1.,rejected_frames=0)
            return self.output_p.copy(),self.output_r.copy(),self.last_info.copy()
        dt=max(.001,min(float(stamp-self.stamp),.8))
        delta=p-self.raw_p;rotvec=Rotation.from_matrix(r@self.raw_r.T).as_rotvec()
        distance=float(np.linalg.norm(delta));angle=float(np.rad2deg(np.linalg.norm(rotvec)))
        position_limit=80. if ambiguous else max(80.,min(140.,600*dt))
        rotation_limit=max(45.,min(75.,180*dt))
        if distance>position_limit or angle>rotation_limit:
            self.rejections+=1;self.total_rejections+=1
            self.last_info=dict(accepted=False,rejected_frames=self.rejections,position_jump_mm=distance,rotation_jump_deg=angle,total_rejections=self.total_rejections)
            return self.output_p.copy(),self.output_r.copy(),self.last_info.copy()
        self.rejections=0
        a=self.alpha(1.,dt)
        self.velocity+=a*(delta/dt-self.velocity);self.angular_velocity+=a*(rotvec/dt-self.angular_velocity)
        # At rest, high gains need stronger noise suppression. During motion,
        # the velocity-dependent cutoff rises to retain responsiveness.
        rest_cutoff=.4/(1+.05*max(0.,float(gain)-1))  # was 1.0 Hz: at ~3 Hz phone samples it barely smoothed
        cutoff=rest_cutoff+.02*np.linalg.norm(self.velocity)
        rotation_cutoff=1.+.04*np.rad2deg(np.linalg.norm(self.angular_velocity))
        self.p+=self.alpha(cutoff,dt)*(p-self.p)
        error=Rotation.from_matrix(r@self.r.T).as_rotvec()
        self.r=Rotation.from_rotvec(error*self.alpha(rotation_cutoff,dt)).as_matrix()@self.r
        self.raw_p=p.copy();self.raw_r=r.copy();self.stamp=stamp
        # Continuous hysteresis on the filtered pose. Unlike rounding or a
        # per-frame threshold, slow real motion accumulates and is not lost.
        # Bound translation tolerance in robot space even at high phone gain.
        position_band=min(2.,2./max(.15,float(gain)))  # tiny-jitter band: 2 mm in robot space
        rotation_band=np.deg2rad(.35)
        step=self.p-self.output_p;length=float(np.linalg.norm(step))
        if length>position_band:self.output_p+=step*(1-position_band/length)
        turn=Rotation.from_matrix(self.r@self.output_r.T).as_rotvec();angle=float(np.linalg.norm(turn))
        if angle>rotation_band:self.output_r=Rotation.from_rotvec(turn*(1-rotation_band/angle)).as_matrix()@self.output_r
        self.last_info=dict(accepted=True,position_band_mm=float(position_band),rotation_band_deg=.35,position_held=bool(length<=position_band),rotation_held=bool(angle<=rotation_band),position_cutoff_hz=float(cutoff),rotation_cutoff_hz=float(rotation_cutoff),rejected_frames=0,total_rejections=self.total_rejections)
        return self.output_p.copy(),self.output_r.copy(),self.last_info.copy()
