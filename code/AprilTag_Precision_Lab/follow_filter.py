"""Time-based smoothing of phone targets between phone samples.

Phone poses arrive at only a few Hz, while the arm loop runs at ~50 Hz. A single 35 ms low-pass left the target
as a staircase (one jump per phone sample). Two cascaded first-order stages with a time constant tied to the
measured phone sample interval turn those steps into a smooth path with continuous velocity.
"""
import numpy as np
from scipy.spatial.transform import Rotation
class FollowFilter:
    def __init__(self,tau=.035):self.tau=tau;self.p=None;self.p1=None;self.r=None
    def reset(self,p,r):self.p=np.asarray(p,float).copy();self.p1=self.p.copy();self.r=np.asarray(r,float).copy()
    def alpha(self,dt,tau=None):return -np.expm1(-max(0.,dt)/(tau or self.tau))
    def position(self,p,dt):
        p=np.asarray(p,float)
        if self.p is None:self.p=p.copy();self.p1=p.copy();return self.p.copy()
        if self.p1 is None:self.p1=self.p.copy()
        a=self.alpha(dt,self.tau/2)
        self.p1=self.p1+a*(p-self.p1);self.p=self.p+a*(self.p1-self.p)
        return self.p.copy()
    def orientation(self,r,dt):
        r=np.asarray(r,float)
        self.r=r.copy() if self.r is None else Rotation.from_rotvec(Rotation.from_matrix(r@self.r.T).as_rotvec()*self.alpha(dt)).as_matrix()@self.r
        return self.r.copy()
