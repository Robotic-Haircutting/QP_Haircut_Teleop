"""Persist haircut coverage and drawn segments, independently of clutch state."""
import hashlib,os,time
from pathlib import Path
import mujoco
import numpy as np
from haircut_sim.coverage import tip_pos
from haircut_sim.viewer import TrailOverlay,DOT_COVERED,PATH_NEAR,PATH_FAR


class PersistentTrailOverlay(TrailOverlay):
    def __init__(self,scene,cfg,path):
        self.path=Path(path);self.segments=[];self.colors=[];self.last_save=0
        super().__init__(scene,cfg)

    def clear(self):
        self.segments=[];self.colors=[]
        super().clear()

    def signature(self):
        return hashlib.sha256(self.cov.pts.tobytes()).hexdigest() if self.cov is not None else ''

    def attach(self,scn):
        super().attach(scn)
        self.restore()

    def restore(self):
        if not self.path.exists():return
        try:
            with np.load(self.path,allow_pickle=False) as data:
                if str(data['signature'])!=self.signature():return
                covered=data['covered'];segments=data['segments'];colors=data['colors']
                if self.cov is not None and covered.shape!=self.cov.covered.shape:return
                if segments.shape[1:]!=(2,3) or colors.shape!=(len(segments),4):return
                if self.cov is not None:
                    self.cov.covered[:]=covered
                    for slot,index in enumerate(self.dot_idx):
                        if covered[index]:self.scn.geoms[slot].rgba[:]=DOT_COVERED
                self.segments=[s.copy() for s in segments];self.colors=[c.copy() for c in colors]
                for segment,color in zip(self.segments,self.colors):
                    if self.scn.ngeom>=self.scn.maxgeom:break
                    geom=self.scn.geoms[self.scn.ngeom]
                    mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_SPHERE,np.zeros(3),np.zeros(3),np.eye(3).flatten(),color)
                    width=3. if np.allclose(color,PATH_NEAR) else 1.5
                    mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_LINE,width,*segment)
                    self.scn.ngeom+=1
                self.last=None
        except (OSError,ValueError,KeyError) as exc:
            print('Coverage restore failed:',exc,flush=True)

    def update(self):
        previous=None if self.last is None else self.last.copy()
        before=self.scn.ngeom
        super().update()
        if previous is not None and self.scn.ngeom>before:
            self.segments.append(np.array([previous,self.last]))
            self.colors.append(self.scn.geoms[self.scn.ngeom-1].rgba.copy())

    def update_if_cutting(self,state,live):
        # Always locate the tip marker, but arming control alone is not a cut.
        p=tip_pos(self.scene.model,self.scene.data,self.offset)
        if self.scn is not None:self.scn.geoms[len(self.dot_idx)].pos[:]=p
        if live and state=='FOLLOWING':
            if self.last is None:self.last=p.copy()
            else:self.update()
        else:self.last=None
        self.save()

    def save(self,force=False):
        if not force and time.monotonic()-self.last_save<1:return
        self.path.parent.mkdir(parents=True,exist_ok=True)
        temporary=self.path.with_suffix('.tmp')
        with temporary.open('wb') as output:
            np.savez(output,signature=self.signature(),covered=self.cov.covered if self.cov is not None else np.zeros(0,bool),
                     segments=np.asarray(self.segments,float).reshape(-1,2,3),colors=np.asarray(self.colors,float).reshape(-1,4))
        os.replace(temporary,self.path);self.last_save=time.monotonic()
