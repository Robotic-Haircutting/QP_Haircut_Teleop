"""Conservative model-only head clearance, including swept joint motion."""
import sys,time
from pathlib import Path
import mujoco
import numpy as np

class HeadGuard:
    margin_m=.010
    def __init__(self,scene_dir,flange=False,alignment=None):
        sys.path.insert(0,str(scene_dir))
        from haircut_sim.scene import build_spec,load_config
        from haircut_sim.coverage import tip_offset
        self.cfg=load_config(None)
        if flange:
            from head_alignment import flange_config
            flange_config(self.cfg)
        spec=build_spec(self.cfg)
        # Model the configured flange-to-tip tool shaft with a 5mm radius.
        if np.linalg.norm(tip_offset(self.cfg))>0:
            spec.body('piper/link6').add_geom(name='head_guard_tool',type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                size=[.005,0,0],fromto=np.r_[np.zeros(3),tip_offset(self.cfg)])
        self.model=spec.compile();self.data=mujoco.MjData(self.model)
        if alignment:
            from head_alignment import apply_alignment
            apply_alignment(self.model,self.data,alignment)
        self.head=self.model.geom('head').id
        self.robot=[i for i in range(self.model.ngeom) if self.model.body(self.model.geom_bodyid[i]).name.startswith('piper/')]
        # Tool-free (flange) mode: no fingers and no clipper (mujoco_haircut scene has a 'clipper' geom since 2026-10-07).
        # With the clipper mounted (head_alignment.TOOL_MOUNTED) fingers, clipper and tool shaft all count.
        from head_alignment import TOOL_MOUNTED
        if flange and not TOOL_MOUNTED:self.robot=[i for i in self.robot if self.model.body(self.model.geom_bodyid[i]).name not in ('piper/link7','piper/link8') and self.model.geom(i).name!='clipper']
        self.address=[self.model.jnt_qposadr[self.model.joint(f'piper/joint{i}').id] for i in range(1,7)]
        self.offset=np.array(self.cfg['piper'].get('joint_offsets_deg',[0,2,-2,0,0,0]))
        # 2m bounds a point's lever arm about any joint in this Piper model.
        # Restrict the model dimensions before relying on that swept bound.
        for gid in self.robot:
            radius=float(self.model.geom_rbound[gid])+np.linalg.norm(self.model.geom_pos[gid])
            body=int(self.model.geom_bodyid[gid])
            while body:
                radius+=np.linalg.norm(self.model.body_pos[body]);body=int(self.model.body_parentid[body])
            if radius>2.:raise ValueError('Piper geometry exceeds validated 2m swept bound')
        self.lever_bound=2.
        self.joint_levers=[]
        for i in range(1,7):
            jid=self.model.joint(f'piper/joint{i}').id;jbody=int(self.model.jnt_bodyid[jid]);lever=0.
            for gid in self.robot:
                body=int(self.model.geom_bodyid[gid]);radius=float(self.model.geom_rbound[gid])+np.linalg.norm(self.model.geom_pos[gid])+np.linalg.norm(self.model.jnt_pos[jid])
                while body and body!=jbody:
                    radius+=np.linalg.norm(self.model.body_pos[body]);body=int(self.model.body_parentid[body])
                if body==jbody:lever=max(lever,radius)
            self.joint_levers.append(lever)
        self.joint_levers=np.asarray(self.joint_levers)

        self.last=None;self.last_result=None
    def clearance(self,q):
        q=np.asarray(q,float)
        if self.last is not None and np.array_equal(q,self.last):return self.last_result.copy()
        self.data.qpos[self.address]=np.deg2rad(q+self.offset);mujoco.mj_kinematics(self.model,self.data)
        best=.25;part='none'
        for gid in self.robot:
            # Bounding spheres only skip pairs whose lower bound exceeds best.
            lower=np.linalg.norm(self.data.geom_xpos[gid]-self.data.geom_xpos[self.head])-self.model.geom_rbound[gid]-self.model.geom_rbound[self.head]
            if lower>best:continue
            distance=float(mujoco.mj_geomDistance(self.model,self.data,gid,self.head,best,None))
            if distance<best:
                best=distance;part=self.model.geom(gid).name or self.model.body(self.model.geom_bodyid[gid]).name
        result=dict(distance_mm=best*1000,part=part,minimum_mm=10.)
        self.last=q.copy();self.last_result=result
        return result.copy()
    def edge_safe(self,a,b,deadline=None):
        a=np.asarray(a,float);b=np.asarray(b,float)
        if not np.isfinite(a).all() or not np.isfinite(b).all():return False
        deadline=deadline if deadline is not None else time.monotonic()+.025
        da=self.clearance(a)['distance_mm']/1000;db=self.clearance(b)['distance_mm']/1000
        def visit(x,y,dx,dy,depth):
            if deadline is not None and time.monotonic()>deadline:return False
            if min(dx,dy)<self.margin_m:return False
            bound=float(self.joint_levers@np.abs(np.deg2rad(y-x)))
            # Any swept point is within bound/2 of its nearest endpoint.
            if min(dx,dy)-bound/2>=self.margin_m:return True
            if depth>=24:return False
            mid=(x+y)/2;dm=self.clearance(mid)['distance_mm']/1000
            return visit(x,mid,dx,dm,depth+1) and visit(mid,y,dm,dy,depth+1)
        return visit(a,b,da,db,0)
    def filter_step(self,a,b):
        deadline=time.monotonic()+.025
        a=np.asarray(a,float);b=np.asarray(b,float)
        before=self.clearance(a);after=self.clearance(b)
        # Hold 1mm outside the hard shell to avoid expensive near-zero
        # clearance certificates and to keep a usable retreat/tangent step.
        floor=11. if before['distance_mm']>=11 else before['distance_mm']
        if before['distance_mm']<10:
            return a.copy(),dict(**before,blocked=True,retreat=False)
        if after['distance_mm']>=floor and self.edge_safe(a,b,deadline):
            return b.copy(),dict(**after,blocked=False,retreat=after['distance_mm']>before['distance_mm'])
        for fraction in (.5,.25,.125,.0625):
            if time.monotonic()>deadline:break
            q=a+(b-a)*fraction;result=self.clearance(q)
            if result['distance_mm']>=floor and self.edge_safe(a,q,deadline):
                return q,dict(**result,blocked=True,retreat=result['distance_mm']>before['distance_mm'])
        # Escape a legacy pose sitting micrometres outside the hard shell.
        # A tiny step bounded by the existing gap is certifiably outside 10mm
        # without unbounded subdivision; repeated outward inputs open the gap.
        if 10<before['distance_mm']<11 and after['distance_mm']>before['distance_mm']:
            bound=float(self.joint_levers@np.abs(np.deg2rad(b-a)))*1000
            fraction=min(1.,.9*(before['distance_mm']-10)/max(bound,1e-12))
            q=a+(b-a)*fraction;result=self.clearance(q)
            if result['distance_mm']>before['distance_mm']:
                return q,dict(**result,blocked=False,retreat=True)
        # If phone rotation blocks an otherwise outward translation, the
        # caller can retry that translation with the tool orientation held.
        return a.copy(),dict(**before,blocked=True,retreat=False)
    def plan_return(self,start,home):
        deadline=time.monotonic()+3
        if self.clearance(start)['distance_mm']<10:return None
        if self.edge_safe(start,home,deadline):return [np.asarray(home).copy()]
        # Detours vary shoulder height and base azimuth; every edge is checked.
        from reachable_step import JOINT_MIN_DEG,JOINT_MAX_DEG
        for shoulder in (-20.,20.,-40.):
            for heading in (-35.,35.,-70.,70.):
                waypoint=np.asarray(start).copy();waypoint[0]+=heading;waypoint[1]+=shoulder
                waypoint=np.clip(waypoint,JOINT_MIN_DEG,JOINT_MAX_DEG)
                approach=np.asarray(home).copy();approach[0]=waypoint[0];approach[1]=waypoint[1]
                path=[waypoint,approach,np.asarray(home).copy()]
                if time.monotonic()>deadline:return None
                if all(self.edge_safe(a,b,deadline) for a,b in zip([start]+path,path)):return path
        return None
