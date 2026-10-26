"""Translation-only crown alignment. All geometry is in MuJoCo world metres.

TOOL_MOUNTED (user 2026-10-07: the same hair clipper as the SO-101 haircut is mounted): the tool is the one in
mujoco_haircut/scene_config.json (tool.tip_offset_m, the 'clipper' geom). The model keeps the clipper and fingers,
the head guard includes them, coverage is measured at the clipper tip and Align Head puts the crown 10 mm below the
clipper tip. Set TOOL_MOUNTED = False to go back to the earlier tool-free, flange-referenced bench setup.
"""
import os,time
import numpy as np
import mujoco

TOOL_MOUNTED=os.environ.get('APRILTAG_TOOL_MOUNTED','1')!='0'  # env APRILTAG_TOOL_MOUNTED=0 -> old tool-free setup

def tool_offset_mm():
    """Clipper tip in the flange frame (mm), from mujoco_haircut/scene_config.json tool.tip_offset_m; zero when
    TOOL_MOUNTED is off. Phone teleoperation steers this point (user 2026-10-07: control the clipper tip, as SO-101)."""
    if not TOOL_MOUNTED:return np.zeros(3)
    import json
    from pathlib import Path
    root=Path(os.environ.get('HHE_ROBOT_ROOT',str(Path(__file__).resolve().parent.parent)))
    cfg=json.loads((root/'mujoco_haircut'/'scene_config.json').read_text(encoding='utf-8'))
    return np.asarray(cfg.get('tool',{}).get('tip_offset_m',[0.,0.,0.]),float)*1000.

def flange_config(cfg):
    if TOOL_MOUNTED:return cfg  # keep the shared clipper tip
    cfg.setdefault('tool',{})['tip_offset_m']=[0.,0.,0.]
    return cfg

def hide_absent_tool(model):
    if TOOL_MOUNTED:return  # clipper and fingers are physically there
    for gid in range(model.ngeom):
        name=model.body(model.geom_bodyid[gid]).name
        # Since 2026-10-07 the mujoco_haircut scene includes the clipper cylinder (geom "clipper"); no tool is mounted here, so hide it as well
        if name in ('piper/link7','piper/link8') or model.geom(gid).name=='clipper':
            model.geom_rgba[gid,3]=0

def crown(model,data):
    gid=model.geom('head').id;mid=int(model.geom_dataid[gid])
    first=int(model.mesh_vertadr[mid]);count=int(model.mesh_vertnum[mid])
    verts=model.mesh_vert[first:first+count]@data.geom_xmat[gid].reshape(3,3).T+data.geom_xpos[gid]
    top=float(verts[:,2].max());p=verts[verts[:,2]>=top-.0005].mean(axis=0);p[2]=top
    return p

def make_alignment(cfg,model,data,flange_mm,joints_deg):
    p=np.asarray(flange_mm,float)
    if p.shape!=(3,) or not np.isfinite(p).all():raise ValueError('Invalid flange feedback.')
    base=cfg['piper'];yaw=np.deg2rad(base.get('yaw_deg',0))
    rotation=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
    world=np.asarray(base.get('pos',[0,0,0]))+rotation@(p/1000)
    model_flange=data.xpos[model.body('piper/link6').id].copy()
    disagreement=float(np.linalg.norm(model_flange-world)*1000)
    if disagreement>5:raise ValueError('MuJoCo flange and measured flange differ by more than 5 mm. Check model and joint zero.')
    reference_point=world;reference='flange_center'
    if TOOL_MOUNTED:
        # Crown goes 10 mm below the clipper tip: measured flange + current flange rotation * tool offset.
        from haircut_sim.coverage import tip_offset
        R=data.xmat[model.body('piper/link6').id].reshape(3,3)
        reference_point=world+R@tip_offset(cfg);reference='tool_tip'
    desired=reference_point-np.array([0,0,.010]);delta=desired-crown(model,data)
    body=model.body('head').id
    return dict(revision=str(time.time_ns()),reference=reference,gap_mm=10.,tool_tip_world_m=reference_point.tolist(),
        flange_base_mm=p.tolist(),flange_world_m=world.tolist(),crown_world_m=desired.tolist(),
        head_body_pos_m=(model.body_pos[body]+delta).tolist(),joints_deg=np.asarray(joints_deg).tolist(),
        model_flange_world_m=model_flange.tolist(),model_flange_error_mm=disagreement,
        head_mesh=cfg['head'].get('file'),head_scale=cfg['head'].get('scale'),head_euler_deg=cfg['head'].get('euler_deg'),
        translation_only=True,physical_cutting_validated=False)

def apply_alignment(model,data,record):
    position=np.asarray(record['head_body_pos_m'],float)
    if position.shape!=(3,) or not np.isfinite(position).all():raise ValueError('Invalid head alignment.')
    body=model.body('head').id;delta=position-model.body_pos[body]
    model.body_pos[body]=position;mujoco.mj_forward(model,data)
    return delta

def translate_coverage(trail,delta):
    if trail is None:return
    cov=trail.cov
    if cov is not None:
        translated=set()
        for name in ('pts','all_verts','hair_verts','center'):
            value=getattr(cov,name,None)
            if isinstance(value,np.ndarray) and id(value) not in translated:value+=delta;translated.add(id(value))
        if trail.scn is not None:
            for slot,index in enumerate(trail.dot_idx):trail.scn.geoms[slot].pos[:]=cov.pts[index]
    # Executed paths remain in robot-base/world coordinates; covered indices survive.
    trail.last=None;trail.save(force=True)
