import sys,tempfile
from pathlib import Path
import mujoco,numpy as np
from unittest.mock import patch
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root));sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'mujoco_haircut'))
from haircut_sim.scene import Scene,load_config
from haircut_sim.viewer import PATH_NEAR,apply,DOT_COVERED
from hair_progress import PersistentTrailOverlay

scene=Scene(load_config(None));scene.forward()
with tempfile.TemporaryDirectory() as directory:
    path=Path(directory)/'progress.npz'
    trail=PersistentTrailOverlay(scene,load_config(None),path)
    scn=mujoco.MjvScene(scene.model,maxgeom=10000);trail.attach(scn)
    assert trail.cov is not None and len(trail.cov.pts)>0
    # A stationary tip placed inside the coverage radius must not count as a cut.
    point=trail.cov.pts[0].copy()
    with patch('hair_progress.tip_pos',return_value=point),patch('haircut_sim.viewer.tip_pos',return_value=point):
        trail.update_if_cutting('PAUSED',True)
        trail.update_if_cutting('FOLLOWING',True)
        trail.update_if_cutting('FOLLOWING',True)
        assert not trail.cov.covered.any() and not trail.segments
        moved=point+np.array([.003,0,0])
        with patch('hair_progress.tip_pos',return_value=moved),patch('haircut_sim.viewer.tip_pos',return_value=moved):
            trail.update_if_cutting('FOLLOWING',True)
        assert trail.cov.covered.any() and trail.segments
    trail.clear()
    print('ARMING PASS: paused and stationary following do not mark coverage; actual movement does',flush=True)
    trail.cov.update(trail.cov.pts[0]);covered=trail.cov.covered.copy();ratio=trail.cov.ratio
    trail.segments=[np.array([trail.cov.pts[0],trail.cov.pts[0]+[.003,0,0]])]
    trail.colors=[PATH_NEAR.copy()];trail.save(force=True)
    apply(scene,dict(piper=dict(q=[14.4,49.3,-57.9,0,38.6,0])));scene.forward()
    trail.update_if_cutting('RETURNING',True)
    assert np.array_equal(trail.cov.covered,covered) and len(trail.segments)==1
    fresh=PersistentTrailOverlay(scene,load_config(None),Path(directory)/'next-session.npz')
    fresh.attach(mujoco.MjvScene(scene.model,maxgeom=10000))
    assert not fresh.cov.covered.any() and not fresh.segments
    assert path.exists(),'new session deleted history'
    print('NEW SESSION PASS: fresh session does not restore earlier green cells or trails; previous record remains',flush=True)
    restored=PersistentTrailOverlay(scene,load_config(None),path)
    restored_scn=mujoco.MjvScene(scene.model,maxgeom=10000);restored.attach(restored_scn)
    assert np.array_equal(restored.cov.covered,covered) and restored.cov.ratio==ratio
    assert len(restored.segments)==1 and restored_scn.ngeom==len(restored.dot_idx)+2
    for slot,index in enumerate(restored.dot_idx):
        if covered[index]:assert np.allclose(restored_scn.geoms[slot].rgba,DOT_COVERED)
    before=restored_scn.ngeom
    restored.update_if_cutting('PAUSED',True);restored.update_if_cutting('FOLLOWING',True)
    assert restored_scn.ngeom==before,'return drew a connector across the preserved trail'
    print('PROGRESS PASS: covered cells/colors and trail survive return and renderer restart; reset path adds no cut/connector',flush=True)
    restored.clear();restored.save(force=True)
    cleared=PersistentTrailOverlay(scene,load_config(None),path);cleared.attach(mujoco.MjvScene(scene.model,maxgeom=10000))
    assert not cleared.cov.covered.any() and not cleared.segments
    print('EXPLICIT CLEAR PASS: only explicit C/reset clears the persisted progress',flush=True)
