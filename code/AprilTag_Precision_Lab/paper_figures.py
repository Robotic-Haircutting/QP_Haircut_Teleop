"""Reuse mujoco_haircut's requested IEEE template, without editing that project."""
import csv,json,os,sys,threading
from pathlib import Path
import numpy as np

PLOT_LOCK=threading.Lock()

def make_session_figures(path):
    path=Path(path)
    with path.open(encoding='utf-8') as file:
        if next(csv.DictReader(file),None) is None:return None
    projects=Path(os.environ.get('HHE_ROBOT_ROOT',str(Path(__file__).resolve().parent.parent)))
    shared=projects/'mujoco_haircut'
    root=path.parent.parent
    cache=root/'.matplotlib-cache';cache.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(cache))
    sys.path.insert(0,str(shared)) if str(shared) not in sys.path else None
    from haircut_sim import figures
    from haircut_sim.report import compute,load_joints
    from haircut_sim.scene import load_config
    import matplotlib
    output=root/'plots'/(path.stem+'_figures');output.mkdir(parents=True,exist_ok=True)
    with PLOT_LOCK,matplotlib.rc_context():
        figures._style()
        data=figures._load_log(path)
        original_save=figures._save
        def save(fig,out_dir,name):
            if name=='solver_performance' and not np.isfinite(data['qp_iters']).any():
                fig.axes[1].text(.5,.5,'N/A (DLS IK, not QP)',transform=fig.axes[1].transAxes,ha='center',va='center',fontsize=8)
            original_save(fig,out_dir,name)
        figures._save=save
        try:
            figures.tracking_figures(data,output)
            from experiment_figures import overview
            overview(data,output,figures._save,'PND' if np.isfinite(data['qp_iters']).any() else 'DLS')
            t,q=load_joints(path)
            cfg=load_config(shared/'scene_config.json')
            coverage=compute(t,q,cfg)
            figures.coverage_figures(coverage,output)
        finally:figures._save=original_save
    names=['overview','tracking_xyz','tracking_orientation','tracking_error','solver_performance','joint_tracking_1_3','joint_tracking_4_6','coverage_views','coverage_ratio']
    manifest=dict(csv=str(path),template=str(shared/'haircut_sim/figures.py'),mode='simulation',dpi=600,
        actual_source='simulation FK (r==rc, q==qc)',orientation='relative rotation vector in degrees',
        qp_iterations='PND iterations' if np.isfinite(data['qp_iters']).any() else 'not applicable: DLS IK',coverage='proximity estimate, not proof of hair cutting',
        coverage_ratio=float(coverage['cov'].ratio),figures=names)
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    cards=''.join(f'<section><h2>{n}</h2><a href="{n}.pdf">PDF</a> · <a href="{n}.png">PNG</a><img src="{n}.png"></section>' for n in names)
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Simulation data figures</title><style>body{font-family:system-ui;margin:24px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:20px}img{width:100%}section{border:1px solid #ddd;padding:12px}</style><h1>Simulation data figures</h1><p>Uses the mujoco_haircut paper figure template. Actual is the simulated FK; coverage is only an estimate of proximity to the hair. PNG at 600 dpi; each PDF can be exported separately.</p><main>'+cards+'</main>',encoding='utf-8')
    return output/'tracking_xyz.png'

if __name__=='__main__':
    for argument in sys.argv[1:]:print(make_session_figures(Path(argument)),flush=True)
