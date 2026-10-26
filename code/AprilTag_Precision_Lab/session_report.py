"""Per-session results in results/<algorithm>_<date>_<time>/, in the mujoco_haircut output format.

For each finished session (Stop & Save / Finish & Rest) this writes one folder containing:
  pnd_20261006_233015.csv               target / command / measured pose, errors, qc1-6, q1-6, solve_ms
  pnd_20261006_233015_figures/          IEEE figures, PDF + 600 dpi PNG
  pnd_20261006_233015.html              rd / rc / r comparison (long_tracking_report)
  pnd_20261006_233015_ieee.html         figure report (tracking_report)
  pnd_20261006_233015_haircut.html      hair coverage report (report.render)
  video_20261006_233015.mp4             MuJoCo window recording of this session (2026-10-07, same layout as SO-101:
                                        sim views left | main view | real camera right), cut from results/twin_video
mujoco_haircut is only imported, never modified.
"""
import csv,json,os,shutil,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PROJECTS=Path(os.environ.get('HHE_ROBOT_ROOT',str(ROOT.parent)))
HAIRCUT=PROJECTS/'mujoco_haircut'
NAMES=['tracking_xyz','tracking_orientation','tracking_error','solver_performance','joint_tracking_1_3','joint_tracking_4_6','coverage_views','coverage_ratio']
# Follow limits used by RobotExperiment (robot_bridge.py, RobotExperiment.__init__): reports split the errors by
# whether a joint sat on these limits, as the SO-101 reports do with their own limits.
FOLLOW_LIMITS_DEG=[[-148,148],[0,178],[-168,0],[-98,98],[-68,68],[-118,118]]
TWIN_VIDEO=ROOT/'results'/'twin_video'

def _segments(meta_path):
    """(start_wall, [(file, start_wall, end_wall), ...]) of one continuous twin recording."""
    meta=json.loads(meta_path.read_text(encoding='utf-8'));t0=float(meta['start_wall']);rows=[]
    lst=meta_path.parent/meta['segment_list']
    if lst.exists():
        with lst.open(encoding='utf-8') as fh:
            for row in csv.reader(fh):
                if len(row)>=3 and (meta_path.parent/row[0]).exists():rows.append((meta_path.parent/row[0],t0+float(row[1]),t0+float(row[2])))
    return rows

def session_video(started,ended,session,wait_s=15.0):
    """Cut [started, ended] (wall clock) out of the continuous MuJoCo-window recording into the session folder.
    Waits up to wait_s for the segment that contains `ended` to be finished by the viewer. Returns the path or None."""
    sys.path.insert(0,str(HAIRCUT)) if str(HAIRCUT) not in sys.path else None
    from haircut_sim.recorder import _ffmpeg
    ff=_ffmpeg()
    if not ff or not TWIN_VIDEO.exists():return None
    deadline=time.time()+wait_s;best=[]
    while True:
        best=[]
        for meta in sorted(TWIN_VIDEO.glob('twin_*.json')):
            try:rows=[r for r in _segments(meta) if r[2]>started and r[1]<ended]
            except (OSError,ValueError,KeyError):continue
            if sum(min(e,ended)-max(s,started) for _,s,e in rows)>sum(min(e,ended)-max(s,started) for _,s,e in best):best=rows
        if (best and best[-1][2]>=ended) or time.time()>deadline:break
        time.sleep(1.0)
    if not best:return None
    out=session/f"video_{time.strftime('%Y%m%d_%H%M%S',time.localtime(started))}.mp4"
    listing=session/'.video_concat.txt'
    listing.write_text(''.join(f"file '{p.as_posix()}'\n" for p,_,_ in best),encoding='utf-8')
    skip=max(0.0,started-best[0][1]);length=max(0.1,min(ended,best[-1][2])-max(started,best[0][1]))
    try:
        subprocess.run([ff,'-loglevel','error','-y','-f','concat','-safe','0','-i',str(listing),'-ss',f'{skip:.3f}','-t',f'{length:.3f}',
                        '-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p',str(out)],check=True,timeout=600,
                       creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    finally:
        listing.unlink(missing_ok=True)
    return out if out.exists() else None

def _stem(out,algorithm,started):
    stem=f"{algorithm}_{time.strftime('%Y%m%d_%H%M%S',time.localtime(started))}";i=2;name=stem
    while (out/name).exists():name=f'{stem}_{i}';i+=1
    return name

def _coverage_reference(alignment):
    from head_alignment import TOOL_MOUNTED
    if TOOL_MOUNTED:return 'clipper tip (mujoco_haircut scene_config tool.tip_offset_m)'
    return 'flange center (no tool)' if alignment else 'scene_config tool tip'

def _coverage(t,q,alignment):
    """report.compute, with the session's head alignment applied (clipper tip when head_alignment.TOOL_MOUNTED, else flange)."""
    import mujoco,numpy as np
    from haircut_sim.coverage import HairCoverage,tip_offset,tip_pos
    from haircut_sim.scene import Scene,load_config
    from head_alignment import flange_config,apply_alignment
    cfg=load_config(HAIRCUT/'scene_config.json')
    if alignment:flange_config(cfg)
    scene=Scene(cfg);m,d=scene.model,scene.data;scene.forward()
    if alignment:apply_alignment(m,d,alignment)
    cov=HairCoverage(m,d,cfg);off=tip_offset(cfg)
    tips,dists,ratio=np.zeros((len(q),3)),np.zeros(len(q)),np.zeros(len(q))
    for i,qi in enumerate(q):
        scene.set_piper(qi);mujoco.mj_kinematics(m,d)
        tips[i]=tip_pos(m,d,off);new,dists[i]=cov.update(tips[i])
        ratio[i]=cov.ratio if len(new) or i==0 else ratio[i-1]
    near=dists<cov.radius;dt=np.diff(t,append=t[-1])
    return {'cov':cov,'tips':tips,'dists':dists,'near':near,'t':t,'ratio':ratio,
            'path_m':float(np.sum(np.linalg.norm(np.diff(tips,axis=0),axis=1))),
            'contact_s':float(dt[near].sum()),'duration_s':float(t[-1]-t[0])}

def export_session(signals,mode,started,algorithm='pnd',alignment=None,out=None):
    """Copy the session CSV into results/ and generate figures + HTML reports. Returns the main HTML path."""
    signals=Path(signals);out=Path(out or ROOT/'results');out.mkdir(parents=True,exist_ok=True)
    cache=out/'.matplotlib-cache';cache.mkdir(exist_ok=True);os.environ.setdefault('MPLCONFIGDIR',str(cache))
    if str(HAIRCUT) not in sys.path:sys.path.insert(0,str(HAIRCUT))
    import matplotlib
    matplotlib.use('Agg')
    from haircut_sim import figures
    from haircut_sim.report import load_joints,render
    from haircut_sim.tracking_report import make_tracking_report
    from haircut_sim.long_tracking_report import make_long_report
    from paper_figures import PLOT_LOCK
    with signals.open(encoding='utf-8') as file:
        if sum(1 for _ in file)<3:raise ValueError('No valid control data; cannot generate results')
    name=_stem(out,algorithm if mode=='hardware' else f'{algorithm}_mock',started)
    session=out/name;session.mkdir()
    csv_path=session/f'{name}.csv';shutil.copyfile(signals,csv_path)
    folder=session/f'{name}_figures';folder.mkdir(exist_ok=True)
    with PLOT_LOCK,matplotlib.rc_context():
        figures._style();data=figures._load_log(csv_path)
        figures.tracking_figures(data,folder)
        t,q=load_joints(csv_path);coverage=_coverage(t,q,alignment)
        figures.coverage_figures(coverage,folder)
    long_html=make_long_report(csv_path,open_browser=False,joint_limits=FOLLOW_LIMITS_DEG)
    ieee_html=make_tracking_report(csv_path,figures_ready=True,joint_limits=FOLLOW_LIMITS_DEG)
    haircut_html=render(csv_path,coverage)
    video=None
    try:video=session_video(started,signals.stat().st_mtime,session)  # signals.csv is closed when the session ends
    except Exception as e:print(f'Session video not cut: {type(e).__name__}: {e}',flush=True)
    (folder/'manifest.json').write_text(json.dumps(dict(csv=csv_path.name,source=str(signals.parent),mode=mode,solver='PND box QP',dpi=600,figures=NAMES,
        actual_source='Piper CAN end-pose and joint feedback' if mode=='hardware' else 'mock feedback',
        head_alignment=alignment.get('revision') if alignment else None,coverage_reference=_coverage_reference(alignment),
        coverage_ratio=float(coverage['cov'].ratio),reports=[long_html.name,ieee_html.name,haircut_html.name],
        video=video.name if video else None,joint_limits_deg=FOLLOW_LIMITS_DEG),ensure_ascii=False,indent=2),encoding='utf-8')
    return ieee_html

if __name__=='__main__':
    # python session_report.py results/experiments/<session>   (re-export an old session)
    for argument in sys.argv[1:]:
        folder=Path(argument);meta=json.loads((folder/'session.json').read_text(encoding='utf-8'))
        print(export_session(folder/'signals.csv',meta['mode'],folder.stat().st_mtime),flush=True)
