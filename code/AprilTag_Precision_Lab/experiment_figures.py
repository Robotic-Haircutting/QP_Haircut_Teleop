"""Legacy separate signals plus a full experiment overview (PNG/PDF)."""
import json,os,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

def overview(data,out,save,solver_label='PND'):
    import matplotlib.pyplot as plt
    t=data['t']-data['t'][0];fig,axes=plt.subplots(4,4,figsize=(10,8),layout='constrained')
    colors=['#1f4e9c','#2e8b57','#c0392b'];styles=['-',':','--'];prefixes=['rd','rc','r']
    for i,k in enumerate('xyz'):
        ax=axes[0,i]
        for prefix,c,style in zip(prefixes,colors,styles):ax.plot(t,data[f'{prefix}_{k}']*1000,color=c,ls=style,label={'rd':'Target','rc':'Command','r':'Measured'}[prefix])
        ax.set_title(f'Position {k.upper()}');ax.set_ylabel('mm')
    ref=Rotation.from_rotvec([data[f'rd_w{k}'][0] for k in 'xyz']).as_matrix()
    for prefix,c,style in zip(prefixes,colors,styles):
        vectors=np.stack([data[f'{prefix}_w{k}'] for k in 'xyz'],axis=1);relative=(Rotation.from_rotvec(vectors)*Rotation.from_matrix(ref).inv()).as_rotvec()*180/np.pi
        for i in range(3):axes[1,i].plot(t,relative[:,i],color=c,ls=style)
    for i,k in enumerate('xyz'):axes[1,i].set_title(f'Relative orientation {k.upper()}');axes[1,i].set_ylabel('deg')
    for ax,key,title,unit in [(axes[0,3],'e_pos_mm','Position error','mm'),(axes[1,3],'e_rot_deg','Orientation error','deg'),(axes[3,2],'solve_ms','Solve time','ms'),(axes[3,3],'qp_iters','PND iterations','iterations')]:
        ax.plot(t,data[key]);ax.set_title(title);ax.set_ylabel(unit)
    for i,ax in enumerate([*axes[2],axes[3,0],axes[3,1]],1):
        ax.plot(t,data[f'qc{i}'],color=colors[1],ls=':');ax.plot(t,data[f'q{i}'],color=colors[2],ls='--');ax.set_title(f'Joint {i}');ax.set_ylabel('deg')
    axes[0,0].legend(fontsize=6)
    for ax in axes.ravel():ax.set_xlabel('Time (s)');ax.tick_params(labelsize=6)
    fig.suptitle(solver_label+' teleoperation experiment — requested / commanded / measured',fontsize=11)
    save(fig,out,'overview')

def make_figures(path,mode):
    path=Path(path);project=Path(os.environ.get('HHE_ROBOT_ROOT',str(Path(__file__).resolve().parent.parent)))/'mujoco_haircut';sys.path.insert(0,str(project))
    cache=path.parent.parent/'.matplotlib-cache';cache.mkdir(exist_ok=True);os.environ.setdefault('MPLCONFIGDIR',str(cache))
    from haircut_sim import figures
    from paper_figures import PLOT_LOCK
    import matplotlib
    out=path.parent/'figures';out.mkdir(exist_ok=True)
    with PLOT_LOCK,matplotlib.rc_context():
        data=figures._load_log(path)
        if not data or not len(data['t']):raise ValueError('No valid control data to plot')
        figures._style();figures.tracking_figures(data,out);overview(data,out,figures._save)
    names=['overview','tracking_xyz','tracking_orientation','tracking_error','solver_performance','joint_tracking_1_3','joint_tracking_4_6']
    manifest=dict(mode=mode,solver='PND box QP',figures=names,dpi=600,actual_source='CAN measured joints and end pose' if mode=='hardware' else 'mock feedback',head_coverage='not measured: tool-free bench experiment')
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>PND experiment overview and signals</title><style>body{font-family:system-ui}img{max-width:100%}</style>'+''.join(f'<h2>{name}</h2><a href="{name}.pdf">PDF</a><img src="{name}.png">' for name in names),encoding='utf-8')
    return out/'overview.png'
