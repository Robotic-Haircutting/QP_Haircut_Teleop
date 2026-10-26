"""Independent measured feedback, commanded FK and requested pose, legacy CSV units."""
import csv,json,time
from pathlib import Path
import numpy as np
from pnd_tracking import fk_jacobian
from scipy.spatial.transform import Rotation
from tracking_csv import HEADER

class ExperimentLog:
    def __init__(self,root,mode):
        self.root=Path(root)/f'{mode}_pnd_{time.time_ns()}';self.root.mkdir(parents=True)
        self.mode=mode;self.wall_started=time.time();self.started=time.perf_counter();self.path=self.root/'signals.csv'
        self.file=self.path.open('w',newline='',encoding='utf-8');self.writer=csv.writer(self.file);self.writer.writerow(HEADER)
        self.events=(self.root/'events.jsonl').open('w',encoding='utf-8',buffering=1)
        self.diagnostics=(self.root/'solver.jsonl').open('w',encoding='utf-8',buffering=1);self.last_flush=self.started;self.closed=False
        (self.root/'session.json').write_text(json.dumps(dict(mode=mode,solver='PND box QP',actual_source='Piper CAN end-pose and joint feedback' if mode=='hardware' else 'mock feedback',position_unit='m',rotation_unit='rotation vector rad',joint_unit='deg',scope='tool-free bench following; head registration unavailable'),indent=2),encoding='utf-8')
    def event(self,name,**fields):self.events.write(json.dumps(dict(t=time.perf_counter()-self.started,event=name,**fields),ensure_ascii=False)+'\n')
    def log(self,p,r,q_command,feedback,info):
        cmd,_=fk_jacobian(np.deg2rad(q_command));actual_p=feedback['p'];actual_r=feedback['r']
        pack=lambda p,r:np.r_[np.asarray(p)/1000,Rotation.from_matrix(r).as_rotvec()]
        errors=lambda p1,r1:[np.linalg.norm(np.asarray(p)-p1),np.rad2deg(Rotation.from_matrix(r@r1.T).magnitude())]
        self.writer.writerow([time.perf_counter()-self.started,*pack(p,r),*pack(cmd[:3,3]*1000,cmd[:3,:3]),*pack(actual_p,actual_r),*errors(actual_p,actual_r),*errors(cmd[:3,3]*1000,cmd[:3,:3]),*q_command,*feedback['q'],info['solve_ms'],info['qp_iters']])
        self.diagnostics.write(json.dumps(dict(t=time.perf_counter()-self.started,feedback_age_s=feedback['age'],**info),ensure_ascii=False)+'\n')
        if time.perf_counter()-self.last_flush>.5:self.file.flush();self.last_flush=time.perf_counter()
    def close(self):
        if not self.closed:
            self.file.close();self.events.close();self.diagnostics.close();self.closed=True
