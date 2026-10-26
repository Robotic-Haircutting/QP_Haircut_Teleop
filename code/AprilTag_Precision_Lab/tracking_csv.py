"""CSV field names/units compatible with piper_teleop/tracking_log.py."""
import csv,time
import numpy as np
from piper_twin import fk_jacobian,rot_log
POSE=['x','y','z','wx','wy','wz']
HEADER=['t']+[f'{prefix}_{k}' for prefix in ['rd','rc','r'] for k in POSE]+['e_pos_mm','e_rot_deg','ec_pos_mm','ec_rot_deg']+[f'qc{i}' for i in range(1,7)]+[f'q{i}' for i in range(1,7)]+['solve_ms','qp_iters']

class TrackingCSV:
    def __init__(self,path):
        self.file=open(path,'w',newline='',encoding='utf-8');self.writer=csv.writer(self.file);self.writer.writerow(HEADER);self.started=time.perf_counter();self.last_flush=self.started
    def log(self,p_mm,r,q,solve_ms,qp_iters=float('nan')):
        t,_=fk_jacobian(np.deg2rad(q))
        desired=np.r_[p_mm/1000,rot_log(r)];actual=np.r_[t[:3,3],rot_log(t[:3,:3])]
        error=[float(np.linalg.norm(t[:3,3]*1000-p_mm)),float(np.degrees(np.linalg.norm(rot_log(r@t[:3,:3].T))))]
        # Simulator has no independent measured hardware state: r==rc and q==qc.
        self.writer.writerow([time.perf_counter()-self.started,*desired,*actual,*actual,*error,*error,*q,*q,solve_ms,qp_iters])
        if time.perf_counter()-self.last_flush>.5:self.file.flush();self.last_flush=time.perf_counter()
    def close(self):self.file.close()

def plot_tracking(path):
    from paper_figures import make_session_figures
    return make_session_figures(path)
