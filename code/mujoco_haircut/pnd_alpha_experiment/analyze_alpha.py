"""Compare single-iteration PND runs. Each subfolder of results/ (e.g. a0.2_fps100) holds the pnd_*.csv of one run (also in nested folders).

Prints, per run: samples, median sampling interval, mean feedback position/orientation error,
joint-acceleration RMS (measured and commanded joints), and the Away/Near split
(near = any commanded joint within 0.5 deg of its limit). Also writes summary.csv next to logs/.
"""
import csv, sys
from pathlib import Path
import numpy as np

# Same as --robot.joint_limits_deg in run_alpha*.bat (J5 = ±65°)
LIM = np.array([[-150,150],[15,150],[-160,-15],[-100,100],[-65,65],[-100,100]], float)

def load(path):
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    col = lambda k: np.array([float(r[k]) for r in rows])
    return {k: col(k) for k in rows[0].keys()} if rows else None

def acc_rms(t, Q):
    if len(t) < 5: return float('nan')
    v = np.gradient(Q, t, axis=0); a = np.gradient(v, t, axis=0)
    return float(np.sqrt(np.mean(a ** 2)))

def summarize(name, d):
    t = d['t']; keep = np.r_[True, np.diff(t) > 0]
    d = {k: v[keep] for k, v in d.items()}; t = d['t']
    qm = np.stack([d[f'q{i}'] for i in range(1, 7)], 1)
    qc = np.stack([d[f'qc{i}'] for i in range(1, 7)], 1)
    near = np.any((qc <= LIM[:, 0] + 0.5) | (qc >= LIM[:, 1] - 0.5), axis=1)
    ep, er = d['e_pos_mm'], d['e_rot_deg']
    m = lambda x, s: float(np.mean(x[s])) if np.any(s) else float('nan')
    return dict(run=name, samples=len(t), dt_ms=float(np.median(np.diff(t)) * 1e3),
                pos_mm=float(ep.mean()), rot_deg=float(er.mean()),
                acc_rms_meas=acc_rms(t, qm), acc_rms_cmd=acc_rms(t, qc),
                near_pct=float(near.mean() * 100),
                pos_away=m(ep, ~near), pos_near=m(ep, near), rot_away=m(er, ~near), rot_near=m(er, near),
                solve_ms=float(d['solve_ms'].mean()), iters_max=int(d['qp_iters'].max()))

def main(root):
    root = Path(root); out = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(sub.rglob("pnd_*.csv")):
            d = load(f)
            if d is not None and len(d['t']) > 10:
                out.append(summarize(f'{sub.name}/{f.stem}', d))
    if not out:
        print('No logs found under', root); return
    keys = list(out[0].keys())
    print(' | '.join(keys))
    for r in out:
        print(' | '.join(f'{r[k]:.2f}' if isinstance(r[k], float) else str(r[k]) for k in keys))
    with open(root.parent / 'summary.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(out)
    print('Saved', root.parent / 'summary.csv')

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / 'logs')
