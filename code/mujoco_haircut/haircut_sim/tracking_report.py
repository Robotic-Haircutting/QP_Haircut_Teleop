"""Compact state report linking the independently exportable IEEE figures."""
from pathlib import Path
import html
from .figures import _load_log, tracking_figures, _style

STATE_FIGURES = [
    ("tracking_xyz", "End-effector position"),
    ("tracking_orientation", "End-effector orientation"),
    ("tracking_error", "Tracking errors"),
    ("solver_performance", "Solver performance"),
    ("joint_tracking_1_3", "Joint tracking: J1-J3"),
    ("joint_tracking_4_6", "Joint tracking: J4-J6"),
]

def limit_breakdown(log, margin_deg=0.5, joint_limits=None):
    """Split the error by whether any joint is at its limit: at a limit the solver cannot reach the target, and most of the error comes from there.
    Limits default to PiperFollowerConfig.joint_limits_deg (the bat files do not change it); other projects pass their own joint_limits.
    Returns None when the CSV has no joint commands."""
    import numpy as np
    from .control.config_piper_follower import PiperFollowerConfig

    if "qc1" not in log:
        return None
    lim = np.asarray(joint_limits if joint_limits is not None else PiperFollowerConfig(id="report", mock=True).joint_limits_deg, dtype=float)
    qc = np.stack([log[f"qc{i}"] for i in range(1, 7)], axis=1)
    at = (qc <= lim[:, 0] + margin_deg) | (qc >= lim[:, 1] - margin_deg)
    sat = at.any(axis=1)
    free = ~sat
    e, r = log["e_pos_mm"], log["e_rot_deg"]
    f = lambda v, m: float(v[m].mean()) if m.any() else float("nan")  # noqa: E731
    return {"sat_frac": float(sat.mean()), "joint_frac": at.mean(axis=0),
            "pos_free": f(e, free), "rot_free": f(r, free), "pos_sat": f(e, sat), "rot_sat": f(r, sat)}


def limit_text(b) -> str:
    worst = int(b["joint_frac"].argmax())
    return (f"Joint-limit saturation: {b['sat_frac'] * 100:.0f}% of frames (mostly J{worst + 1}, {b['joint_frac'][worst] * 100:.0f}%). "
            f"Without saturation: {b['pos_free']:.2f} mm / {b['rot_free']:.2f} deg; during saturation: {b['pos_sat']:.2f} mm / {b['rot_sat']:.2f} deg.")


def make_tracking_report(csv_path, figures_ready=False, joint_limits=None):
    csv_path = Path(csv_path)
    folder = csv_path.with_name(csv_path.stem + "_figures")
    folder.mkdir(exist_ok=True)
    log = _load_log(csv_path)
    if not figures_ready:
        _style()
        tracking_figures(log, folder)
    cards = []
    for name, label in STATE_FIGURES:
        stem = f"{folder.name}/{name}"
        cards.append(f'<section><h2>{html.escape(label)}</h2><a href="{stem}.pdf">PDF</a> · <a href="{stem}.png">PNG</a><img src="{stem}.png" alt="{html.escape(label)}"></section>')
    stats = f"Duration: {log['t'][-1]-log['t'][0]:.1f} s; mean position error: {log['e_pos_mm'].mean():.2f} mm; mean orientation error: {log['e_rot_deg'].mean():.2f} deg; mean solve time: {log['solve_ms'].mean():.2f} ms."
    b = limit_breakdown(log, joint_limits=joint_limits)
    if b is not None:
        stats += " " + limit_text(b)
    doc = '<!doctype html><html><meta charset="utf-8"><title>Robot tracking results</title><style>body{font-family:system-ui;margin:24px;color:#222}h1{font-size:22px}h2{font-size:16px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:20px}section{border:1px solid #ddd;padding:12px}img{display:block;width:100%;max-width:620px;margin:auto}</style><h1>Robot tracking results</h1><p>'+html.escape(stats)+'</p><main>'+''.join(cards)+'</main></html>'
    out = csv_path.with_name(csv_path.stem + "_ieee.html")
    i = 2
    while out.exists():
        out = csv_path.with_name(f"{csv_path.stem}_ieee_{i}.html")
        i += 1
    out.write_text(doc, encoding="utf-8")
    print("State report:", out)
    return out
