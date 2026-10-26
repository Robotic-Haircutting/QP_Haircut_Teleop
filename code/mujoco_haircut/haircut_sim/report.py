"""Haircut coverage report: recompute the clipper-tip trajectory from the teleoperation tracking log (Piper joint feedback q1-q6 in logs/*.csv)
and write an HTML file that works offline (coverage ratio + hair coverage maps from the back/left/right/top).

  python -m haircut_sim.report                  # latest log in the mujoco_haircut root
  python -m haircut_sim.report logs/xxx.csv     # a given log
Writes <same name>_haircut.html and opens it in the browser.
"""

import csv
import html
import sys
import webbrowser
from pathlib import Path

import mujoco
import numpy as np

from .coverage import REGION_NAMES, HairCoverage, tip_offset, tip_pos
from .scene import Scene, load_config

GREEN, RED, PATH, PATH_FAR = "#16a34a", "#dc2626", "#ea580c", "#94a3b8"


def default_log_dir() -> Path:
    """Haircut logs are stored by default in mujoco_haircut/results/<solver_date_time>/ (--robot.log_dir in run_haircut.bat)."""
    return Path(__file__).resolve().parent.parent / "results"


def latest_csv(log_dir: Path) -> Path:
    files = sorted(log_dir.rglob("*.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise SystemExit(f"No tracking logs in {log_dir}")
    return files[-1]


def load_joints(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} is empty")
    t = np.array([float(r["t"]) for r in rows])
    q = np.array([[float(r[f"q{i}"]) for i in range(1, 7)] for r in rows])
    return t, q


def compute(t: np.ndarray, q: np.ndarray, cfg: dict) -> dict:
    scene = Scene(cfg)
    m, d = scene.model, scene.data
    scene.forward()
    cov = HairCoverage(m, d, cfg)
    off = tip_offset(cfg)
    tips, dists, ratio = np.zeros((len(q), 3)), np.zeros(len(q)), np.zeros(len(q))
    for i, qi in enumerate(q):
        scene.set_piper(qi)
        mujoco.mj_kinematics(m, d)
        tips[i] = tip_pos(m, d, off)
        new, dists[i] = cov.update(tips[i])
        ratio[i] = cov.ratio if len(new) or i == 0 else ratio[i - 1]
    near = dists < cov.radius
    dt = np.diff(t, append=t[-1])
    return {"cov": cov, "tips": tips, "dists": dists, "near": near, "t": t, "ratio": ratio,
            "path_m": float(np.sum(np.linalg.norm(np.diff(tips, axis=0), axis=1))),
            "contact_s": float(dt[near].sum()), "duration_s": float(t[-1] - t[0])}


def _view_svg(title: str, cov: HairCoverage, tips: np.ndarray, near: np.ndarray, view: np.ndarray, up: np.ndarray, size=360) -> str:
    right = np.cross(view, up)
    c = cov.center + np.array([0, 0, (cov.all_verts[:, 2].max() - cov.center[2]) / 2])
    pts, covered = cov.hair_verts, cov.vert_covered
    facing = (pts - c) @ view < 0  # draw only the half facing the viewer
    order = np.argsort(-((pts - c) @ view))  # far points first
    order = order[facing[order]]
    if len(order) > 5000:
        order = order[:: len(order) // 5000 + 1]
    allp = cov.all_verts - c
    span = max(np.ptp(allp @ right), np.ptp(allp @ up)) * 1.1
    X = lambda p: size / 2 + (p - c) @ right / span * size  # noqa: E731
    Y = lambda p: size / 2 - (p - c) @ up / span * size  # noqa: E731
    out = [f'<svg viewBox="0 0 {size} {size + 24}" width="{size}" height="{size + 24}"><text x="8" y="16" class="ttl">{html.escape(title)}</text><g transform="translate(0,24)">']
    sil = cov.all_verts[:: max(1, len(cov.all_verts) // 4000)]
    out += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.6" fill="#e7d3c2"/>' for x, y in zip(X(sil), Y(sil), strict=True)]
    xs, ys = X(pts[order]), Y(pts[order])
    out += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.5" fill="{GREEN if k else RED}"/>' for x, y, k in zip(xs, ys, covered[order], strict=True)]
    # trajectory segments near the head (within 3x the coverage radius)
    close = np.flatnonzero(np.linalg.norm(tips - c, axis=1) < span)
    if len(close) > 1:
        step = max(1, len(close) // 3000)
        idx = close[::step]
        tx, ty = X(tips[idx]), Y(tips[idx])
        for a in range(len(idx) - 1):
            if idx[a + 1] - idx[a] <= 3 * step:
                col = PATH if near[idx[a]] else PATH_FAR
                out.append(f'<line x1="{tx[a]:.1f}" y1="{ty[a]:.1f}" x2="{tx[a + 1]:.1f}" y2="{ty[a + 1]:.1f}" stroke="{col}" stroke-width="1" opacity="0.8"/>')
    out.append("</g></svg>")
    return "\n".join(out)


def render(path: Path, r: dict) -> Path:
    cov = r["cov"]
    z = np.array([0.0, 0.0, 1.0])
    f = cov.face_dir
    left = np.cross(z, f)
    views = [("Back (seen from behind)", f, z), ("Left side", -left, z), ("Right side", left, z), ("Top (face up)", -z, f)]
    svgs = "\n".join(_view_svg(t, cov, r["tips"], r["near"], v, u) for t, v, u in views)
    stats = (f"Hair coverage <b>{cov.ratio * 100:.1f}%</b> (hair within {cov.radius * 100:.1f} cm of the clipper tip counts as cut)"
             f"<br>Coverage by region: " + " · ".join(f"{REGION_NAMES[k]} {v * 100:.1f}%" for k, v in cov.region_ratios().items()) +
             f"<br>Log duration {r['duration_s']:.1f} s · time close to the hair {r['contact_s']:.1f} s · total clipper-tip path {r['path_m']:.2f} m")
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>Haircut coverage {html.escape(path.stem)}</title>
<style>body{{font-family:system-ui,"Microsoft YaHei",sans-serif;margin:16px;color:#111;background:#fff}}
.ttl{{font-size:13px;font-weight:600}} .grid{{display:flex;flex-wrap:wrap;gap:12px}} svg{{border:1px solid #e5e7eb;border-radius:6px;background:#fafafa}}
.lg span{{display:inline-block;margin-right:14px}} .dot{{display:inline-block;width:10px;height:10px;border-radius:5px;margin-right:4px;vertical-align:middle}}</style></head>
<body><h2>Haircut coverage report: {html.escape(path.name)}</h2><p>{stats}</p>
<p class="lg"><span><i class="dot" style="background:{GREEN}"></i>Hair cut</span><span><i class="dot" style="background:{RED}"></i>Hair not cut</span>
<span><i class="dot" style="background:{PATH}"></i>Clipper-tip path (near the hair)</span><span><i class="dot" style="background:{PATH_FAR}"></i>Clipper-tip path (away from the hair)</span><span><i class="dot" style="background:#e7d3c2"></i>Face and neck (not hair)</span></p>
<div class="grid">{svgs}</div>
<p style="color:#666;font-size:12px">The hair region, clipper-tip offset and coverage radius are set under hair / tool in mujoco_haircut/scene_config.json.</p></body></html>"""
    out = path.with_name(path.stem + "_haircut.html")
    out.write_text(doc, encoding="utf-8")
    return out


def make_report(csv_path: Path, cfg_path=None, open_browser=True) -> Path:
    t, q = load_joints(csv_path)
    r = compute(t, q, load_config(cfg_path))
    out = render(csv_path, r)
    print(f"Hair coverage {r['cov'].ratio * 100:.1f}%, report: {out}")
    if open_browser:
        webbrowser.open(out.resolve().as_uri())
    return out


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = Path(args[0]) if args else latest_csv(default_log_dir())
    make_report(path, open_browser="--no-open" not in sys.argv)


if __name__ == "__main__":
    main()
