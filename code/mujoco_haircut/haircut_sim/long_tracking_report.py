"""Plot rd (leader target) against r (actual Piper end effector) and write an HTML file that works offline (no matplotlib / network needed).

Usage:
  python tools/plot_tracking.py                 # plot the latest log in logs/
  python tools/plot_tracking.py logs/qp_xxx.csv # plot a given file
Writes a .html with the same name and opens it in the browser; if one exists it saves as _2.html, _3.html… instead of overwriting.
"""

import csv
import html
import sys
import webbrowser
from pathlib import Path

import numpy as np

from .control.transforms import matrix_to_rotvec, rotvec_to_matrix

ROOT = Path(__file__).resolve().parent.parent
COLORS = {"rd": "#2563eb", "r": "#dc2626", "rc": "#16a34a"}
LABELS = {"rd": "rd leader target", "r": "r Piper actual (FK of joint feedback)", "rc": "rc solver command (FK of commanded joints)"}


def load(path: Path) -> dict[str, np.ndarray]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} is empty")
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def svg_chart(t, series: list[tuple[str, np.ndarray, str]], title: str, unit: str, w=900, h=220) -> str:
    pad_l, pad_r, pad_t, pad_b = 60, 10, 28, 28
    ys = np.concatenate([s for _, s, _ in series])
    y0, y1 = float(np.min(ys)), float(np.max(ys))
    if y1 - y0 < 1e-9:
        y0, y1 = y0 - 1, y1 + 1
    m = 0.05 * (y1 - y0)
    y0, y1 = y0 - m, y1 + m
    t0, t1 = float(t[0]), float(max(t[-1], t[0] + 1e-6))
    X = lambda v: pad_l + (v - t0) / (t1 - t0) * (w - pad_l - pad_r)  # noqa: E731
    Y = lambda v: pad_t + (y1 - v) / (y1 - y0) * (h - pad_t - pad_b)  # noqa: E731
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px">']
    out.append(f'<text x="{pad_l}" y="18" class="ttl">{html.escape(title)} ({unit})</text>')
    for k in range(5):
        v = y0 + (y1 - y0) * k / 4
        out.append(f'<line x1="{pad_l}" x2="{w - pad_r}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/>')
        out.append(f'<text x="{pad_l - 6}" y="{Y(v) + 4:.1f}" class="ax" text-anchor="end">{v:.3g}</text>')
    for k in range(6):
        v = t0 + (t1 - t0) * k / 5
        out.append(f'<text x="{X(v):.1f}" y="{h - 8}" class="ax" text-anchor="middle">{v:.1f}s</text>')
    step = max(1, len(t) // 1500)
    for _, s, color in series:
        pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(t[::step], s[::step], strict=True))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.4"/>')
    out.append("</svg>")
    return "\n".join(out)


def _limit_rows(d, joint_limits=None) -> str:
    """Split the error by whether a joint is at its limit (at a limit = the solver cannot reach the target)."""
    from .tracking_report import limit_breakdown

    b = limit_breakdown(d, joint_limits=joint_limits)
    if b is None:
        return ""
    per = ", ".join(f"J{i + 1} {v * 100:.0f}%" for i, v in enumerate(b["joint_frac"]) if v > 0.005) or "none"
    return (f"<tr><th>Share of time with a joint at its limit</th><td>{b['sat_frac'] * 100:.0f}% ({per})</td></tr>"
            f"<tr><th>Position / orientation error, no joint at limit</th><td>mean {b['pos_free']:.2f} mm / {b['rot_free']:.2f}°</td></tr>"
            f"<tr><th>Position / orientation error, joint at limit</th><td>mean {b['pos_sat']:.2f} mm / {b['rot_sat']:.2f}°</td></tr>")


def make_long_report(csv_path, open_browser=False, joint_limits=None):
    path = Path(csv_path)
    d = load(path)
    t = d["t"]

    def stat(k):
        v = d[k]
        return f"mean {v.mean():.2f}, 95% {np.percentile(v, 95):.2f}, max {v.max():.2f}"

    parts = [
        f"<h1>rd / r tracking comparison</h1><p class='sub'>{html.escape(str(path))} · {len(t)} frames · {t[-1]:.1f} s · "
        f"mean {len(t) / max(t[-1], 1e-6):.1f} Hz</p>",
        "<table>"
        f"<tr><th>Position error of actual r, mm</th><td>{stat('e_pos_mm')}</td></tr>"
        f"<tr><th>Orientation error of actual r, °</th><td>{stat('e_rot_deg')}</td></tr>"
        f"<tr><th>Position error of solver command rc, mm</th><td>{stat('ec_pos_mm')}</td></tr>"
        f"<tr><th>Orientation error of solver command rc, °</th><td>{stat('ec_rot_deg')}</td></tr>"
        f"<tr><th>Solve time, ms</th><td>{stat('solve_ms')}</td></tr>"
        + _limit_rows(d, joint_limits) +
        "</table>",
        "<p class='legend'>" + " ".join(f"<span style='color:{COLORS[k]}'>━ {LABELS[k]}</span>" for k in COLORS) + "</p>",
        "<p class='note'>rc − rd = the solver itself not keeping up (constraints, speed limits, singularities); r − rc = Piper motor execution lag.</p>",
    ]
    for k, name, unit, scale in [("x", "End effector X", "mm", 1000), ("y", "End effector Y", "mm", 1000), ("z", "End effector Z", "mm", 1000)]:
        series = [(p, d[f"{p}_{k}"] * scale, COLORS[p]) for p in ("rd", "rc", "r")]
        parts.append(svg_chart(t, series, name, unit))
    # Orientation is not plotted as a raw rotation vector: when |rotation vector| is near 180° it flips between ±π, so the curves look far apart although the orientations are almost the same.
    # Instead plot "rotation relative to the first rd orientation" (rotation vector log(R·R_ref^T) in the base frame): small, continuous and comparable.
    R_ref = rotvec_to_matrix([d["rd_wx"][0], d["rd_wy"][0], d["rd_wz"][0]])
    rel = {}
    for p in ("rd", "rc", "r"):
        W = np.stack([d[f"{p}_wx"], d[f"{p}_wy"], d[f"{p}_wz"]], axis=1)
        rel[p] = np.rad2deg(np.array([matrix_to_rotvec(rotvec_to_matrix(w) @ R_ref.T) for w in W]))
    for i, ax in enumerate("xyz"):
        series = [(p, rel[p][:, i], COLORS[p]) for p in ("rd", "rc", "r")]
        parts.append(svg_chart(t, series, f"Orientation: rotation about base {ax} relative to the start pose (rotation-vector component)", "°"))
    parts.append(svg_chart(t, [("r", d["e_pos_mm"], COLORS["r"]), ("rc", d["ec_pos_mm"], COLORS["rc"])], "Position error |r − rd|", "mm"))
    parts.append(svg_chart(t, [("r", d["e_rot_deg"], COLORS["r"]), ("rc", d["ec_rot_deg"], COLORS["rc"])], "Orientation error", "°"))
    parts.append(svg_chart(t, [("", d["solve_ms"], "#7c3aed")], "Solve time", "ms"))
    for i in range(1, 7):
        parts.append(svg_chart(t, [("c", d[f"qc{i}"], COLORS["rc"]), ("m", d[f"q{i}"], COLORS["r"])], f"Joint J{i} command vs. actual", "°", h=160))

    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>rd r tracking comparison</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;color:#111}}h1{{font-size:20px;margin:0}}
.sub{{color:#555;margin:4px 0 12px}}table{{border-collapse:collapse;margin:8px 0}}th,td{{border:1px solid #ddd;padding:4px 10px;text-align:left;font-size:14px}}
.legend span{{margin-right:18px;font-size:14px}}.note{{color:#555;font-size:13px}}
.ttl{{font-size:13px;font-weight:600}}.ax{{font-size:10px;fill:#666}}.grid{{stroke:#eee}}svg{{display:block;margin:6px 0}}</style>
</head><body>{"".join(parts)}</body></html>"""
    # Do not overwrite existing plots: add a number if an HTML file with the same name exists
    out = path.with_suffix(".html")
    k = 2
    while out.exists():
        out = path.with_name(f"{path.stem}_{k}.html")
        k += 1
    out.write_text(page, encoding="utf-8")
    print(f"Created {out}")
    if open_browser:
        webbrowser.open(out.resolve().as_uri())
    return out
