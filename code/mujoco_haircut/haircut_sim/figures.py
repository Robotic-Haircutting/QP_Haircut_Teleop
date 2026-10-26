"""Paper figures: PDF (Type 42 fonts embedded) + 600 dpi PNG, IEEE single column 3.5 in / double column 7.16 in.

  python -m haircut_sim.figures                 # latest log in the mujoco_haircut root
  python -m haircut_sim.figures xxx.csv         # a given log
Output goes to <log name>_figures/:
  tracking_xyz      end-effector x/y/z: leader target r_d vs. actual Piper r
  tracking_error    position error (mm) / orientation error (deg)
  coverage_views    hair coverage: back / left / right / top views
  coverage_ratio    coverage over time
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .report import compute, default_log_dir, latest_csv, load_joints  # noqa: E402
from .scene import load_config  # noqa: E402
from .control.transforms import matrix_to_rotvec, rotvec_to_matrix

COL_W = 3.5  # IEEE single-column width (in)
DBL_W = 7.16  # double-column width
C_TARGET, C_ACTUAL = "#1f4e9c", "#c0392b"
C_COVERED, C_UNCOVERED, C_SKIN = "#2e8b57", "#d64541", "#e9d8c8"
C_PATH = "#e67e22"


def _style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.6, "lines.linewidth": 0.9,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.direction": "in", "ytick.direction": "in",
        "axes.grid": True, "grid.linewidth": 0.3, "grid.alpha": 0.5,
        "legend.frameon": False, "savefig.bbox": None,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _save(fig, out_dir: Path, name: str) -> None:
    fig.savefig(out_dir / f"{name}.pdf", dpi=600)
    fig.savefig(out_dir / f"{name}.png", dpi=600)
    plt.close(fig)


def _load_log(path: Path) -> dict[str, np.ndarray]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def tracking_figures(log: dict, out_dir: Path) -> None:
    t = log["t"] - log["t"][0]
    fig, axes = plt.subplots(3, 1, figsize=(COL_W, 3.6), sharex=True, layout="constrained")
    for ax, k in zip(axes, "xyz", strict=True):
        ax.plot(t, log[f"rd_{k}"] * 1000, color=C_TARGET, label=r"Target $r_d$")
        ax.plot(t, log[f"rc_{k}"] * 1000, color="#2e8b57", linestyle=":", label=r"Command $r_c$")
        ax.plot(t, log[f"r_{k}"] * 1000, color=C_ACTUAL, linestyle="--", label=r"Actual $r$")
        ax.set_ylabel(f"${k}$ (mm)")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.36), ncol=3, columnspacing=0.7, handlelength=1.2)
    axes[-1].set_xlabel("Time (s)")
    axes[-1].set_xlim(t[0], t[-1])
    fig.align_ylabels(axes)
    _save(fig, out_dir, "tracking_xyz")

    fig, axes = plt.subplots(2, 1, figsize=(COL_W, 2.8), sharex=True, layout="constrained")
    axes[0].plot(t, log["e_pos_mm"], color=C_ACTUAL)
    axes[0].set_ylabel("Position error (mm)")
    axes[1].plot(t, log["e_rot_deg"], color=C_TARGET)
    axes[1].set_ylabel("Orientation error (deg)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_xlim(t[0], t[-1])
    for ax, key, unit in ((axes[0], "e_pos_mm", "mm"), (axes[1], "e_rot_deg", "deg")):
        v = log[key]
        ax.text(0.99, 0.95, f"mean {v.mean():.2f} {unit}, RMS {np.sqrt(np.mean(v**2)):.2f} {unit}", transform=ax.transAxes,
                ha="right", va="top", fontsize=7)
    fig.align_ylabels(axes)
    _save(fig, out_dir, "tracking_error")

    # Relative rotation avoids wrapping the absolute rotation-vector components.
    R_ref = rotvec_to_matrix([log[f"rd_w{k}"][0] for k in "xyz"])
    relative = {}
    for prefix in ("rd", "rc", "r"):
        vectors = np.stack([log[f"{prefix}_w{k}"] for k in "xyz"], axis=1)
        relative[prefix] = np.rad2deg(np.array([
            matrix_to_rotvec(rotvec_to_matrix(v) @ R_ref.T) for v in vectors]))
    fig, axes = plt.subplots(3, 1, figsize=(COL_W, 3.6), sharex=True, layout="constrained")
    for i, ax in enumerate(axes):
        for prefix, color, style, label in (("rd", C_TARGET, "-", "Target"),
                ("rc", "#2e8b57", ":", "Command"), ("r", C_ACTUAL, "--", "Actual")):
            ax.plot(t, relative[prefix][:, i], color=color, linestyle=style, label=label)
        ax.set_ylabel(rf"$\phi_{'xyz'[i]}$ (deg)")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.36), ncol=3)
    axes[-1].set_xlabel("Time (s)")
    axes[-1].set_xlim(t[0], t[-1])
    _save(fig, out_dir, "tracking_orientation")

    fig, axes = plt.subplots(2, 1, figsize=(COL_W, 2.6), sharex=True, layout="constrained")
    axes[0].plot(t, log["solve_ms"], color="#7c3aed")
    axes[0].set_ylabel("Solve time (ms)")
    axes[1].plot(t, log["qp_iters"], color=C_TARGET)
    axes[1].set_ylabel("Iterations")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_xlim(t[0], t[-1])
    _save(fig, out_dir, "solver_performance")

    for first, last in ((1, 3), (4, 6)):
        fig, axes = plt.subplots(3, 1, figsize=(COL_W, 3.6), sharex=True, layout="constrained")
        for ax, j in zip(axes, range(first, last+1), strict=True):
            ax.plot(t, log[f"qc{j}"], color="#2e8b57", linestyle=":", label="Command")
            ax.plot(t, log[f"q{j}"], color=C_ACTUAL, linestyle="--", label="Actual")
            ax.set_ylabel(rf"$q_{j}$ (deg)")
        axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.36), ncol=2)
        axes[-1].set_xlabel("Time (s)")
        axes[-1].set_xlim(t[0], t[-1])
        _save(fig, out_dir, f"joint_tracking_{first}_{last}")


def _view(ax, r: dict, view: np.ndarray, up: np.ndarray, title: str) -> None:
    cov = r["cov"]
    right = np.cross(view, up)
    c = cov.center + np.array([0, 0, (cov.all_verts[:, 2].max() - cov.center[2]) / 2])
    allv = cov.all_verts
    near_all = (allv - c) @ view < 0
    sil = allv[near_all][:: max(1, near_all.sum() // 6000)]
    ax.scatter((sil - c) @ right * 100, (sil - c) @ up * 100, s=0.4, c=C_SKIN, linewidths=0, rasterized=True)
    pts, covered = cov.hair_verts, cov.vert_covered
    facing = (pts - c) @ view < 0
    order = np.argsort(-((pts - c) @ view))
    order = order[facing[order]]
    col = np.where(covered[order], C_COVERED, C_UNCOVERED)
    ax.scatter((pts[order] - c) @ right * 100, (pts[order] - c) @ up * 100, s=0.6, c=col, linewidths=0, rasterized=True)
    tips, near = r["tips"], r["near"]
    seg = np.where(near, (tips - c) @ right * 100, np.nan), np.where(near, (tips - c) @ up * 100, np.nan)
    ax.plot(*seg, color=C_PATH, linewidth=0.5, alpha=0.9)
    ax.set_title(title, pad=2)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def coverage_figures(r: dict, out_dir: Path) -> None:
    cov = r["cov"]
    z = np.array([0.0, 0.0, 1.0])
    f = cov.face_dir
    left = np.cross(z, f)
    views = [(f, z, "(a) Back"), (-left, z, "(b) Left"), (left, z, "(c) Right"), (-z, f, "(d) Top")]
    fig, axes = plt.subplots(1, 4, figsize=(DBL_W, 2.6))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.2, top=0.86, wspace=0.06)
    for ax, (v, u, title) in zip(axes, views, strict=True):
        _view(ax, r, v, u, title)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=4, color=C_COVERED, label="Covered hair"),
               plt.Line2D([], [], marker="o", linestyle="", markersize=4, color=C_UNCOVERED, label="Uncovered hair"),
               plt.Line2D([], [], color=C_PATH, label="Tool-tip path (near hair)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.01))
    reg = cov.region_ratios()
    fig.suptitle(f"Hair coverage {cov.ratio * 100:.1f}% (top {reg['top'] * 100:.0f}%, back {reg['back'] * 100:.0f}%, left {reg['left'] * 100:.0f}%, "
                 f"right {reg['right'] * 100:.0f}%; proximity radius {cov.radius * 1000:.0f} mm)", y=0.97, fontsize=9)
    _save(fig, out_dir, "coverage_views")

    t = r["t"] - r["t"][0]
    fig, ax = plt.subplots(figsize=(COL_W, 1.9), layout="constrained")
    ax.plot(t, r["ratio"] * 100, color=C_COVERED)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Hair coverage (%)")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, max(5.0, r["ratio"].max() * 110))
    _save(fig, out_dir, "coverage_ratio")


def make_figures(csv_path: Path, cfg_path=None) -> Path:
    _style()
    out_dir = csv_path.with_name(csv_path.stem + "_figures")
    out_dir.mkdir(exist_ok=True)
    log = _load_log(csv_path)
    if "rd_x" in log:
        tracking_figures(log, out_dir)
    t, q = load_joints(csv_path)
    coverage_figures(compute(t, q, load_config(cfg_path)), out_dir)
    print(f"Paper figures (PDF + 600 dpi PNG) saved to {out_dir}")
    return out_dir


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    make_figures(Path(args[0]) if args else latest_csv(default_log_dir()))


if __name__ == "__main__":
    main()
