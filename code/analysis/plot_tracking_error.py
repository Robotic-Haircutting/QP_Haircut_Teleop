import sys
from pathlib import Path
# Usage: python <script> <pnd_*.csv for alpha=1> <pnd_*.csv for alpha=0.2> [output_dir]
if len(sys.argv) < 3: sys.exit(__doc__ or "usage: script.py A1_CSV A02_CSV [OUT_DIR]")
A1, A02 = sys.argv[1], sys.argv[2]
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(".")
OUT.mkdir(parents=True, exist_ok=True)
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.serif":["TeX Gyre Termes","Nimbus Roman"],"mathtext.fontset":"stix",
 "font.size":9,"axes.labelsize":9,"axes.titlesize":9,"legend.fontsize":8,"xtick.labelsize":8,"ytick.labelsize":8,
 "axes.linewidth":0.6,"lines.linewidth":0.9,"xtick.major.width":0.6,"ytick.major.width":0.6,"xtick.direction":"in","ytick.direction":"in",
 "axes.grid":True,"grid.linewidth":0.3,"grid.alpha":0.5,"legend.frameon":False,"pdf.fonttype":42})
for n,p in [("a1",A1),("a02",A02)]:
    d=pd.read_csv(p); t=d.t-d.t.iloc[0]
    fig,axes=plt.subplots(2,1,figsize=(3.5,2.8),sharex=True,layout="constrained")
    axes[0].plot(t,d.e_pos_mm,color="#c0392b"); axes[0].set_ylabel("Position error (mm)")
    axes[1].plot(t,d.e_rot_deg,color="#1f4e9c"); axes[1].set_ylabel("Orientation error (deg)")
    axes[1].set_xlabel("Time (s)"); axes[1].set_xlim(t.iloc[0],t.iloc[-1]); fig.align_ylabels(axes)
    fig.savefig(OUT/f"{n}_tracking_error.pdf",dpi=600)
    fig.savefig(OUT/f"{n}_tracking_error.png",dpi=150)
