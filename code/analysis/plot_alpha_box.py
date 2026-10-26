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
 "font.size":9,"axes.labelsize":9,"xtick.labelsize":8,"ytick.labelsize":8,"axes.linewidth":0.6,"xtick.direction":"in","ytick.direction":"in",
 "axes.grid":True,"grid.linewidth":0.3,"grid.alpha":0.5,"pdf.fonttype":42})
P=[A1,A02]
D=[pd.read_csv(p) for p in P]; cols=["#1f4e9c","#c0392b"]
fig,ax=plt.subplots(1,2,figsize=(3.5,1.9))
for a,k,lab in zip(ax,["e_pos_mm","e_rot_deg"],["Position error (mm)","Orientation error ($^\\circ$)"]):
    b=a.boxplot([d[k].values for d in D],whis=(5,95),showfliers=False,widths=0.5,patch_artist=True,
        medianprops=dict(color="k",lw=1),whiskerprops=dict(lw=0.7),capprops=dict(lw=0.7),boxprops=dict(lw=0.7))
    for patch,c in zip(b["boxes"],cols): patch.set_facecolor(c); patch.set_alpha(0.35); patch.set_edgecolor(c)
    for i,d in enumerate(D): a.plot(i+1,d[k].mean(),marker="^",color=cols[i],ms=4,ls="")
    a.set_xticks([1,2]); a.set_xticklabels([r"$\alpha=1$",r"$\alpha=0.2$"]); a.set_ylabel(lab); a.set_ylim(bottom=0); a.grid(axis="x",visible=False)
    for i,d in enumerate(D): print(k,[round(np.percentile(d[k],q),2) for q in (5,25,50,75,95)],round(d[k].mean(),2))
for a,s in zip(ax,"ab"): a.set_title(f"({s})",y=-0.38,fontsize=9)
fig.tight_layout(pad=0.3,w_pad=1.2)
fig.savefig(OUT/"alpha_box.pdf",bbox_inches="tight",pad_inches=0.02); fig.savefig(OUT/"alpha_box.png",dpi=200,bbox_inches="tight",pad_inches=0.02)
