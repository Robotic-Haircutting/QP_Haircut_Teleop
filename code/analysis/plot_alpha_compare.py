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
 "font.size":9,"axes.labelsize":9,"legend.fontsize":8,"xtick.labelsize":8,"ytick.labelsize":8,"axes.linewidth":0.6,
 "lines.linewidth":1.0,"xtick.direction":"in","ytick.direction":"in","axes.grid":True,"grid.linewidth":0.3,"grid.alpha":0.5,
 "legend.frameon":False,"pdf.fonttype":42})
P=[(r"$\alpha=1$",A1,"#1f4e9c","-"),
   (r"$\alpha=0.2$",A02,"#c0392b","--")]
fig,ax=plt.subplots(1,2,figsize=(3.5,1.9),layout="constrained")
for lab,p,c,ls in P:
    d=pd.read_csv(p)
    for a,k in zip(ax[:2],["e_pos_mm","e_rot_deg"]):
        v=np.sort(d[k].values); a.plot(v,np.arange(1,len(v)+1)/len(v)*100,color=c,ls=ls,label=lab)
    t=d.t.values-d.t.values[0]; tt=np.arange(0,t[-1],0.01)
    lags=np.arange(0,61)  # 0..600 ms
    cc=np.zeros(len(lags))
    for k in "xyz":
        a_=np.interp(tt,t,d[f"rd_{k}"]); b_=np.interp(tt,t,d[f"rc_{k}"])
        a_=a_-a_.mean(); b_=b_-b_.mean()
        cc+=np.array([np.dot(a_[:len(a_)-L],b_[L:])/np.sqrt(np.dot(a_[:len(a_)-L],a_[:len(a_)-L])*np.dot(b_[L:],b_[L:])) for L in lags])/3
    print(lab,lags[cc.argmax()]*10)
ax[0].set_xlabel("Position error (mm)"); ax[0].set_ylabel("Samples (%)"); ax[0].set_xlim(0,60); ax[0].set_xticks([0,20,40,60])
ax[1].set_xlabel("Orientation error ($^\\circ$)"); ax[1].set_xlim(0,40)
for a,s in zip(ax,"ab"): a.set_title(f"({s})",y=-0.55,fontsize=9)
ax[0].legend(loc="lower right")
fig.savefig(OUT/"alpha_compare.pdf",bbox_inches="tight",pad_inches=0.02); fig.savefig(OUT/"alpha_compare.png",dpi=200,bbox_inches="tight",pad_inches=0.02)
