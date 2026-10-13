#!/usr/bin/env python3
"""Boxplot PI_RMSD near/far por region (regiones v2), CalVal, todas las estaciones
disponibles al radio dado. Uso: python make_figF_proximity_boxplot_v2.py [R_km]
  R=50 -> figF_proximity_boxplot_v2 (Fig. 4);  R=10 -> figF_proximity_boxplot_v2_R10."""
import os
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
HERE = Path(__file__).resolve().parent; OUT = HERE.parent / "figures"
d = pd.read_csv(HERE / "per_station_radius_with_pixels_v2.csv")
RAD = int(sys.argv[1]) if len(sys.argv) > 1 else 50
d = d[(d.phase == "CalVal") & (d.radius == RAD)]
GROUPS = ["All stations", "Transition zone", "North West Shelf", "Baltic Sea", "Mediterranean Sea", "Iberia-Biscay-Ireland"]
C_NEAR, C_FAR = "#1b9e77", "#d95f02"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 10.5,
                     "xtick.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(11, 5.4))
positions, data, colors, xt, xl = [], [], [], [], []
pos = 0
for g in GROUPS:
    gg = d if g == "All stations" else d[d.reg == g]
    a, b = gg[gg.near].pi.dropna(), gg[~gg.near].pi.dropna()
    if len(a) < 3:      # sin estaciones near suficientes a este radio
        continue
    data += [a.values, b.values]; colors += [C_NEAR, C_FAR]; positions += [pos, pos + 0.8]
    xt.append(pos + 0.4); xl.append(f"{g}\n(n={len(a)} / {len(b)})"); pos += 2.2
bp = ax.boxplot(data, positions=positions, widths=0.65, patch_artist=True, showfliers=False, whis=1.5,
                medianprops=dict(color="black", lw=1.6), boxprops=dict(lw=0.8),
                whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c); patch.set_alpha(0.55)
for i, v in enumerate(data):
    m = np.median(v)
    ax.annotate(f"{m:+.1f}", (positions[i], m), textcoords="offset points", xytext=(0, 5),
                ha="center", fontsize=8.2, fontweight="bold")
ax.axhline(0, color="0.6", lw=0.8, zorder=0)
ax.set_xticks(xt); ax.set_xticklabels(xl)
ax.set_ylabel("$\\mathrm{PI}_{\\mathrm{RMSD}}$ [%]"); ax.set_ylim(*((-45, 70) if RAD == 50 else (-70, 85)))
ax.set_title(f"$R$ = {RAD} km", fontsize=11, loc="left")
ax.grid(axis="y", lw=0.4, alpha=0.4)
ax.legend(handles=[Patch(facecolor=C_NEAR, alpha=0.55, label="Near-swath"),
                   Patch(facecolor=C_FAR, alpha=0.55, label="Far-swath")], loc="upper right", frameon=False)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / (f"figF_proximity_boxplot_v2.{ext}" if RAD == 50 else f"figF_proximity_boxplot_v2_R{RAD}.{ext}"), dpi=300, bbox_inches="tight")
print("OK")
