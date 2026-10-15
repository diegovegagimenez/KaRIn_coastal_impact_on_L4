#!/usr/bin/env python3
"""Fig. 6 (v3): median PI_RMSD vs extraction radius for near/far-swath stations,
CalVal, fixed station panel, with 95% bootstrap CI of the median (10 000 resamples
over stations). Top axis: median number of grid cells per station.
Input: per_station_radius_with_pixels.csv (verify_v3_numbers.py)."""
import os
from pathlib import Path
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "figures"
RADII = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
NB = 10000
rng = np.random.default_rng(42)

D = pd.read_csv(HERE / "per_station_radius_with_pixels.csv")
D = D[D.phase == "CalVal"]
panel = set.intersection(*[set(D[D.radius == r].station) for r in RADII])
D = D[D.station.isin(panel)]
P = pd.DataFrame(json.load(open(HERE / "final_v1_scale.json"))["nearfar"])
P = P[P.phase == "CalVal"].set_index("radius")


def boot(x):
    x = np.asarray(x)
    idx = rng.integers(0, len(x), (NB, len(x)))
    m = np.median(x[idx], axis=1)
    return np.median(x), *np.percentile(m, [2.5, 97.5])


rows = []
for r in RADII:
    d = D[D.radius == r]
    a, b = d[d.near].pi.values, d[~d.near].pi.values
    # CI of the difference of medians (independent resampling of each group)
    ia = rng.integers(0, len(a), (NB, len(a)))
    ib = rng.integers(0, len(b), (NB, len(b)))
    diff = np.median(a[ia], axis=1) - np.median(b[ib], axis=1)
    rows.append(dict(radius=r, n_near=len(a), n_far=len(b), pix=d.ncells_my.median(),
                     **dict(zip(["near", "near_lo", "near_hi"], boot(a))),
                     **dict(zip(["far", "far_lo", "far_hi"], boot(b))),
                     diff=np.median(a) - np.median(b), diff_lo=np.percentile(diff, 2.5),
                     diff_hi=np.percentile(diff, 97.5), p=P.loc[r, "p"]))
S = pd.DataFrame(rows)
S.to_csv(HERE / "nearfar_bootstrap_ci.csv", index=False, float_format="%.3f")
print(S.round(2).to_string(index=False))

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                     "font.size": 11, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(7.5, 5))
for key, c, lab in [("near", "#1b9e77", "Near-swath"), ("far", "#d95f02", "Far-swath")]:
    n = int(S[f"n_{key}"].iloc[0])
    ax.fill_between(S.radius, S[f"{key}_lo"], S[f"{key}_hi"], color=c, alpha=0.2, lw=0)
    ax.plot(S.radius, S[key], color=c, lw=2, marker="o", label=f"{lab} ($n$={n})")
sig = S.p < 0.05
ax.scatter(S.radius[sig], S.near[sig], facecolor="#1b9e77", edgecolor="k", s=55, zorder=5)
ax.scatter(S.radius[~sig], S.near[~sig], facecolor="white", edgecolor="#1b9e77", s=55, zorder=5)
ax.axhline(0, color="0.6", lw=0.8, zorder=0)
ax.set_xlabel("Extraction radius $R$ [km]")
ax.set_ylabel("Median $\\mathrm{PI}_{\\mathrm{RMSD}}$ [%]")
ax.set_xticks(RADII)
top = ax.secondary_xaxis("top")
top.set_xticks(RADII)
top.set_xticklabels([int(round(x)) for x in S.pix], fontsize=9, color="0.35")
top.set_xlabel("Median number of grid cells in the disc", fontsize=10, color="0.35")
ax.legend(frameon=False, loc="upper right")
ax.grid(lw=0.4, alpha=0.4)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"figE_nearfar_bootstrap.{ext}", dpi=300, bbox_inches="tight")
print("OK")
