#!/usr/bin/env python3
"""Cifras del manuscrito v3 con la regionalizacion v2 (7 regiones: se anade la
'Transition zone' = IHO Kattegat [incl. Sound y Belts] + IHO Baltic Sea al oeste
de 13E; ver regions_v2.py). Todo a partir de per_station_radius_with_pixels.csv.

Imprime: Tabla 2 (regional, 50 km, ambas fases, CI bootstrap), near/far a 50 km
por region (Fig. 4 / Tabla S1), barrido near/far por region en panel fijo
(Tabla 4 / S3), medianas regionales por radio (texto 3.4.2), test pareado de
escala, y estadisticos de referencia de los grupos near.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu as mw, wilcoxon, spearmanr

HERE = Path(__file__).resolve().parent
pd.set_option("display.width", 250)
rng = np.random.default_rng(0)
NB = 10000
R = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
ORDER = ["Transition zone", "North West Shelf", "Baltic Sea", "Iberia-Biscay-Ireland",
         "Mediterranean Sea", "Arctic", "Black Sea"]

S = pd.read_csv(HERE / "station_regions_v2.csv").set_index("station")
d = pd.read_csv(HERE / "per_station_radius_with_pixels.csv")
d["reg"] = d.station.map(S.region_v2)
d.to_csv(HERE / "per_station_radius_with_pixels_v2.csv", index=False)


def ci(x):
    x = np.asarray(x)
    m = x[rng.integers(0, len(x), (NB, len(x)))].mean(axis=1)
    return x.mean(), *np.percentile(m, [2.5, 97.5])


print("=" * 30, "TABLE 2 (R=50, all stations)")
rows = []
for ph in ("CalVal", "Science"):
    x = d[(d.radius == 50) & (d.phase == ph)]
    for reg in ["All"] + ORDER:
        g = x if reg == "All" else x[x.reg == reg]
        m, lo, hi = ci(g.dr.values)
        rows.append(dict(phase=ph, region=reg, n=len(g), npix=g.ncells_my.median(),
                         rmsd_my=g.rmsd_my.median(), rmsd_mk=g.rmsd_miost.median(),
                         dr=m, lo=lo, hi=hi, sig="*" if lo > 0 or hi < 0 else "",
                         pct=100 * (g.dr > 0).mean(), medPI=g.pi.median(), meanPI=g.pi.mean()))
T2 = pd.DataFrame(rows)
print(T2.round(3).to_string(index=False))
T2.to_csv(HERE / "v3_table2_regions.csv", index=False)

print("=" * 30, "NEAR/FAR at R=50, CalVal, all stations (Fig. 4 / Table S1)")
c = d[(d.radius == 50) & (d.phase == "CalVal")]
rows = []
for reg in ["All"] + ORDER:
    g = c if reg == "All" else c[c.reg == reg]
    a, b = g[g.near], g[~g.near]
    p = mw(a.pi, b.pi).pvalue if len(a) > 2 and len(b) > 2 else np.nan
    rows.append(dict(region=reg, n_near=len(a), near=a.pi.median(), n_far=len(b), far=b.pi.median(),
                     diff=a.pi.median() - b.pi.median(), p=p, pix_near=a.ncells_my.median(),
                     pix_far=b.ncells_my.median()))
NF = pd.DataFrame(rows)
print(NF.round(3).to_string(index=False))
NF.to_csv(HERE / "v3_nearfar_R50_regions.csv", index=False)
a, b = c[c.near], c[~c.near]
for col in ("tg_std", "rmsd_my", "corr_my"):
    print(f"baseline {col}: near {a[col].median():.3f} far {b[col].median():.3f} p={mw(a[col], b[col]).pvalue:.2f}")
print("share of near in TZ: %.0f%%; share of far in TZ: %.0f%%" % (100 * (a.reg == "Transition zone").mean(),
                                                                100 * (b.reg == "Transition zone").mean()))

print("=" * 30, "PANEL near/far by region x radius (Table 4 / S3)")
cv = d[d.phase == "CalVal"]
panel = set.intersection(*[set(cv[cv.radius == r].station) for r in R])
p = cv[cv.station.isin(panel)]
g = p.groupby(["reg", "near", "radius"])
print("n:"); print(p[p.radius == 50].groupby(["reg", "near"]).size().to_string())
print(g.pi.median().unstack().round(2).to_string())
print("Npix panel:", p.groupby("radius").ncells_my.median().tolist())
print("near NOT in panel, per radius (all available):")
q = cv[cv.near & cv.reg.isin(["Iberia-Biscay-Ireland", "Arctic"])]
print(q.groupby(["reg", "radius"]).pi.agg(["size", "median"]).round(2).unstack(0).to_string())
g.pi.median().unstack().round(2).to_csv(HERE / "v3_table4_panel.csv")

print("=" * 30, "Reference skill of near groups (panel, R=50)")
x = p[(p.radius == 50) & p.near]
gg = x.groupby("reg")
print(pd.DataFrame(dict(n=gg.size(), rmsd_my=gg.rmsd_my.median(), rmsd_mk=gg.rmsd_miost.median(),
                        corr_my=gg.corr_my.median(), tg_std=gg.tg_std.median(), pi=gg.pi.median())).round(2))
print("Spearman(rmsd_my, PI) over near panel: %.2f (p=%.3f)" % spearmanr(x.rmsd_my, x.pi))
y = x[x.reg != "Transition zone"]
print("near outside TZ: n=%d medPI %.2f" % (len(y), y.pi.median()))

print("=" * 30, "REGIONAL medians by radius, all available stations (text 3.4.2)")
for ph in ("CalVal", "Science"):
    x = d[d.phase == ph]
    print(ph); print(x.groupby(["reg", "radius"]).pi.median().unstack().round(2).to_string())
    print(x.groupby(["reg", "radius"]).size().unstack().agg(["min", "max"], axis=1).T.to_string())

print("=" * 30, "PAIRED scale tests (panel, CalVal)")
P = p.pivot(index="station", columns="radius", values="pi")
meta = p.drop_duplicates("station").set_index("station")
grp = {"near all": meta.near, "near TZ": meta.near & (meta.reg == "Transition zone"),
       "near other": meta.near & (meta.reg != "Transition zone"), "far": ~meta.near}
for lab, m in grp.items():
    for a_, b_ in ((10, 50), (10, 100), (50, 100)):
        df = (P[a_] - P[b_])[m]
        print("%-11s PI(%3d)-PI(%3d) n=%3d median %+6.2f  >0: %3.0f%%  p=%.1e" %
              (lab, a_, b_, len(df), df.median(), 100 * (df > 0).mean(), wilcoxon(df).pvalue))
