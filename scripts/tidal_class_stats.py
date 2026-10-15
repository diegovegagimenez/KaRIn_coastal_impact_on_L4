#!/usr/bin/env python3
"""Improvement of MIOST-K over MY by tidal regime, R = 50 km.

Classes by sigma_tide (standard deviation of the hourly tide gauge series minus
its 25 h running mean, CalVal; see tidal_regime_proxy.py): microtidal < 5 cm,
weak 5-20 cm, meso 20-60 cm, macrotidal > 60 cm.
Input: tidal_regime_proxy.csv, station_regions_v2.csv, per-station metrics."""
import os
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
S = pd.read_csv(HERE / "station_regions_v2.csv").set_index("station")
T = pd.read_csv(HERE / "tidal_regime_proxy.csv").set_index("station").tide_std_cm
d = pd.read_csv(HERE / "per_station_radius_with_pixels_v2.csv")
d = d[d.radius == 50].copy()
d["tide"], d["iho"] = d.station.map(T), d.station.map(S.iho)
d = d.dropna(subset=["tide"])
bins, labs = [0, 5, 20, 60, 400], ["micro <5", "weak 5-20", "meso 20-60", "macro >60"]
d["tb"] = pd.cut(d.tide, bins, labels=labs)
c, s = d[d.phase == "CalVal"], d[d.phase == "Science"]
pd.set_option("display.width", 200)


def table(x):
    g = x.groupby("tb", observed=True)
    return pd.DataFrame(dict(n=g.size(), medPI=g.pi.median(), pct_improved=g.apply(lambda q: 100 * (q.dr > 0).mean()),
                             rmsd_my=g.rmsd_my.median(), corr_my=g.corr_my.median())).round(2)


print("== median sigma_tide by region (cm)"); print(c.groupby("reg").tide.median().round(1).to_string())
print("\n== CalVal, near-swath stations by tidal class"); print(table(c[c.near]).to_string())
print("\n== CalVal, far-swath stations by tidal class"); print(table(c[~c.near]).to_string())
print("\n== Science, all stations by tidal class"); print(table(s).to_string())
a, b, e = c[c.near], c[~c.near], s
print("\nSpearman(tide, PI): CalVal near %.2f (p=%.1e) | CalVal far %.2f (p=%.2f) | Science %.2f (p=%.1e)" %
      (*spearmanr(a.tide, a.pi), *spearmanr(b.tide, b.pi), *spearmanr(e.tide, e.pi)))
w = a[(a.tide >= 5) & (a.tide < 20)]
print("weak-tide near stations: n=%d | transition zone %d | northern Kattegat %d | other %d" %
      (len(w), (w.reg == "Transition zone").sum(), ((w.iho == "Kattegat") & (w.reg != "Transition zone")).sum(),
       ((w.iho != "Kattegat") & (w.reg != "Transition zone")).sum()))
print("macrotidal near stations by region:", a[a.tide >= 60].reg.value_counts().to_dict())
