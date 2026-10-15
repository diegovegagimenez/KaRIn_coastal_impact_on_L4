#!/usr/bin/env python3
"""Station-level statistics of Sect. 3.2: stations with the largest RMSD
reduction/degradation, overlap between phases and change in correlation."""
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
S = pd.read_csv(HERE / "station_regions_v2.csv").set_index("station")
d = pd.read_csv(HERE / "per_station_radius_with_pixels_v2.csv")
d = d[d.radius == 50].copy()
d["iho"] = d.station.map(S.iho)
d["name"] = d.station.str.replace(r"^[A-Z]{2}_TS_TG_", "", regex=True)
pd.set_option("display.width", 220)

for ph in ("CalVal", "Science"):
    x = d[d.phase == ph]
    cols = ["name", "latitude", "longitude", "iho", "reg", "rmsd_my", "rmsd_miost", "dr", "pi"] + (["near"] if ph == "CalVal" else [])
    print(f"===== {ph}: 12 largest RMSD reductions"); print(x.nlargest(12, "dr")[cols].round(2).to_string(index=False))
    print(f"----- {ph}: 6 largest degradations"); print(x.nsmallest(6, "dr")[cols].round(2).to_string(index=False))
    t = x.nlargest(20, "dr")
    print("top-20 by region:", t.reg.value_counts().to_dict(), "| near-swath:", int(t.near.sum()) if ph == "CalVal" else "n/a")

top = {ph: set(d[d.phase == ph].nlargest(20, "dr").station) for ph in ("CalVal", "Science")}
print("\nin the top-20 of BOTH phases:", sorted(s.split("TG_")[-1] for s in top["CalVal"] & top["Science"]))

rng = np.random.default_rng(0)


def ci(v):
    v = np.asarray(v)
    m = v[rng.integers(0, len(v), (10000, len(v)))].mean(axis=1)
    return v.mean(), *np.percentile(m, [2.5, 97.5])


print("\n== correlation change")
for ph in ("CalVal", "Science"):
    x = d[d.phase == ph].copy()
    x["dc"] = x.corr_miost - x.corr_my
    imp, deg = x[x.dr > 0], x[x.dr < 0]
    tz = x[x.reg == "Transition zone"]
    print(f"{ph}: corr increases at {100 * (imp.dc > 0).mean():.0f}% of stations with lower RMSD and decreases at "
          f"{100 * (deg.dc < 0).mean():.0f}% of those with higher RMSD | network mean dr_corr {ci(x.dc)[0]:+.4f} "
          f"[{ci(x.dc)[1]:+.4f},{ci(x.dc)[2]:+.4f}] | transition zone {ci(tz.dc)[0]:+.3f} [{ci(tz.dc)[1]:+.3f},{ci(tz.dc)[2]:+.3f}]")
