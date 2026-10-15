#!/usr/bin/env python3
"""Per-station table with the number of grid cells (common 0.125 deg grid) used,
by phase, radius, region and proximity.
Output: per_station_radius_with_pixels.csv"""
import os
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import Point
import final_v1_scale_cfg as C

BASE = Path(os.environ.get("KARIN_PROJECT_DIR", "data") + "/PIPELINE_FINAL_LIMPIO")
WIN = BASE / "results_periodo_extendido"
RADII = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
def load(r, ph):
    d = pd.read_csv(WIN / f"optimal_radius_fixed{r}" / f"overall_metrics_LOESS_PARALLEL_{ph}_v7.csv")
    d.columns = [c.replace("_loess", "") for c in d.columns]
    rc = [c for c in d.columns if c.startswith("rmsd_")]
    d = d[~(d[rc] > 50).any(axis=1)].reset_index(drop=True)
    d["pi"] = 100 * (d.rmsd_my - d.rmsd_miost) / d.rmsd_my
    d["dr"] = d.rmsd_my - d.rmsd_miost
    d["phase"] = ph; d["radius"] = r
    return d
D = pd.concat([load(r, ph) for r in RADII for ph in ("CalVal", "Science")])
sta = D.drop_duplicates("station")[["station", "latitude", "longitude"]].reset_index(drop=True)
gdf = gpd.read_file(os.environ.get("KARIN_IHO_SHP", os.environ.get("KARIN_PROJECT_DIR", "data") + "/World_Seas_IHO_v2/World_Seas_IHO_v2.shp")).to_crs("EPSG:4326")
pts = gpd.GeoDataFrame(sta, geometry=[Point(x, y) for x, y in zip(sta.longitude, sta.latitude)], crs="EPSG:4326")
j = gpd.sjoin_nearest(pts, gdf[["geometry", "NAME"]], how="left"); j = j[~j.index.duplicated()]
REG = dict(zip(j.station, j.NAME.map(lambda n: C.RMAP.get(n, "Unknown"))))
prox = pd.read_csv(BASE / "swot_proximity_by_station.csv")
DIST = dict(zip(prox.station, prox.swot_track_distance_km))
D["region"] = D.station.map(REG); D["near"] = D.station.map(DIST).fillna(999) <= 2
NC = pd.read_csv(WIN / "sensitivity_per_station_radius.csv")[["station", "phase", "radius", "ncells_my", "ncells_miost"]]
D = D.merge(NC, on=["station", "phase", "radius"], how="left")
assert (D.ncells_my == D.ncells_miost).all()
D.to_csv(Path(__file__).parent / "per_station_radius_with_pixels.csv", index=False)

print("== 50 km all stations")
for ph in ("CalVal", "Science"):
    x = D[(D.phase == ph) & (D.radius == 50)]
    print(ph, len(x), "meanPI %.2f medPI %.2f meandr %.3f pct %.1f maxdr %.2f" % (x.pi.mean(), x.pi.median(), x.dr.mean(), 100*(x.dr > 0).mean(), x.dr.max()),
          "pix med %d [%d-%d] tot %d" % (x.ncells_my.median(), x.ncells_my.min(), x.ncells_my.max(), x.ncells_my.sum()))
    print(x.groupby("region").agg(n=("station", "size"), pix=("ncells_my", "median"), pmin=("ncells_my", "min"), pmax=("ncells_my", "max"), pct=("dr", lambda s: 100*(s > 0).mean())).to_string())
    if ph == "CalVal":
        print(x.groupby(["region", "near"]).agg(n=("station", "size"), pix=("ncells_my", "median"), med=("pi", "median")).to_string())
x = D[D.radius == 50]; print("pooled favour", (x.dr > 0).sum(), len(x))
# fixed panel
for ph in ("CalVal", "Science"):
    s = None
    for r in RADII:
        ss = set(D[(D.phase == ph) & (D.radius == r)].station); s = ss if s is None else s & ss
    P = D[(D.phase == ph) & D.station.isin(s)]
    print("== panel", ph, len(s), "near", P[P.radius == 10].near.sum())
    print(P.groupby(["radius", "near"]).agg(n=("station", "size"), pix=("ncells_my", "median"), pmin=("ncells_my", "min"), pmax=("ncells_my", "max")).unstack().to_string())
    print(P.groupby("radius").agg(pix=("ncells_my", "median"), pmin=("ncells_my", "min"), pmax=("ncells_my", "max"), med=("pi", "median"), mean=("pi", "mean"), pct=("dr", lambda s: 100*(s > 0).mean())).to_string())
    if ph == "CalVal":
        print(P[P.region.isin(["North West Shelf", "Baltic Sea", "Mediterranean Sea"])].groupby(["region", "near", "radius"]).agg(n=("station", "size"), pix=("ncells_my", "median")).unstack().to_string())
# regional, all available stations: pixels
print("== regional pixels (all available)")
print(D.groupby(["phase", "region", "radius"]).ncells_my.median().unstack().to_string())
