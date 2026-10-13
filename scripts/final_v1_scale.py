#!/usr/bin/env python3
"""Barrido de escala 10-100 km + near/far SWATH,
EXCLUSIVAMENTE sobre la ventana extendida (registro DT completo) y
EXCLUSIVAMENTE sobre la comparacion MY -> MIOST-K.

Reproduce la logica de highlights.py (mismo cap de RMSD, mismo mapeo de region,
mismas distancias SWOT) pero:
  - solo window = extendida
  - solo comparison = total (my -> miost)
  - anade el control de PANEL FIJO (estaciones presentes a todos los radios),
    que en V3 solo existia para la ventana comun.
"""
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy import stats

BASE = Path(os.environ.get("KARIN_PROJECT_DIR", "data") + "/PIPELINE_FINAL_LIMPIO")
IHO = Path(os.environ.get("KARIN_IHO_SHP", os.environ.get("KARIN_PROJECT_DIR", "data") + "/World_Seas_IHO_v2/World_Seas_IHO_v2.shp"))
OUT = Path(__file__).resolve().parent

WIN = BASE / "results_periodo_extendido"
RADII = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
PHASES = ["CalVal", "Science"]
CAP = 50.0
REGION_ORDER = ["Arctic", "Baltic Sea", "Black Sea", "Iberia-Biscay-Ireland",
                "Mediterranean Sea", "North West Shelf"]
RMAP = {'Mediterranean Sea - Western Basin':'Mediterranean Sea','Balearic (Iberian Sea)':'Mediterranean Sea',
'Alboran Sea':'Mediterranean Sea','Ligurian Sea':'Mediterranean Sea','Tyrrhenian Sea':'Mediterranean Sea',
'Strait of Gibraltar':'Mediterranean Sea','Mediterranean Sea - Eastern Basin':'Mediterranean Sea',
'Adriatic Sea':'Mediterranean Sea','Ionian Sea':'Mediterranean Sea','Aegean Sea':'Mediterranean Sea',
'Sea of Marmara':'Mediterranean Sea','Baltic Sea':'Baltic Sea','Gulf of Finland':'Baltic Sea',
'Gulf of Bothnia':'Baltic Sea','Gulf of Riga':'Baltic Sea','North Sea':'North West Shelf',
'Skagerrak':'North West Shelf','Kattegat':'North West Shelf','English Channel':'North West Shelf',
"Irish Sea and St. George's Channel":'North West Shelf','Bristol Channel':'North West Shelf',
'Inner Seas off the West Coast of Scotland':'North West Shelf','Celtic Sea':'Iberia-Biscay-Ireland',
'Bay of Biscay':'Iberia-Biscay-Ireland','North Atlantic Ocean':'Iberia-Biscay-Ireland',
'Norwegian Sea':'Arctic','Greenland Sea':'Arctic','Barentsz Sea':'Arctic','Black Sea':'Black Sea'}


def load(r, ph):
    f = WIN / f"optimal_radius_fixed{r}" / f"overall_metrics_LOESS_PARALLEL_{ph}_v7.csv"
    d = pd.read_csv(f)
    d.columns = [c.replace("_loess", "") for c in d.columns]
    rc = [c for c in d.columns if c.startswith("rmsd_")]
    return d[~(d[rc] > CAP).any(axis=1)].reset_index(drop=True)


# ---- region + distancia SWOT ------------------------------------------------
coords = {}
data = {}
for r in RADII:
    for ph in PHASES:
        d = load(r, ph)
        data[(r, ph)] = d
        for t in d.itertuples():
            coords.setdefault(t.station, (t.latitude, t.longitude))
sta = list(coords)
gdf = gpd.read_file(IHO).to_crs("EPSG:4326")
pts = gpd.GeoDataFrame({"station": sta},
                       geometry=[Point(coords[s][1], coords[s][0]) for s in sta], crs="EPSG:4326")
j = gpd.sjoin_nearest(pts, gdf[["geometry", "NAME"]], how="left", distance_col="_d")
j = j[~j.index.duplicated(keep="first")]
REGION = dict(zip(j["station"], j["NAME"].map(lambda n: RMAP.get(n, "Unknown"))))
prox = pd.read_csv(BASE / "results_P2_periodo_comun_50km" / "advanced_analysis_v2_fixed" /
                   "miost_multivariate__vs__miost_multivariate_karin" / "swot_proximity_by_station.csv")
DIST = dict(zip(prox.station, prox.swot_track_distance_km))
print(f"{len(sta)} estaciones; {sum(v <= 2 for v in DIST.values())} near-SWOT (<=2 km)")

# ---- PI por estacion --------------------------------------------------------
for (r, ph), d in data.items():
    with np.errstate(divide="ignore", invalid="ignore"):
        d["pi"] = 100 * (d.rmsd_my - d.rmsd_miost) / d.rmsd_my
    d["pi"] = d["pi"].replace([np.inf, -np.inf], np.nan)
    d["dr"] = d.rmsd_my - d.rmsd_miost
    d["region"] = d.station.map(REGION)
    d["dist"] = d.station.map(DIST)

# ---- panel fijo por fase: estaciones presentes a los 10 radios --------------
PANEL = {}
for ph in PHASES:
    s = set(data[(RADII[0], ph)].station)
    for r in RADII[1:]:
        s &= set(data[(r, ph)].station)
    PANEL[ph] = s
    n_near = sum(DIST.get(x, 999) <= 2 for x in s)
    print(f"panel fijo {ph}: {len(s)} estaciones ({n_near} near, {len(s)-n_near} far)")

res = {"nearfar": [], "network": [], "regional": [], "allsta": []}

for ph in PHASES:
    P = PANEL[ph]
    for r in RADII:
        d = data[(r, ph)]
        p = d[d.station.isin(P)]
        # near/far sobre panel fijo
        a = p[p.dist <= 2].pi.dropna()
        b = p[p.dist > 2].pi.dropna()
        _, pv = stats.mannwhitneyu(a, b, alternative="two-sided")
        res["nearfar"].append(dict(phase=ph, radius=r, n_near=len(a), n_far=len(b),
                                   near=round(float(a.median()), 3), far=round(float(b.median()), 3),
                                   gap=round(float(a.median() - b.median()), 3), p=float(pv)))
        # network-wide sobre panel fijo, y sobre todas las estaciones disponibles
        res["network"].append(dict(phase=ph, radius=r,
                                   panel=round(float(p.pi.dropna().median()), 3), n_panel=len(p),
                                   allst=round(float(d.pi.dropna().median()), 3), n_all=len(d)))
        # regional (todas las estaciones disponibles a ese radio, como en V3 tab:regional_scale)
        for g in ["Global"] + REGION_ORDER:
            gg = d if g == "Global" else d[d.region == g]
            v = gg.pi.dropna()
            if len(v) < 3:
                continue
            res["regional"].append(dict(phase=ph, radius=r, region=g, n=len(gg),
                                        pi=round(float(v.median()), 3),
                                        dr=round(float(gg.dr.dropna().mean()), 4),
                                        pct=round(float((v > 0).mean() * 100), 1)))

json.dump(res, open(OUT / "final_v1_scale.json", "w"), indent=1)

pd.set_option("display.width", 200)
print("\n=== NEAR/FAR, panel fijo, ventana extendida ===")
print(pd.DataFrame(res["nearfar"]).to_string(index=False))
print("\n=== NETWORK-WIDE mediana PI, panel fijo vs todas ===")
print(pd.DataFrame(res["network"]).to_string(index=False))
print("\n=== REGIONAL x radio (todas las estaciones disponibles) ===")
R = pd.DataFrame(res["regional"])
for ph in PHASES:
    print(f"\n-- {ph} --")
    print(R[R.phase == ph].pivot(index="region", columns="radius", values="pi").to_string())
print("\n=== n por region/radio (Science) ===")
print(R[R.phase == "Science"].pivot(index="region", columns="radius", values="n").to_string())
print("\n=== TOP configuraciones (ventana extendida, n>=20, por PI mediano) ===")
top = R[(R.region != "Global") & (R.n >= 20)].nlargest(10, "pi")
print(top.to_string(index=False))
print("\n=== TOP global ===")
print(R[R.region == "Global"].nlargest(6, "pi").to_string(index=False))
