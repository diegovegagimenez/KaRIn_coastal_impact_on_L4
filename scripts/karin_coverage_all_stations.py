#!/usr/bin/env python3
"""Cobertura real de KaRIn L3 (producto por defecto: ssha_filtered, quality_flag==0)
alrededor de TODAS las estaciones validadas en CalVal.

Para cada estacion y cada pasada CalVal (pases 1/3/14/16) cuenta:
  - n_valid_R : pixeles L3 validos (ssha_filtered finito y QF==0) a <= R km
  - n_grid_R  : pixeles de la malla del swath (validos o no) a <= R km
para R = 10 y 50 km. Resumen por estacion: dias con dato valido, mediana de
pixeles por pasada, y si la estacion cae geometricamente bajo el swath.

Salida: results/l3_near_swath_scale/karin_coverage_all_stations.csv
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

ART = Path(os.environ.get("KARIN_PROJECT_DIR", "data"))
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "l3_near_swath_scale"
RE = 6371.0
RADII = [10, 50]

D = pd.read_csv(HERE / "per_station_radius_with_pixels.csv")
st = D[D.phase == "CalVal"].drop_duplicates("station")[["station", "latitude", "longitude", "region", "near"]]
st = st.reset_index(drop=True)


def xyz(lat, lon):
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


def chord(r_km):
    return 2 * np.sin(r_km / RE / 2)


P = xyz(st.latitude.values, st.longitude.values)
rows = []
files = sorted((ART / "data/raw/swot_l3_expert").glob("SWOT_L3_LR_SSH_Expert_*.nc"))
for i, f in enumerate(files):
    ds = xr.open_dataset(f)
    lat = ds.latitude.values
    lon = ((ds.longitude.values + 180) % 360) - 180
    ok = np.isfinite(ds.ssha_filtered.values) & (ds.quality_flag.values == 0)
    t = pd.Timestamp(np.nanmedian(ds.time.values.astype("int64"))).normalize()
    ds.close()
    box = (lat > 28) & (lat < 74) & (lon > -32) & (lon < 47)
    if not box.any():
        continue
    tg = cKDTree(xyz(lat[box], lon[box]))
    tv = cKDTree(xyz(lat[box & ok], lon[box & ok])) if (box & ok).any() else None
    rec = {"station": st.station.values, "file": f.name, "date": t}
    for r in RADII:
        rec[f"n_grid_R{r}"] = tg.query_ball_point(P, chord(r), return_length=True)
        rec[f"n_valid_R{r}"] = (tv.query_ball_point(P, chord(r), return_length=True)
                                if tv is not None else np.zeros(len(st), int))
    df = pd.DataFrame(rec)
    rows.append(df[df.n_grid_R50 > 0])
    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{len(files)}", flush=True)

A = pd.concat(rows)
A.to_csv(OUT / "karin_coverage_per_overpass.csv", index=False)
g = A.groupby("station")
S = pd.DataFrame({
    "n_overpasses": g.size(),
    "days_grid_R50": g.apply(lambda x: x.date.nunique()),
    "days_valid_R50": g.apply(lambda x: x[x.n_valid_R50 > 0].date.nunique()),
    "days_valid_R10": g.apply(lambda x: x[x.n_valid_R10 > 0].date.nunique()),
    "med_valid_R50": g.n_valid_R50.median(),
    "med_grid_R50": g.n_grid_R50.median(),
    "med_valid_R10": g.n_valid_R10.median(),
})
S = st.set_index("station").join(S).fillna(0)
S.to_csv(OUT / "karin_coverage_all_stations.csv")
pd.set_option("display.width", 220)
print(f"stations: {len(S)}; near (<2 km swath grid): {int(S.near.sum())}")
print("with swath grid within 50 km:", int((S.days_grid_R50 > 0).sum()))
print("with >=1 day of valid KaRIn within 50 km:", int((S.days_valid_R50 > 0).sum()))
print("\nNEAR stations, days with valid KaRIn data within 50 km (of ~100 CalVal days):")
print(S[S.near].days_valid_R50.describe().round(1).to_string())
print("\nNEAR stations with < 30 days of valid data within 50 km:")
print(S[S.near & (S.days_valid_R50 < 30)][["region", "latitude", "longitude", "days_grid_R50",
                                            "days_valid_R50", "med_valid_R50"]].to_string())
print("\nFAR stations by days of valid data within 50 km:")
print(pd.cut(S[~S.near].days_valid_R50, [-1, 0, 29, 59, 200]).value_counts().sort_index().to_string())
