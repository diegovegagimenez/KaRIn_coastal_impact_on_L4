#!/usr/bin/env python3
"""Distance from each tide gauge to the KaRIn swath footprint of the CalVal passes.

Stations within 2 km of a swath pixel are classified as near-swath.

Input : SWOT L3 files in $KARIN_PROJECT_DIR/data/raw/swot_l3_expert and the
        P1 station table (R = 50 km, CalVal and Science).
Output: $KARIN_PROJECT_DIR/PIPELINE_FINAL_LIMPIO/swot_proximity_by_station.csv
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

BASE = Path(os.environ.get("KARIN_PROJECT_DIR", "data"))
L3_DIR = BASE / "data" / "raw" / "swot_l3_expert"
P1_DIR = BASE / "PIPELINE_FINAL_LIMPIO" / "results_periodo_extendido" / "optimal_radius_fixed50"
OUT = BASE / "PIPELINE_FINAL_LIMPIO" / "swot_proximity_by_station.csv"

PASSES = [1, 3, 14, 16]          # CalVal passes over the domain
EXTENT = [-12, 42, 30, 67]       # lon_min, lon_max, lat_min, lat_max
NEAR_KM = 2.0
FILES_PER_PASS = 10


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.deg2rad, [lon1, lat1, lon2, lat2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def swath_points(pass_num):
    lats, lons = [], []
    files = sorted(L3_DIR.glob(f"SWOT_L3_LR_SSH_Expert_*_{pass_num:03d}_*.nc"))
    for f in files[:FILES_PER_PASS]:
        with xr.open_dataset(f) as ds:
            la = ds["latitude"].values.ravel()
            lo = ds["longitude"].values.ravel()
        lo = np.where(lo > 180, lo - 360, lo)
        ok = (np.isfinite(la) & np.isfinite(lo) & (la > EXTENT[2]) & (la < EXTENT[3])
              & (lo > EXTENT[0]) & (lo < EXTENT[1]))
        lats.append(la[ok])
        lons.append(lo[ok])
    return np.concatenate(lats), np.concatenate(lons)


def main():
    st = pd.concat([pd.read_csv(P1_DIR / f"overall_metrics_LOESS_PARALLEL_{ph}_v7.csv")
                    [["station", "latitude", "longitude"]] for ph in ("CalVal", "Science")])
    st = st.drop_duplicates("station").reset_index(drop=True)
    tracks = {p: swath_points(p) for p in PASSES}
    dist = np.full(len(st), np.inf)
    for la, lo in tracks.values():
        for i, (y, x) in enumerate(zip(st.latitude, st.longitude)):
            dist[i] = min(dist[i], haversine(lo, la, x, y).min())
    st["swot_track_distance_km"] = dist
    st["is_near_swot"] = dist <= NEAR_KM
    OUT.parent.mkdir(parents=True, exist_ok=True)
    st.to_csv(OUT, index=False)
    print(f"{len(st)} stations, {int(st.is_near_swot.sum())} near-swath -> {OUT}")


if __name__ == "__main__":
    main()
