#!/usr/bin/env python3
"""L3 SWOT KaRIn vs MIOST-K vs MY vs tide gauges, CalVal, near-swath stations,
radius sweep 10-100 km.

Builds the caches (tide gauge + DAC, MIOST-K, MY, L3) used by
l3_near_swath_scale_qc.py and l3_instantaneous.py. The L3 series of the paper
come from l3_near_swath_scale_qc.py (default product, quality_flag == 0).

- Tide gauge: P1.process_tg_detided (UTide) minus the DAC (P1.extract_dac_daily,
  fixed radius 50 km).
- MIOST-K: spatial median of `sla` within R km on the native 0.125 deg grid.
- MY: conservative regrid to the 0.125 deg grid, then spatial median within R.
- L3: spatial median of the L3 SSH within R km, per pass (CalVal passes
  1, 3, 14, 16).
- Joint 3-sigma filter, re-centring and LOESS (15 d) per station and radius on
  the merge of common dates.
- RMSD and correlation with respect to the tide gauge;
  PI = 100 * (RMSD_L4 - RMSD_L3) / RMSD_L4 (positive = L3 better than L4).

Output (results/l3_near_swath_scale/):
  per_station_radius.csv, summary_by_radius.csv, l3_cache.pkl, tg_dac_cache.pkl

Usage:
    python l3_near_swath_scale.py
"""
from __future__ import annotations

import os
import pickle
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================
ART_DIR = Path(os.environ.get("KARIN_PROJECT_DIR", "data"))
PIPE_DIR = ART_DIR / "PIPELINE_FINAL_LIMPIO"
OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "l3_near_swath_scale"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PIPE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import P1_altimetry_tidegauge_processing as P1  # noqa: E402

PATHS = {
    "tg_dir": ART_DIR / "data/processed_NRT/tide_gauges_nrt",
    "l3_dir": ART_DIR / "data/raw/swot_l3_expert",
    "l4_dir": ART_DIR / "data/raw/miost_l4",
    "dac_dir": ART_DIR / "data/raw/dac_l4",
    "duacs_my": ART_DIR / "data/raw/duacs_l4_my/duacs_sla_eur_MY_0625deg_swot_period.nc",
    "scale_dir": PIPE_DIR / "results_periodo_extendido",
    "prox_csv": PIPE_DIR / "swot_proximity_by_station.csv",
}

RADII = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
RMSD_CAP = 50.0           # cm
NEAR_KM = 2.0              # near-swath threshold
CALVAL_START = pd.Timestamp("2023-03-28")
CALVAL_END = pd.Timestamp("2023-07-05")

LOESS_WINDOW_DAYS = 15.0
SIGMA_THRESHOLD = 3.0
DAC_RADIUS_KM = P1.DAC_RADIUS_DEFAULT     # 50 km, fixed
BOX_MARGIN = 1.25                          # same default margin as the pipeline

# --- SWOT L3 quality_flag ---------------------------------------------------
# This script uses `ssha_unedited` (calibrated, not edited) and masks only the
# flags that are not ocean data (100 = bad_swath_extremities, 101 = not_on_sea,
# 102 = no_data). The paper uses l3_near_swath_scale_qc.py (ssha_filtered,
# quality_flag == 0).
QF_MEANING = {
    0: "good", 3: "eclipse", 5: "local_outliers", 10: "bad_quality_coast",
    18: "ocean_unsure", 19: "ice_unsure", 20: "ice", 25: "rain",
    30: "soft_outliers", 50: "extremes", 70: "mission_event",
    100: "bad_swath_extremities", 101: "not_on_sea", 102: "no_data",
}
QF_EXCLUDE = {100, 101, 102}


def haversine(lon1, lat1, lon2, lat2):
    lon1, lon2, lat1, lat2 = map(np.deg2rad, [lon1, lon2, lat1, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * 6371.0 * np.arcsin(np.sqrt(a))


# =============================================================================
# 1. FIXED PANEL OF NEAR-SWATH STATIONS (CalVal, 10-100 km)
# =============================================================================

def build_near_panel() -> pd.DataFrame:
    sets = []
    for r in RADII:
        d = pd.read_csv(PATHS["scale_dir"] / f"optimal_radius_fixed{r}" / "overall_metrics_LOESS_PARALLEL_CalVal_v7.csv")
        d.columns = [c.replace("_loess", "") for c in d.columns]
        rc = [c for c in d.columns if c.startswith("rmsd_")]
        d = d[~(d[rc] > RMSD_CAP).any(axis=1)]
        sets.append(set(d.station))
    panel = set.intersection(*sets)

    prox = pd.read_csv(PATHS["prox_csv"])
    dist = dict(zip(prox.station, prox.swot_track_distance_km))
    near = sorted(s for s in panel if dist.get(s, np.inf) <= NEAR_KM)

    coords = pd.read_csv(PATHS["scale_dir"] / "optimal_radius_fixed50" / "overall_metrics_LOESS_PARALLEL_CalVal_v7.csv")
    coords = coords[coords.station.isin(near)][["station", "latitude", "longitude"]].drop_duplicates("station")
    coords = coords.reset_index(drop=True)
    print(f"[PANEL] near-swath CalVal, fixed panel 10-100 km: n={len(coords)} "
          f"(of {len(panel)} in the fixed panel, threshold {NEAR_KM} km)")
    return coords


# =============================================================================
# 2. L3 EXTRACTION (one pass per file, all radii and stations)
# =============================================================================

def extract_l3(stations: pd.DataFrame) -> dict[str, pd.DataFrame]:
    l3_dir = PATHS["l3_dir"]
    files = sorted(l3_dir.glob("SWOT_L3_LR_SSH_Expert_*.nc"))
    print(f"[L3] {len(files)} local files to process (CalVal passes 1/3/14/16)")
    print(f"  variable: ssha_unedited + mask excluding only "
          f"{sorted(QF_EXCLUDE)} ({', '.join(QF_MEANING[f] for f in sorted(QF_EXCLUDE))})")

    rows: dict[str, list[dict]] = {sid: [] for sid in stations.station}
    max_r = max(RADII)

    for i, nc_path in enumerate(files):
        try:
            ds = xr.open_dataset(nc_path)
            lat = ds["latitude"].values
            lon = ((ds["longitude"].values + 180.0) % 360.0) - 180.0
            ssha = ds["ssha_unedited"].values
            qflag = ds["quality_flag"].values
            time_vals = ds["time"].values
            ds.close()
        except Exception as e:
            print(f"  [WARN] {nc_path.name}: {e}")
            continue

        # only land (not_on_sea) and no_data/swath edges are excluded; the other
        # flags are kept
        ssha = np.where(np.isin(qflag, list(QF_EXCLUDE)), np.nan, ssha)

        for sid, lat_tg, lon_tg in stations.itertuples(index=False):
            lat_diff = np.abs(lat - lat_tg)
            lon_diff = np.abs(lon - lon_tg)
            bbox = (lat_diff < 1.0) & (lon_diff < 1.0 / np.cos(np.deg2rad(lat_tg)))
            if not np.any(bbox):
                continue

            dist = np.full_like(lat, np.inf)
            dist[bbox] = haversine(lon[bbox], lat[bbox], lon_tg, lat_tg)

            line_mask_max = (dist <= max_r).any(axis=1)
            line_idx = np.where(line_mask_max)[0]
            if len(line_idx) == 0:
                continue

            rep_time = pd.Timestamp(np.median(time_vals[line_idx].astype("int64")))
            row = {"time": rep_time}
            any_valid = False
            for r in RADII:
                m = dist <= r
                vals = ssha[m]
                valid = vals[np.isfinite(vals)]
                if len(valid) > 0:
                    row[f"SLA_L3_R{r}"] = float(np.nanmedian(valid))
                    any_valid = True
                else:
                    row[f"SLA_L3_R{r}"] = np.nan
            if any_valid:
                rows[sid].append(row)

        if (i + 1) % 50 == 0:
            print(f"  [L3] {i + 1}/{len(files)} files processed")

    out = {}
    for sid, recs in rows.items():
        if not recs:
            out[sid] = pd.DataFrame(columns=["time"] + [f"SLA_L3_R{r}" for r in RADII])
            continue
        df = pd.DataFrame(recs).sort_values("time").reset_index(drop=True)
        df["time"] = pd.to_datetime(df["time"]).dt.normalize()
        # if there is more than one pass on the same day, take the median
        df = df.groupby("time", as_index=False).median(numeric_only=True)
        out[sid] = df
    return out


# =============================================================================
# 3a. Detided tide gauge (UTide, full record) + DAC (fixed radius 50 km),
#     with the P1 functions. Independent of the altimetry extraction radius.
# =============================================================================

class _FilteredDacDir:
    """Wrapper of PATHS['dac_dir'] whose .glob() only returns the 6-hourly DAC
    files within the CalVal window."""

    def __init__(self, real_dir: Path, start: pd.Timestamp, end: pd.Timestamp):
        self.real_dir = real_dir
        epoch = pd.Timestamp("1950-01-01")
        self.jday_min = (start - epoch).days - 1
        self.jday_max = (end - epoch).days + 1

    def glob(self, pattern):
        out = []
        for f in sorted(self.real_dir.glob(pattern)):
            m = re.match(r"dac_dif_(\d+)_(\d+)", f.stem)
            if not m:
                continue
            jday = int(m.group(1))
            if self.jday_min <= jday <= self.jday_max:
                out.append(f)
        return out


def build_tg_dac(stations: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return {station: DataFrame(time, SLEV_detided_mdac)} not re-centred
    (re-centring is done in p4_filter_smooth, after intersecting with L3/L4)."""
    tg_coords = {}
    detided = {}
    for sid, lat_tg, lon_tg in stations.itertuples(index=False):
        pkl_path = PATHS["tg_dir"] / f"{sid}.pkl"
        res, info = P1.process_tg_detided(pkl_path)
        if res is None:
            print(f"  [WARN] TG detiding failed for {sid}: {info}")
            continue
        detided[sid] = res
        tg_coords[sid] = (lat_tg, lon_tg)

    print(f"[TG] {len(detided)}/{len(stations)} stations detided (UTide)")

    dac_dir = _FilteredDacDir(PATHS["dac_dir"], CALVAL_START, CALVAL_END)
    dac = P1.extract_dac_daily(dac_dir, tg_coords, DAC_RADIUS_KM, BOX_MARGIN)

    out = {}
    for sid, tg_df in detided.items():
        d = tg_df.copy()
        dac_df = dac.get(sid)
        if dac_df is not None and len(dac_df):
            d = pd.merge(d, dac_df, on="time_d", how="left")
        if "dac" in d.columns:
            d["SLEV_detided_mdac"] = d["SLEV_detided"] - d["dac"]
        else:
            d["SLEV_detided_mdac"] = d["SLEV_detided"]
        d = d.rename(columns={"time_d": "time"})
        d["time"] = pd.to_datetime(d["time"]).dt.normalize()
        d = d[(d.time >= CALVAL_START) & (d.time <= CALVAL_END)]
        out[sid] = d[["time", "SLEV_detided_mdac"]]
    return out


# =============================================================================
# 3b. MIOST-K L4 EXTRACTION (median within R km, native 0.125 deg grid;
#     same as extract_miost_multiradius with common_grid='coarsest')
# =============================================================================

def extract_l4(stations: pd.DataFrame) -> dict[str, pd.DataFrame]:
    l4_dir = PATHS["l4_dir"]
    all_files = sorted(l4_dir.glob("dt_global_allsat_phy_l4_*.nc"))

    def file_date(p: Path):
        m = re.search(r"phy_l4_(\d{8})_", p.name)
        return pd.Timestamp(m.group(1)) if m else None

    files = [(f, file_date(f)) for f in all_files]
    files = [(f, d) for f, d in files if d is not None and CALVAL_START <= d <= CALVAL_END]
    print(f"[L4] {len(files)} MIOST-K files in the CalVal window "
          f"({CALVAL_START.date()} -> {CALVAL_END.date()})")

    if not files:
        raise RuntimeError("No MIOST L4 files found in the CalVal window")

    ds_ref = xr.open_dataset(files[0][0])
    lat1d = ds_ref["latitude"].values
    lon1d = ds_ref["longitude"].values
    ds_ref.close()

    max_r = max(RADII)
    station_idx: dict[str, tuple] = {}
    for sid, lat_tg, lon_tg in stations.itertuples(index=False):
        deg_pad = max_r / 111.0 + 0.2
        lat_mask = np.abs(lat1d - lat_tg) <= deg_pad
        lon_diff = np.abs(((lon1d - lon_tg + 180.0) % 360.0) - 180.0)
        lon_mask = lon_diff <= deg_pad / np.cos(np.deg2rad(lat_tg))
        lat_idx = np.where(lat_mask)[0]
        lon_idx = np.where(lon_mask)[0]
        sub_lat = lat1d[lat_idx]
        sub_lon = lon1d[lon_idx]
        LON, LAT = np.meshgrid(sub_lon, sub_lat)
        dist = haversine(LON, LAT, lon_tg, lat_tg)
        station_idx[sid] = (lat_idx, lon_idx, dist)

    rows: dict[str, list[dict]] = {sid: [] for sid in stations.station}

    for i, (fpath, date) in enumerate(files):
        try:
            ds = xr.open_dataset(fpath)
            sla_full = ds["sla"].isel(time=0).values
            ds.close()
        except Exception as e:
            print(f"  [WARN] {fpath.name}: {e}")
            continue

        for sid, (lat_idx, lon_idx, dist) in station_idx.items():
            sub = sla_full[np.ix_(lat_idx, lon_idx)]
            row = {"time": date}
            any_valid = False
            for r in RADII:
                m = dist <= r
                vals = sub[m]
                valid = vals[np.isfinite(vals)]
                if len(valid) > 0:
                    row[f"SLA_L4_R{r}"] = float(np.nanmedian(valid))
                    any_valid = True
                else:
                    row[f"SLA_L4_R{r}"] = np.nan
            if any_valid:
                rows[sid].append(row)

        if (i + 1) % 20 == 0:
            print(f"  [L4] {i + 1}/{len(files)} files processed")

    out = {}
    for sid, recs in rows.items():
        if not recs:
            out[sid] = pd.DataFrame(columns=["time"] + [f"SLA_L4_R{r}" for r in RADII])
            continue
        df = pd.DataFrame(recs).sort_values("time").reset_index(drop=True)
        df["time"] = pd.to_datetime(df["time"]).dt.normalize()
        out[sid] = df
    return out


# =============================================================================
# 3c. MY L4 EXTRACTION (nadir only): conservative regrid of the DUACS native grid
#     (0.0625 deg) to the MIOST grid (0.125 deg), then spatial median per radius.
# =============================================================================

def extract_duacs_my(stations: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ref_file = sorted(PATHS["l4_dir"].glob("dt_global_allsat_phy_l4_*.nc"))[0]
    ds_ref = xr.open_dataset(ref_file)
    tgt_lat = ds_ref["latitude"].values
    tgt_lon = ds_ref["longitude"].values
    ds_ref.close()

    tg_coords = {sid: (lat, lon) for sid, lat, lon in stations.itertuples(index=False)}

    print(f"[DUACS-MY] {PATHS['duacs_my'].name}: extract_single_nc_multiradius "
          f"(conservative regrid -> MIOST grid) ...")
    out_raw, _ncells = P1.extract_single_nc_multiradius(
        PATHS["duacs_my"], "sla_my", tg_coords, RADII, BOX_MARGIN,
        tgt_lat=tgt_lat, tgt_lon=tgt_lon, method="conservative")

    out = {}
    for sid, df in out_raw.items():
        d = df.rename(columns={"time_d": "time"})
        d["time"] = pd.to_datetime(d["time"]).dt.normalize()
        d = d[(d.time >= CALVAL_START) & (d.time <= CALVAL_END)]
        rename = {f"sla_my|{r}": f"SLA_DUACS_R{r}" for r in RADII}
        d = d.rename(columns=rename)
        keep_cols = ["time"] + [c for c in rename.values() if c in d.columns]
        out[sid] = d[keep_cols]
    print(f"  -> {len(out)} stations with DUACS-MY data")
    return out


# =============================================================================
# 5. P4: joint 3-sigma filter (+ re-centring) [+ optional 15 d LOESS]
#
# `smooth=True`  -> 3-sigma + LOESS (as in the paper).
# `smooth=False` -> no temporal filtering: 3-sigma filter and re-centring only;
#   raw daily values of TG, L3 and MIOST-K are compared.
# =============================================================================

def p4_filter_smooth(df: pd.DataFrame, cols: list[str], smooth: bool = True) -> pd.DataFrame:
    d = df.dropna(subset=cols).copy()
    if len(d) < 8:
        return d.iloc[0:0]

    mask = pd.Series(True, index=d.index)
    for c in cols:
        mu, sd = d[c].mean(), d[c].std()
        mask &= (np.abs(d[c] - mu) <= SIGMA_THRESHOLD * sd)
    d = d[mask].copy()
    if len(d) < 8:
        return d.iloc[0:0]

    for c in cols:
        d[c] = d[c] - d[c].mean()

    if not smooth:
        return d

    t_num = (d["time"] - d["time"].min()).dt.total_seconds() / 86400.0
    t_range = t_num.max() - t_num.min()
    if t_range > 0:
        frac = min(LOESS_WINDOW_DAYS / t_range, 1.0)
        for c in cols:
            d[c] = lowess(d[c].values, t_num.values, frac=frac, return_sorted=False)
    return d


# =============================================================================
# 6. PER STATION / PER RADIUS METRICS
# =============================================================================

def compute_metrics(stations: pd.DataFrame, l3: dict, tg_dac: dict, l4: dict,
                    smooth: bool = True) -> pd.DataFrame:
    records = []
    for sid, lat_tg, lon_tg in stations.itertuples(index=False):
        df_l3 = l3.get(sid, pd.DataFrame())
        df_tg = tg_dac.get(sid, pd.DataFrame())
        df_l4 = l4.get(sid, pd.DataFrame())
        if df_l3.empty or df_tg.empty or df_l4.empty:
            continue

        for r in RADII:
            l3_col = f"SLA_L3_R{r}"
            l4_col = f"SLA_L4_R{r}"
            if l4_col not in df_l4.columns:
                continue

            merged = pd.merge(df_tg, df_l3[["time", l3_col]], on="time", how="inner")
            merged = pd.merge(merged, df_l4[["time", l4_col]], on="time", how="inner")
            if merged.empty:
                continue

            cols = ["SLEV_detided_mdac", l3_col, l4_col]
            d = p4_filter_smooth(merged, cols, smooth=smooth)
            if len(d) < 8:
                continue

            tg = d["SLEV_detided_mdac"].values * 100.0
            l3v = d[l3_col].values * 100.0
            l4v = d[l4_col].values * 100.0

            rmsd_l3 = float(np.sqrt(np.mean((tg - l3v) ** 2)))
            rmsd_l4 = float(np.sqrt(np.mean((tg - l4v) ** 2)))
            corr_l3 = float(np.corrcoef(tg, l3v)[0, 1])
            corr_l4 = float(np.corrcoef(tg, l4v)[0, 1])

            # RMSD cap as in the rest of the paper: a few high-latitude stations (ice in
            # the Gulf of Bothnia) show spurious SSH of tens of metres that the
            # 3-sigma filter does not detect; these rows are discarded.
            if rmsd_l3 > RMSD_CAP or rmsd_l4 > RMSD_CAP:
                continue

            records.append(dict(
                station=sid, latitude=lat_tg, longitude=lon_tg, radius=r, n=len(d),
                rmsd_l3=rmsd_l3, rmsd_l4=rmsd_l4,
                corr_l3=corr_l3, corr_l4=corr_l4,
                delta_rmsd=rmsd_l4 - rmsd_l3,
                pi_rmsd=100.0 * (rmsd_l4 - rmsd_l3) / rmsd_l4 if rmsd_l4 != 0 else np.nan,
            ))

    return pd.DataFrame(records)


def compute_metrics_3products(stations: pd.DataFrame, l3: dict, tg_dac: dict, l4: dict,
                              duacs: dict, smooth: bool = True) -> pd.DataFrame:
    """As compute_metrics, adding MY (nadir only) as a third product. The joint
    3-sigma filter and the LOESS are applied to the four series {TG, L3,
    MIOST-K, MY} at once."""
    records = []
    for sid, lat_tg, lon_tg in stations.itertuples(index=False):
        df_l3 = l3.get(sid, pd.DataFrame())
        df_tg = tg_dac.get(sid, pd.DataFrame())
        df_l4 = l4.get(sid, pd.DataFrame())
        df_du = duacs.get(sid, pd.DataFrame())
        if df_l3.empty or df_tg.empty or df_l4.empty or df_du.empty:
            continue

        for r in RADII:
            l3_col = f"SLA_L3_R{r}"
            l4_col = f"SLA_L4_R{r}"
            du_col = f"SLA_DUACS_R{r}"
            if l4_col not in df_l4.columns or du_col not in df_du.columns:
                continue

            merged = pd.merge(df_tg, df_l3[["time", l3_col]], on="time", how="inner")
            merged = pd.merge(merged, df_l4[["time", l4_col]], on="time", how="inner")
            merged = pd.merge(merged, df_du[["time", du_col]], on="time", how="inner")
            if merged.empty:
                continue

            cols = ["SLEV_detided_mdac", l3_col, l4_col, du_col]
            d = p4_filter_smooth(merged, cols, smooth=smooth)
            if len(d) < 8:
                continue

            tg = d["SLEV_detided_mdac"].values * 100.0
            l3v = d[l3_col].values * 100.0
            l4v = d[l4_col].values * 100.0
            duv = d[du_col].values * 100.0

            rmsd_l3 = float(np.sqrt(np.mean((tg - l3v) ** 2)))
            rmsd_l4 = float(np.sqrt(np.mean((tg - l4v) ** 2)))
            rmsd_duacs = float(np.sqrt(np.mean((tg - duv) ** 2)))
            corr_l3 = float(np.corrcoef(tg, l3v)[0, 1])
            corr_l4 = float(np.corrcoef(tg, l4v)[0, 1])
            corr_duacs = float(np.corrcoef(tg, duv)[0, 1])

            if max(rmsd_l3, rmsd_l4, rmsd_duacs) > RMSD_CAP:
                continue

            records.append(dict(
                station=sid, latitude=lat_tg, longitude=lon_tg, radius=r, n=len(d),
                rmsd_l3=rmsd_l3, rmsd_l4=rmsd_l4, rmsd_duacs=rmsd_duacs,
                corr_l3=corr_l3, corr_l4=corr_l4, corr_duacs=corr_duacs,
                # PI = 100*(baseline - test)/baseline, positivo = test mejor
                pi_l3_vs_duacs=100.0 * (rmsd_duacs - rmsd_l3) / rmsd_duacs if rmsd_duacs else np.nan,
                pi_l4_vs_duacs=100.0 * (rmsd_duacs - rmsd_l4) / rmsd_duacs if rmsd_duacs else np.nan,
                pi_l3_vs_l4=100.0 * (rmsd_l4 - rmsd_l3) / rmsd_l4 if rmsd_l4 else np.nan,
            ))

    return pd.DataFrame(records)


def summarize_3products(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in RADII:
        d = df[df.radius == r]
        if len(d) < 5:
            continue
        rows.append(dict(
            radius=r, n=len(d),
            median_rmsd_l3=d.rmsd_l3.median(), median_rmsd_l4=d.rmsd_l4.median(),
            median_rmsd_duacs=d.rmsd_duacs.median(),
            mean_rmsd_l3=d.rmsd_l3.mean(), mean_rmsd_l4=d.rmsd_l4.mean(),
            mean_rmsd_duacs=d.rmsd_duacs.mean(),
            median_corr_l3=d.corr_l3.median(), median_corr_l4=d.corr_l4.median(),
            median_corr_duacs=d.corr_duacs.median(),
            median_pi_l4_vs_duacs=d.pi_l4_vs_duacs.median(),   # valor anadido KaRIn (L4 vs DUACS)
            median_pi_l3_vs_duacs=d.pi_l3_vs_duacs.median(),   # L3 crudo vs DUACS
            median_pi_l3_vs_l4=d.pi_l3_vs_l4.median(),         # L3 vs L4
            pct_l4_better_than_duacs=100.0 * (d.rmsd_l4 < d.rmsd_duacs).mean(),
            pct_l3_better_than_duacs=100.0 * (d.rmsd_l3 < d.rmsd_duacs).mean(),
        ))
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in RADII:
        d = df[df.radius == r]
        if len(d) < 5:
            continue
        pi = d.pi_rmsd.dropna()
        w_stat, w_p = (np.nan, np.nan)
        if len(d) >= 5:
            try:
                w_stat, w_p = stats.wilcoxon(d.rmsd_l4, d.rmsd_l3)
            except Exception:
                pass
        rows.append(dict(
            radius=r, n=len(d),
            median_rmsd_l3=d.rmsd_l3.median(), median_rmsd_l4=d.rmsd_l4.median(),
            mean_rmsd_l3=d.rmsd_l3.mean(), mean_rmsd_l4=d.rmsd_l4.mean(),
            median_pi=pi.median(), mean_pi=pi.mean(),
            pct_l3_better=100.0 * (d.rmsd_l3 < d.rmsd_l4).mean(),
            wilcoxon_p=w_p,
        ))
    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def _load_or_build(cache_name: str, builder, *args):
    cache_path = OUT_DIR / cache_name
    if cache_path.exists():
        print(f"  -> using existing cache ({cache_path})")
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)
    result = builder(*args)
    with open(cache_path, "wb") as fh:
        pickle.dump(result, fh)
    print(f"  -> cache guardada en {cache_path}")
    return result


def main():
    stations = build_near_panel()
    stations.to_csv(OUT_DIR / "near_panel_stations.csv", index=False)

    print("\n[STEP] L3 extraction (this may take a while) ...")
    l3 = _load_or_build("l3_cache.pkl", extract_l3, stations)

    print("\n[STEP] Detided TG (UTide) + DAC (fixed radius 50 km) ...")
    tg_dac = _load_or_build("tg_dac_cache.pkl", build_tg_dac, stations)

    print("\n[STEP] MIOST-K L4 extraction (median within R km, native grid) ...")
    l4 = _load_or_build("l4_cache.pkl", extract_l4, stations)

    pd.set_option("display.width", 200)

    print("\n[STEP] Merge + P4 filter (3-sigma) + LOESS 15 d + metrics per station/radius ...")
    per_sr = compute_metrics(stations, l3, tg_dac, l4, smooth=True)
    per_sr.to_csv(OUT_DIR / "per_station_radius.csv", index=False, float_format="%.5f")
    print(f"  -> {len(per_sr)} rows (station x radius) saved")

    summary = summarize(per_sr)
    summary.to_csv(OUT_DIR / "summary_by_radius.csv", index=False, float_format="%.4f")
    print("\n" + "=" * 100)
    print("SUMMARY [LOESS 15 d]: L3 vs L4 vs TG -- CalVal, near-swath")
    print("=" * 100)
    print(summary.to_string(index=False))

    print("\n[STEP] Version without temporal filtering (3-sigma + re-centring only) ...")
    per_sr_raw = compute_metrics(stations, l3, tg_dac, l4, smooth=False)
    per_sr_raw.to_csv(OUT_DIR / "per_station_radius_raw.csv", index=False, float_format="%.5f")
    print(f"  -> {len(per_sr_raw)} rows (station x radius) saved")

    summary_raw = summarize(per_sr_raw)
    summary_raw.to_csv(OUT_DIR / "summary_by_radius_raw.csv", index=False, float_format="%.4f")
    print("\n" + "=" * 100)
    print("SUMMARY [no LOESS, raw daily values]: L3 vs L4 vs TG, CalVal, near-swath")
    print("=" * 100)
    print(summary_raw.to_string(index=False))

    print("\n[STEP] MY extraction (nadir only) ...")
    duacs = _load_or_build("duacs_cache.pkl", extract_duacs_my, stations)

    print("\n[STEP] 3 productos {L3, MIOST-K, DUACS-MY} + filtro P4 (3-sigma) + LOESS 15d ...")
    per_sr_3p = compute_metrics_3products(stations, l3, tg_dac, l4, duacs, smooth=True)
    per_sr_3p.to_csv(OUT_DIR / "per_station_radius_3products.csv", index=False, float_format="%.5f")
    print(f"  -> {len(per_sr_3p)} rows (station x radius) saved")

    summary_3p = summarize_3products(per_sr_3p)
    summary_3p.to_csv(OUT_DIR / "summary_by_radius_3products.csv", index=False, float_format="%.4f")
    print("\n" + "=" * 100)
    print("SUMMARY [LOESS 15 d]: L3 vs MIOST-K vs DUACS-MY vs TG -- "
          "CalVal, near-swath")
    print("=" * 100)
    print(summary_3p.to_string(index=False))

    print(f"\n[DONE] Results in: {OUT_DIR}")


if __name__ == "__main__":
    main()
