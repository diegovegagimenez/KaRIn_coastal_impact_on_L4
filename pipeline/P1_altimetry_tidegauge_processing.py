#!/usr/bin/env python3
"""P1: altimetry extraction and tide gauge validation.

For each tide gauge, the daily sea level anomaly of MY, MY-MIOST and MIOST-K is
extracted as the spatial median of the grid cells within a radius R of the
station (R = 5-400 km, step 5 km), after regridding to a common grid. Tide gauge
records are detided (UTide), corrected for the DAC, low-pass filtered (LOESS,
15 d) and compared with the altimetry (RMSD and correlation).

Usage (extended delayed-time record, fixed radius R):

    python P1_altimetry_tidegauge_processing.py --fixed-radius-km 50 \\
           --skip-per-station --skip-per-region \\
           --science-end 2025-07-06 --operational-window

Input (under $KARIN_PROJECT_DIR):
    data/processed_NRT/tide_gauges_nrt/*.pkl
    data/raw/duacs_l4_my/duacs_sla_eur_MY_0625deg_swot_period.nc
    data/raw/duacs_l4_my_miost/duacs_sla_eur_MY_MIOST_0625deg_swot_period.nc
    data/raw/miost_l4/dt_global_allsat_phy_l4_*.nc
    data/raw/dac_l4/dac_dif_*.nc
Output (--outdir):
    optimal_radius_fixed{R}/overall_metrics_LOESS_PARALLEL_{phase}_v7.csv
    sensitivity_per_station_radius.csv

"""
from __future__ import annotations

import os
import argparse
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")

try:                                   # optional progress bar
    from tqdm import tqdm
except Exception:                      # pragma: no cover
    def tqdm(x, **k):
        return x

# ============================================================================
# CONFIG. Set KARIN_PROJECT_DIR or pass --base-dir.
# ============================================================================
BASE_DIR_DEFAULT = Path(os.environ.get("KARIN_PROJECT_DIR", "data"))

PHASES = {                              # CalVal and Science phases
    "CalVal":  ("2023-03-28", "2023-07-05"),
    "Science": ("2023-07-06", None),
}
PRODUCTS = ["my", "my_miost", "miost"]
PRODUCT_SLA_COL = {"my": "sla_my", "my_miost": "sla_my_miost", "miost": "sla_miost"}

TIDAL_CONSTITUENTS = ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1"]
MIN_SPAN_YEARS = 0.25
MAX_GAP_DAYS = 15
MIN_COVERAGE = 0.90
SIGMA_THRESHOLD = 3.0
LOESS_WINDOW = 15
RMSD_PLAUSIBILITY_CAP_CM = 100.0
LAT_MIN, LAT_MAX = 27.0, 72.0
LON_MIN, LON_MAX = -15.0, 42.0
DAC_EPOCH = pd.Timestamp("1950-01-01")
EARTH_R_KM = 6371.0

RADII_DEFAULT = list(range(5, 401, 5))   # 5,10,...,400 km
DAC_RADIUS_DEFAULT = 50.0                # DAC is smooth: fixed radius (no sweep)

PRODUCT_STYLE = {                        # color-blind friendly (Wong)
    "my":       {"color": "#E69F00", "label": "MY (MIOST std., nadir)"},
    "my_miost": {"color": "#0072B2", "label": "MY-MIOST (MIOST multivariate, nadir)"},
    "miost":    {"color": "#009E73", "label": "MIOST_KaRIn (MIOST multivariate, +SWOT)"},
}


@dataclass
class Config:
    base_dir: Path = BASE_DIR_DEFAULT
    outdir: Path = Path("results_radius_sensitivity_commongrid_cipollini_r400")
    radii: list = field(default_factory=lambda: list(RADII_DEFAULT))
    dac_radius: float = DAC_RADIUS_DEFAULT
    box_margin: float = 1.25
    # common grid (regridding)
    common_grid: str = "coarsest"   # 'coarsest' (0.125°) | 'finest' | float (°)
    regrid_method: str = "conservative"  # 'conservative' (downsample) | 'bilinear' | 'nearest'
    # outputs
    export_ts: str = "optimal"   # 'none' | 'optimal' | 'all'
    emit_p1_metrics: bool = True
    from_csv: Path = None        # resume from the per-station CSV (no re-extraction)
    fixed_radius_km: int = 50
    # science_end: end date of the Science phase (YYYY-MM-DD) or None (open;
    #   closed by the common coverage of the products).
    #   "2025-07-06" truncates Science at the last delayed-time day (excludes NRT).
    science_end: str = None
    # triple_intersect: if True, the three products are evaluated over exactly
    #   the same days. If False, my_miost joins with an outer (left) join and does
    #   not trim MY/MIOST-K (extended window, MY vs MIOST-K).
    triple_intersect: bool = True

    @property
    def phases(self) -> dict:
        """Phase definition derived from the config (allows truncating Science)."""
        sci_end = self.science_end if self.science_end else None
        return {
            "CalVal":  ("2023-03-28", "2023-07-05"),
            "Science": ("2023-07-06", sci_end),
        }

    @property
    def paths(self) -> dict:
        b = self.base_dir
        return {
            "tg_dir":         b / "data/processed_NRT/tide_gauges_nrt",
            "duacs_my":       b / "data/raw/duacs_l4_my/duacs_sla_eur_MY_0625deg_swot_period.nc",
            "duacs_my_miost": b / "data/raw/duacs_l4_my_miost/duacs_sla_eur_MY_MIOST_0625deg_swot_period.nc",
            "miost_dir":      b / "data/raw/miost_l4",
            "dac_dir":        b / "data/raw/dac_l4",
        }


# ============================================================================
# HELPERS
# ============================================================================
def haversine(lon1, lat1, lon2, lat2):
    lon1, lon2, lat1, lat2 = map(np.deg2rad, [lon1, lon2, lat1, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))


def calculate_metrics(ground_truth, predicted):
    """RMSD (cm), VAR (cm²), STD (cm) and Pearson correlation."""
    diff = ground_truth - predicted
    N = np.sum(~np.isnan(diff))
    if N < 10:
        return np.nan, np.nan, np.nan, np.nan
    rmsd = np.sqrt(np.nanmean(diff ** 2)) * 100
    var = np.nanvar(diff) * 10_000
    std = np.nanstd(diff) * 100
    if np.nanstd(ground_truth) == 0 or np.nanstd(predicted) == 0:
        corr = np.nan
    else:
        mask = ~np.isnan(ground_truth) & ~np.isnan(predicted)
        corr = np.corrcoef(ground_truth[mask], predicted[mask])[0, 1]
    return rmsd, var, std, corr


def check_data_continuity(times, min_years=MIN_SPAN_YEARS,
                          max_gap_days=MAX_GAP_DAYS, min_coverage=MIN_COVERAGE):
    if len(times) < 2:
        return False
    ts = pd.Series(times).sort_values()
    span = (ts.max() - ts.min()).days / 365.25
    if span < min_years:
        return False
    gaps = ts.diff().dt.days.dropna()
    if gaps.max() > max_gap_days:
        return False
    expected = (ts.max() - ts.min()).days + 1
    if len(ts) / expected < min_coverage:
        return False
    return True


def apply_loess_filter(series, time_numeric, window_days=LOESS_WINDOW):
    """LOESS low-pass (fraction f = W/(T_max-T_min)). Falls back to the unfiltered series."""
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
    except Exception:
        return series.copy()
    t_range = time_numeric.max() - time_numeric.min()
    if t_range <= 0:
        return series.copy()
    frac = min(max(window_days / t_range, 0.01), 1.0)
    smoothed = lowess(series.values, time_numeric.values, frac=frac, return_sorted=False)
    return pd.Series(smoothed, index=series.index)


def utide_detide(time_vals, slev_vals, latitude):
    """UTide harmonic analysis (prior 3-sigma filter) -> detided SLEV."""
    try:
        from utide import solve, reconstruct
    except Exception as e:
        raise RuntimeError("'utide' is required to detide the tide gauge records.") from e
    try:
        slev = slev_vals.copy().astype(float)
        mu, sig = np.nanmean(slev), np.nanstd(slev)
        slev[np.abs(slev - mu) > SIGMA_THRESHOLD * sig] = np.nan
        t_num = (time_vals - pd.Timestamp("2000-01-01")) / pd.Timedelta("1D")
        coef = solve(t_num.values, slev, lat=latitude, nodal=False, trend=False,
                     constit=TIDAL_CONSTITUENTS, method="ols", conf_int="linear")
        tide = reconstruct(t_num.values, coef)
        return slev_vals - tide.h
    except Exception as e:
        print(f"  [UTide] Error: {e}")
        return None


def box_halfwidths(lat_deg: float, r_km: float, margin: float):
    """Box half-widths (degrees) containing a disc of radius r_km at that latitude."""
    dlat = r_km / 111.32 * margin
    coslat = max(np.cos(np.radians(abs(lat_deg))), np.cos(np.radians(80.0)))
    dlon = r_km / (111.32 * coslat) * margin
    return dlat, dlon


# ============================================================================
# TIDE GAUGE PROCESSING
# ============================================================================
def process_tg_detided(pkl_file: Path):
    """Return (daily df[time_d, SLEV_detided], (lat, lon)) or (None, reason)."""
    sid = Path(pkl_file).stem
    try:
        tg = pd.read_pickle(pkl_file).reset_index()
        if "TIME" in tg.columns:
            tg = tg.rename(columns={"TIME": "time_d"})
        tg["time_d"] = pd.to_datetime(tg["time_d"])
        if "DEPTH" in tg.columns:
            tg = tg.drop(columns=["DEPTH"])
        lat = float(tg["LATITUDE"].dropna().iloc[0])
        lon = float(tg["LONGITUDE"].dropna().iloc[0])
        tg = tg.dropna(subset=["SLEV"]).copy()
        if "SLEV_QC" in tg.columns:
            tg = tg[tg["SLEV_QC"].isin([1, 2])]
        tg = tg[(tg["SLEV"] > -15.0) & (tg["SLEV"] < 15.0)]
        if len(tg) < 48:
            return None, f"Strict QC (<48): {sid}"
        tg = tg.set_index("time_d").sort_index()
        if tg.index.to_series().diff().median() < pd.Timedelta("1h"):
            tg = tg.resample("1h").mean(numeric_only=True)
        tg = tg.reset_index().dropna(subset=["SLEV"]).copy()
        detided = utide_detide(tg["time_d"], tg["SLEV"].values, lat)
        if detided is None:
            return None, f"UTide failed: {sid}"
        tg["SLEV_detided"] = detided
        tg = (tg.set_index("time_d").resample("1D").mean(numeric_only=True)
                .dropna(subset=["SLEV_detided"]).reset_index())
        if not check_data_continuity(tg["time_d"]):
            return None, f"Continuity (span/gaps/coverage): {sid}"
        return tg[["time_d", "SLEV_detided"]], (lat, lon)
    except Exception as e:
        return None, f"Error {sid}: {e}"


# ============================================================================
# MULTI-RADIUS ALTIMETRY EXTRACTION (each dataset is read once)
# ============================================================================
def build_target_grid(cfg: "Config", paths: dict):
    """1-D (lat, lon) coordinates of the common target grid, clipped to the domain.
    'coarsest' -> MIOST native grid (~0.125°); 'finest' -> DUACS native grid
    (~0.0625°); float -> regular grid at that step."""
    opt = str(cfg.common_grid).strip().lower()
    if opt in ("coarsest", "coarse", "miost"):
        files = sorted(paths["miost_dir"].glob("dt_global_allsat_phy_l4_*.nc"))
        if not files:
            raise SystemExit("common_grid='coarsest' requires the MIOST files.")
        ds = xr.open_dataset(files[0]); tlat = ds["latitude"].values; tlon = ds["longitude"].values; ds.close()
        label = "MIOST native (~0.125°)"
    elif opt in ("finest", "fine", "duacs"):
        ds = xr.open_dataset(paths["duacs_my"]); tlat = ds["latitude"].values; tlon = ds["longitude"].values; ds.close()
        label = "DUACS native (~0.0625°)"
    else:
        sp = float(cfg.common_grid)
        tlat = np.arange(LAT_MIN, LAT_MAX + 1e-9, sp)
        tlon = np.arange(LON_MIN, LON_MAX + 1e-9, sp)
        label = f"regular {sp}°"
    tlat = np.sort(np.asarray(tlat, float)); tlon = np.sort(np.asarray(tlon, float))
    tlat = tlat[(tlat >= LAT_MIN - 1) & (tlat <= LAT_MAX + 1)]
    tlon = tlon[(tlon >= LON_MIN - 1) & (tlon <= LON_MAX + 1)]
    print(f"  Common target grid: {label}  ({tlat.size}×{tlon.size} cells, method={cfg.regrid_method})")
    return tlat, tlon


def _assign_idx(src_centers, tgt_centers):
    """For each source centre, index of the nearest target centre (by midpoints)."""
    if len(tgt_centers) == 1:
        return np.zeros(len(src_centers), dtype=int)
    edges = (tgt_centers[:-1] + tgt_centers[1:]) / 2.0
    return np.clip(np.searchsorted(edges, src_centers), 0, len(tgt_centers) - 1)


def regrid_box(src_lat, src_lon, cube, tgt_lat, tgt_lon, method="conservative"):
    """Regrid the (time, nlat, nlon) cube from the source grid to the box target grid.
    'conservative' = area mean (nanmean of source cells in each target cell), for
    downsampling (ignores land NaN, no coastal erosion).
    'nearest'/'bilinear' = interpolation (upsampling)."""
    if (src_lat.shape == tgt_lat.shape and src_lon.shape == tgt_lon.shape
            and np.allclose(src_lat, tgt_lat) and np.allclose(src_lon, tgt_lon)):
        return cube                                            # identical grid -> identity
    if method == "conservative":
        ilat = _assign_idx(src_lat, tgt_lat)
        ilon = _assign_idx(src_lon, tgt_lon)
        T, ntl, nto = cube.shape[0], len(tgt_lat), len(tgt_lon)
        valid = np.isfinite(cube)
        src0 = np.where(valid, cube, 0.0)
        sum_lat = np.zeros((T, ntl, cube.shape[2])); cnt_lat = np.zeros_like(sum_lat)
        np.add.at(sum_lat, (slice(None), ilat, slice(None)), src0)
        np.add.at(cnt_lat, (slice(None), ilat, slice(None)), valid.astype(float))
        sum_t = np.zeros((T, ntl, nto)); cnt_t = np.zeros_like(sum_t)
        np.add.at(sum_t, (slice(None), slice(None), ilon), sum_lat)
        np.add.at(cnt_t, (slice(None), slice(None), ilon), cnt_lat)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(cnt_t > 0, sum_t / np.where(cnt_t > 0, cnt_t, 1.0), np.nan)
    # nearest / bilinear (on the small box)
    from scipy.interpolate import RegularGridInterpolator
    kind = "nearest" if method == "nearest" else "linear"
    Tlon, Tlat = np.meshgrid(tgt_lon, tgt_lat)
    pts = np.column_stack([Tlat.ravel(), Tlon.ravel()])
    out = np.empty((cube.shape[0], len(tgt_lat), len(tgt_lon)))
    for t in range(cube.shape[0]):
        f = RegularGridInterpolator((src_lat, src_lon), cube[t], method=kind,
                                    bounds_error=False, fill_value=np.nan)
        out[t] = f(pts).reshape(len(tgt_lat), len(tgt_lon))
    return out


def _target_box(tgt_lat, tgt_lon, lat_tg, lon_tg, r_max, margin):
    """Common-grid cells within the tide gauge box and their haversine distances."""
    dlat, dlon = box_halfwidths(lat_tg, r_max, margin)
    tla = np.where(np.abs(tgt_lat - lat_tg) <= dlat)[0]
    tlo = np.where(np.abs(tgt_lon - lon_tg) <= dlon)[0]
    if tla.size == 0 or tlo.size == 0:
        return None
    blat, blon = tgt_lat[tla], tgt_lon[tlo]
    lon2d, lat2d = np.meshgrid(blon, blat)
    return blat, blon, haversine(lon2d, lat2d, lon_tg, lat_tg)


def _series_from_box(sub_cube, dist2d, radii):
    """sub_cube: (time, nlat, nlon). Return (dict r -> time series, dict r -> n cells)."""
    nt = sub_cube.shape[0]
    flat = sub_cube.reshape(nt, -1)
    dflat = dist2d.ravel()
    series, ncells = {}, {}
    for r in radii:
        m = dflat <= r
        ncells[r] = int(m.sum())
        if m.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                series[r] = np.nanmedian(flat[:, m], axis=1)
        else:
            series[r] = np.full(nt, np.nan)
    return series, ncells


def extract_single_nc_multiradius(nc_path: Path, sla_col: str, tg_coords: dict,
                                  radii: list, margin: float,
                                  tgt_lat=None, tgt_lon=None, method="conservative"):
    """DUACS in a single NetCDF. Each box is regridded to the common grid
    (tgt_lat/tgt_lon) before the per-radius median."""
    if not nc_path.exists():
        print(f"    [WARN] {sla_col} not found: {nc_path}")
        return {}, {}
    ds = xr.open_dataset(nc_path)
    da = ds["sla"]
    if set(("time", "latitude", "longitude")).issubset(da.dims):
        da = da.transpose("time", "latitude", "longitude")
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    times = pd.to_datetime(ds["time"].values).normalize()
    r_max = max(radii)
    out, ncells = {}, {}
    for sid, (lat_tg, lon_tg) in tqdm(tg_coords.items(), desc=f"  {sla_col}", ncols=80):
        tb = _target_box(tgt_lat, tgt_lon, lat_tg, lon_tg, r_max, margin)
        if tb is None:
            continue
        blat, blon, dist = tb
        dlat, dlon = box_halfwidths(lat_tg, r_max, margin)
        la = np.where(np.abs(lat - lat_tg) <= dlat)[0]
        lo = np.where(np.abs(lon - lon_tg) <= dlon)[0]
        if la.size == 0 or lo.size == 0:
            continue
        sub = da.isel(latitude=la, longitude=lo).values          # (time, nla, nlo) source grid
        cube = regrid_box(lat[la], lon[lo], sub, blat, blon, method)   # -> common grid
        series, nc = _series_from_box(cube, dist, radii)
        df = pd.DataFrame({f"{sla_col}|{r}": series[r] for r in radii})
        df["time_d"] = times.values
        out[sid] = df
        ncells[sid] = nc
    ds.close()
    print(f"    -> {len(out)} stations with data {sla_col}")
    return out, ncells


def extract_miost_multiradius(miost_dir: Path, sla_col: str, tg_coords: dict,
                              radii: list, margin: float,
                              tgt_lat=None, tgt_lon=None, method="conservative"):
    """MIOST (0.125°, daily files). If the common grid is the MIOST grid (default),
    the regrid is the identity; otherwise each box is regridded."""
    files = sorted(miost_dir.glob("dt_global_allsat_phy_l4_*.nc"))
    if not files:
        print(f"    [WARN] No {sla_col} files in {miost_dir}")
        return {}, {}
    ds0 = xr.open_dataset(files[0])
    ref_lat = ds0["latitude"].values
    ref_lon = ds0["longitude"].values
    ds0.close()
    r_max = max(radii)
    blocks, src_coords, tboxes, ncells = {}, {}, {}, {}
    for sid, (lat_tg, lon_tg) in tg_coords.items():
        tb = _target_box(tgt_lat, tgt_lon, lat_tg, lon_tg, r_max, margin)
        if tb is None:
            continue
        blat, blon, dist = tb
        dlat, dlon = box_halfwidths(lat_tg, r_max, margin)
        la = np.where(np.abs(ref_lat - lat_tg) <= dlat)[0]
        lo = np.where(np.abs(ref_lon - lon_tg) <= dlon)[0]
        if la.size == 0 or lo.size == 0:
            continue
        la0, la1, lo0, lo1 = la[0], la[-1] + 1, lo[0], lo[-1] + 1
        s_lat, s_lon = ref_lat[la0:la1], ref_lon[lo0:lo1]
        identity = (s_lat.shape == blat.shape and s_lon.shape == blon.shape
                    and np.allclose(s_lat, blat) and np.allclose(s_lon, blon))
        blocks[sid] = (la0, la1, lo0, lo1)
        src_coords[sid] = (s_lat, s_lon)
        tboxes[sid] = (blat, blon, dist, identity)
        ncells[sid] = {r: int((dist.ravel() <= r).sum()) for r in radii}
    recs = {sid: {"time": [], **{r: [] for r in radii}} for sid in blocks}
    for f in tqdm(files, desc=f"  {sla_col} files", ncols=80):
        m = re.search(r"phy_l4_(\d{8})_", f.name)
        if not m:
            continue
        date = pd.Timestamp(m.group(1))
        try:
            ds = xr.open_dataset(f)
            da = ds["sla"].isel(time=0)
            if set(("latitude", "longitude")).issubset(da.dims):
                da = da.transpose("latitude", "longitude")
            full = da.values
            for sid, (la0, la1, lo0, lo1) in blocks.items():
                blk = full[la0:la1, lo0:lo1]
                blat, blon, dist, identity = tboxes[sid]
                if identity:
                    cell_vals = blk
                else:
                    s_lat, s_lon = src_coords[sid]
                    cell_vals = regrid_box(s_lat, s_lon, blk[None, ...], blat, blon, method)[0]
                flat, dflat = cell_vals.ravel(), dist.ravel()
                recs[sid]["time"].append(date)
                for r in radii:
                    mask = dflat <= r
                    if mask.any() and np.any(np.isfinite(flat[mask])):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            recs[sid][r].append(np.nanmedian(flat[mask]))
                    else:
                        recs[sid][r].append(np.nan)
            ds.close()
        except Exception:
            pass
    out = {}
    for sid, d in recs.items():
        if d["time"]:
            df = pd.DataFrame({f"{sla_col}|{r}": d[r] for r in radii})
            df["time_d"] = pd.to_datetime(d["time"])
            out[sid] = df
    print(f"    -> {len(out)} stations with data {sla_col}")
    return out, ncells


def extract_dac_daily(dac_dir: Path, tg_coords: dict, dac_radius: float, margin: float):
    """DAC (6-hourly, 0.25°) at a fixed radius -> daily mean per station."""
    files = sorted(dac_dir.glob("dac_dif_*.nc"))
    if not files:
        print("    [WARN] No DAC files (continuing without DAC correction).")
        return {}
    ds0 = xr.open_dataset(files[0])
    ref_lat = ds0["latitude"].values
    raw_lon = ds0["longitude"].values
    ref_lon = ((raw_lon + 180) % 360) - 180
    sort_idx = np.argsort(ref_lon)
    ref_lon = ref_lon[sort_idx]
    ds0.close()
    blocks, dists = {}, {}
    for sid, (lat_tg, lon_tg) in tg_coords.items():
        dlat, dlon = box_halfwidths(lat_tg, dac_radius, margin)
        la = np.where(np.abs(ref_lat - lat_tg) <= dlat)[0]
        lo = np.where(np.abs(ref_lon - lon_tg) <= dlon)[0]
        if la.size == 0 or lo.size == 0:
            continue
        la0, la1, lo0, lo1 = la[0], la[-1] + 1, lo[0], lo[-1] + 1
        lon2d, lat2d = np.meshgrid(ref_lon[lo0:lo1], ref_lat[la0:la1])
        dist = haversine(lon2d, lat2d, lon_tg, lat_tg)
        blocks[sid] = (la0, la1, lo0, lo1)
        dists[sid] = (dist.ravel() <= dac_radius)
    recs = {sid: [] for sid in blocks}
    for f in tqdm(files, desc="  DAC files", ncols=80):
        parts = f.stem.split("_")
        try:
            jday, hour = int(parts[2]), int(parts[3])
        except Exception:
            continue
        date = DAC_EPOCH + pd.Timedelta(days=jday) + pd.Timedelta(hours=hour)
        try:
            ds = xr.open_dataset(f)
            full = ds["dac"].values[:, sort_idx]
            for sid, (la0, la1, lo0, lo1) in blocks.items():
                blk = full[la0:la1, lo0:lo1].ravel()
                mask = dists[sid]
                val = np.nanmedian(blk[mask]) if (mask.any() and np.any(np.isfinite(blk[mask]))) else np.nan
                recs[sid].append({"datetime": date, "dac": val})
            ds.close()
        except Exception:
            pass
    out = {}
    for sid, r in recs.items():
        if r:
            d = pd.DataFrame(r).set_index("datetime").resample("1D").mean().reset_index()
            d = d.rename(columns={"datetime": "time_d"})
            d["time_d"] = pd.to_datetime(d["time_d"]).dt.normalize()
            out[sid] = d[["time_d", "dac"]]
    print(f"    -> {len(out)} stations with DAC")
    return out


# ============================================================================
# METRICS PER (STATION, RADIUS, PHASE)
# ============================================================================
def metrics_station_radius(tg_df, alt_by_prod, dac_df, phase_dates,
                           triple_intersect=True):
    """alt_by_prod: dict product -> time series (at one radius). Return a metrics dict or None.

    triple_intersect:
      True  -> my_miost joins with an inner join: the three products share exactly
               the same days.
      False -> my_miost joins with a left join and does not trim MY/MIOST-K
               (extended window, where my_miost covers fewer days).
    """
    present = [p for p in PRODUCTS if p in alt_by_prod and alt_by_prod[p] is not None]
    if "my" not in present or "miost" not in present:
        return None  # require at least MY and MIOST
    # assemble altimetry: inner my/miost always; my_miost inner|left by flag
    base = pd.DataFrame({"time_d": alt_by_prod["my"].index,
                         "sla_my": alt_by_prod["my"].values})
    mi = pd.DataFrame({"time_d": alt_by_prod["miost"].index,
                       "sla_miost": alt_by_prod["miost"].values})
    alt = pd.merge(base, mi, on="time_d", how="inner")
    if "my_miost" in present:
        mm = pd.DataFrame({"time_d": alt_by_prod["my_miost"].index,
                           "sla_my_miost": alt_by_prod["my_miost"].values})
        how_mm = "inner" if triple_intersect else "left"
        alt = pd.merge(alt, mm, on="time_d", how=how_mm)
    merged = pd.merge(tg_df[["time_d", "SLEV_detided"]], alt, on="time_d", how="inner")
    if merged.empty:
        return None
    if dac_df is not None and len(dac_df):
        merged = pd.merge(merged, dac_df, on="time_d", how="left")
    if "dac" in merged.columns:
        merged["SLEV_detided_mdac"] = merged["SLEV_detided"] - merged["dac"]
    else:
        merged["SLEV_detided_mdac"] = merged["SLEV_detided"]
    merged["SLEV_detided_mdac"] -= merged["SLEV_detided_mdac"].mean()

    start, end = phase_dates
    mask = merged["time_d"] >= pd.Timestamp(start)
    if end is not None:
        mask &= merged["time_d"] <= pd.Timestamp(end)
    merged = merged.loc[mask].copy()
    if len(merged) < 30:
        return None

    sla_cols = [c for c in ["sla_my", "sla_my_miost", "sla_miost"] if c in merged.columns]
    for c in sla_cols:
        merged[c] -= merged[c].mean()
    jm = pd.Series(True, index=merged.index)
    for c in ["SLEV_detided_mdac"] + sla_cols:
        mu, sig = merged[c].mean(), merged[c].std()
        if sig > 0:
            jm &= ~((merged[c] < mu - SIGMA_THRESHOLD * sig) |
                    (merged[c] > mu + SIGMA_THRESHOLD * sig))
    merged = merged.loc[jm].copy()
    for c in ["SLEV_detided_mdac"] + sla_cols:
        merged[c] -= merged[c].mean()
    if len(merged) < 20:
        return None

    out = {"N": len(merged),
           "tg_std": np.std(merged["SLEV_detided_mdac"].values) * 100,
           "tg_var": np.var(merged["SLEV_detided_mdac"].values) * 10_000}
    rmsd_vals = []
    for p in PRODUCTS:
        col = f"sla_{p}"
        if col in merged.columns:
            rmsd, var, std, corr = calculate_metrics(
                merged["SLEV_detided_mdac"].values, merged[col].values)
            out[f"rmsd_{p}"], out[f"var_{p}"], out[f"std_{p}"], out[f"corr_{p}"] = rmsd, var, std, corr
            if not np.isnan(rmsd):
                rmsd_vals.append(rmsd)
        else:
            out[f"rmsd_{p}"] = out[f"var_{p}"] = out[f"std_{p}"] = out[f"corr_{p}"] = np.nan
    if rmsd_vals and max(rmsd_vals) > RMSD_PLAUSIBILITY_CAP_CM:
        return None  # desalineamiento, no variabilidad real

    merged = merged.sort_values("time_d").copy()
    merged["time_num"] = (merged["time_d"] - merged["time_d"].min()).dt.days.astype(float)
    for c in ["SLEV_detided_mdac"] + sla_cols:
        try:
            merged[f"{c}_loess"] = apply_loess_filter(merged[c], merged["time_num"])
        except Exception:
            merged[f"{c}_loess"] = merged[c]
    out["tg_std_loess"] = np.std(merged["SLEV_detided_mdac_loess"].values) * 100
    out["tg_var_loess"] = np.var(merged["SLEV_detided_mdac_loess"].values) * 10_000
    for p in PRODUCTS:
        col = f"sla_{p}_loess"
        if col in merged.columns:
            rmsd, var, std, corr = calculate_metrics(
                merged["SLEV_detided_mdac_loess"].values, merged[col].values)
            out[f"rmsd_{p}_loess"], out[f"var_{p}_loess"], out[f"std_{p}_loess"], out[f"corr_{p}_loess"] = \
                rmsd, var, std, corr
        else:
            out[f"rmsd_{p}_loess"] = out[f"var_{p}_loess"] = out[f"std_{p}_loess"] = out[f"corr_{p}_loess"] = np.nan
    out["_merged"] = merged   # to export the series at the chosen radius
    return out


# ============================================================================
# AGGREGATION AND OUTPUT
# ============================================================================
def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ph in df["phase"].unique():
        for r in sorted(df["radius"].unique()):
            sub = df[(df["phase"] == ph) & (df["radius"] == r)]
            rec = {"phase": ph, "radius": r, "n_stations_my": int(sub["rmsd_my"].notna().sum())}
            for p in PRODUCTS:
                for base in ["rmsd", "corr"]:
                    for suf in ["", "_loess"]:
                        c = f"{base}_{p}{suf}"
                        if c in sub.columns:
                            v = sub[c].dropna().values
                            rec[f"{c}_median"] = np.nanmedian(v) if len(v) else np.nan
                            rec[f"{c}_mean"] = np.nanmean(v) if len(v) else np.nan
                            rec[f"{c}_q25"] = np.nanpercentile(v, 25) if len(v) else np.nan
                            rec[f"{c}_q75"] = np.nanpercentile(v, 75) if len(v) else np.nan
            rows.append(rec)
    return pd.DataFrame(rows)


def emit_p1_metrics(df_full_rows: list, r_opt: int, outdir: Path):
    """Write overall_metrics_[LOESS_]PARALLEL_{phase}_v7.csv at the optimal radius."""
    sub = [r for r in df_full_rows if r["radius"] == r_opt]
    by_phase = {}
    for r in sub:
        by_phase.setdefault(r["phase"], []).append(r)
    mdir = outdir / "optimal_radius"
    mdir.mkdir(parents=True, exist_ok=True)
    for phase, rows in by_phase.items():
        raw_cols = ["station", "latitude", "longitude", "phase", "N", "tg_std", "tg_var"]
        loe_cols = ["station", "latitude", "longitude", "phase", "N", "tg_std_loess", "tg_var_loess"]
        for p in PRODUCTS:
            raw_cols += [f"rmsd_{p}", f"var_{p}", f"std_{p}", f"corr_{p}"]
            loe_cols += [f"rmsd_{p}_loess", f"var_{p}_loess", f"std_{p}_loess", f"corr_{p}_loess"]
        df_raw = pd.DataFrame(rows)[[c for c in raw_cols if c in rows[0]]]
        df_loe = pd.DataFrame(rows)[[c for c in loe_cols if c in rows[0]]]
        df_raw.to_csv(mdir / f"overall_metrics_PARALLEL_{phase}_v7.csv", index=False, float_format="%.6f")
        df_loe.to_csv(mdir / f"overall_metrics_LOESS_PARALLEL_{phase}_v7.csv", index=False, float_format="%.6f")
    print(f"  overall_metrics_*_v7.csv at radius {r_opt} km -> {mdir}")


def export_timeseries(merged_store: dict, r_target, outdir: Path):
    """Export TS_data_{station}_{phase}.csv at the given radius."""
    base = outdir / "time_series_optimal"
    n = 0
    for (sid, phase, r), merged in merged_store.items():
        if r != r_target:
            continue
        d = base / phase
        d.mkdir(parents=True, exist_ok=True)
        cols = [c for c in merged.columns if c != "time_num"]
        merged[cols].to_csv(d / f"TS_data_{sid}_{phase}.csv", index=False)
        n += 1
    print(f"  {n} time series exported at radius {r_target} km")


# ============================================================================
# ORCHESTRATION
# ============================================================================
def emit_fixed_radius_metrics(rows: list, r_fixed: int, outdir: Path):
    """Write overall_metrics_[LOESS_]PARALLEL_{phase}_v7.csv for all stations at the fixed
    radius, in outdir/optimal_radius_fixed{R}."""
    mdir = outdir / f"optimal_radius_fixed{r_fixed}"
    emit_p1_metrics(rows, r_fixed, outdir)
    # emit_p1_metrics() writes to outdir/"optimal_radius": rename it
    src = outdir / "optimal_radius"
    if src.exists() and src != mdir:
        mdir.parent.mkdir(parents=True, exist_ok=True)
        if mdir.exists():
            import shutil
            shutil.rmtree(mdir)
        src.rename(mdir)
    print(f"  overall_metrics_*_v7.csv (fixed radius {r_fixed} km) -> {mdir}")


def export_timeseries_fixed(merged_store: dict, r_fixed: int, outdir: Path):
    export_timeseries(merged_store, r_fixed, outdir / f"optimal_radius_fixed{r_fixed}")


def _finalize(df: pd.DataFrame, rows: list, merged_store: dict, cfg: Config):
    """Aggregate the sweep and write the fixed-radius metrics."""
    agg = aggregate(df)
    agg.to_csv(cfg.outdir / "sensitivity_aggregate.csv", index=False, float_format="%.6f")
    print("  sensitivity_aggregate.csv")
    if cfg.fixed_radius_km not in cfg.radii:
        raise SystemExit(f"fixed_radius_km={cfg.fixed_radius_km} is not in the radius sweep.")
    if cfg.emit_p1_metrics:
        emit_fixed_radius_metrics(rows, cfg.fixed_radius_km, cfg.outdir)
    if cfg.export_ts != "none":
        export_timeseries_fixed(merged_store, cfg.fixed_radius_km, cfg.outdir)


def run(cfg: Config):
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    paths = cfg.paths
    print("=" * 70)
    print(f"  Radius sensitivity: {cfg.radii[0]}-{cfg.radii[-1]} km (step "
          f"{cfg.radii[1]-cfg.radii[0] if len(cfg.radii) > 1 else '-'})")
    print(f"  base_dir = {cfg.base_dir}")
    print("=" * 70)

    if cfg.from_csv is not None:
        print(f"\n[resume] Loading metrics from {cfg.from_csv} (no re-extraction)...")
        df = pd.read_csv(cfg.from_csv)
        if cfg.export_ts != "none":
            print("  (--export-ts requires a full run; skipped with --from-csv)")
            cfg.export_ts = "none"
        return _finalize(df, df.to_dict("records"), {}, cfg)

    # 1) Stations
    pkls = sorted(paths["tg_dir"].glob("*.pkl"))
    print(f"\n[1] Tide gauges: {len(pkls)} pickles")
    tg_coords, tg_detided = {}, {}
    failed = []
    for pk in tqdm(pkls, desc="  TG QC+detide", ncols=80):
        try:
            df = pd.read_pickle(pk)
            lat = float(df["LATITUDE"].dropna().iloc[0])
            lon = float(df["LONGITUDE"].dropna().iloc[0])
        except Exception:
            continue
        if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
            continue
        res, info = process_tg_detided(pk)
        if res is None:
            failed.append(info)
            continue
        sid = pk.stem
        tg_detided[sid] = res
        tg_coords[sid] = info  # (lat, lon)
    print(f"    valid stations: {len(tg_coords)}  (discarded: {len(failed)})")
    if not tg_coords:
        raise SystemExit("No valid stations; check --base-dir and the data.")

    # 2) DAC (fixed radius)
    print(f"\n[2] DAC (fixed radius {cfg.dac_radius:.0f} km)...")
    dac = extract_dac_daily(paths["dac_dir"], tg_coords, cfg.dac_radius, cfg.box_margin)

    # 3) Multi-radius altimetry (all on the common grid)
    print("\n[3] Multi-radius altimetry (regridding to the common grid)...")
    tgt_lat, tgt_lon = build_target_grid(cfg, paths)
    alt_my, nc_my = extract_single_nc_multiradius(paths["duacs_my"], "sla_my", tg_coords, cfg.radii, cfg.box_margin, tgt_lat, tgt_lon, cfg.regrid_method)
    alt_mm, nc_mm = extract_single_nc_multiradius(paths["duacs_my_miost"], "sla_my_miost", tg_coords, cfg.radii, cfg.box_margin, tgt_lat, tgt_lon, cfg.regrid_method)
    alt_mi, nc_mi = extract_miost_multiradius(paths["miost_dir"], "sla_miost", tg_coords, cfg.radii, cfg.box_margin, tgt_lat, tgt_lon, cfg.regrid_method)

    # 4) Metrics per (station, phase, radius)
    print("\n[4] Metrics per station/phase/radius...")
    rows, merged_store = [], {}
    for sid in tqdm(tg_coords, desc="  metrics", ncols=80):
        lat, lon = tg_coords[sid]
        dac_df = dac.get(sid)
        for phase, pd_dates in cfg.phases.items():
            for r in cfg.radii:
                alt_by_prod = {}
                for prod, store, ncs in [("my", alt_my, nc_my),
                                         ("my_miost", alt_mm, nc_mm),
                                         ("miost", alt_mi, nc_mi)]:
                    if sid in store and ncs.get(sid, {}).get(r, 0) > 0:
                        col = f"{PRODUCT_SLA_COL[prod]}|{r}"
                        s = store[sid][["time_d", col]].dropna()
                        if len(s):
                            alt_by_prod[prod] = pd.Series(s[col].values,
                                                          index=pd.to_datetime(s["time_d"]).dt.normalize())
                m = metrics_station_radius(tg_detided[sid], alt_by_prod, dac_df, pd_dates,
                                           triple_intersect=cfg.triple_intersect)
                if m is None:
                    continue
                merged_store[(sid, phase, r)] = m.pop("_merged")
                row = {"station": sid, "latitude": lat, "longitude": lon,
                       "phase": phase, "radius": r,
                       "ncells_my": nc_my.get(sid, {}).get(r, 0),
                       "ncells_my_miost": nc_mm.get(sid, {}).get(r, 0),
                       "ncells_miost": nc_mi.get(sid, {}).get(r, 0)}
                row.update(m)
                rows.append(row)
    if not rows:
        raise SystemExit("No metrics were produced; check data, phases and radii.")
    df = pd.DataFrame(rows)
    keep = [c for c in df.columns if c != "_merged"]
    df[keep].to_csv(cfg.outdir / "sensitivity_per_station_radius.csv", index=False, float_format="%.6f")
    print(f"\n  sensitivity_per_station_radius.csv ({len(df)} rows)")

    return _finalize(df, rows, merged_store, cfg)


def build_cfg_from_args(argv=None) -> Config:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-dir", type=Path, default=BASE_DIR_DEFAULT)
    ap.add_argument("--outdir", type=Path, default=Path("results_radius_sensitivity"))
    ap.add_argument("--rmin", type=int, default=5)
    ap.add_argument("--rmax", type=int, default=400)
    ap.add_argument("--rstep", type=int, default=5)
    ap.add_argument("--dac-radius", type=float, default=DAC_RADIUS_DEFAULT)
    ap.add_argument("--box-margin", type=float, default=1.25)
    ap.add_argument("--common-grid", default="coarsest",
                    help="Common grid: 'coarsest' (0.125 deg), 'finest' or step in degrees")
    ap.add_argument("--regrid-method", default="conservative",
                    choices=["conservative", "bilinear", "nearest"])
    ap.add_argument("--export-ts", default="optimal", choices=["none", "optimal", "all"])
    ap.add_argument("--no-p1-metrics", action="store_true")
    ap.add_argument("--from-csv", type=Path, default=None,
                    help="Resume from sensitivity_per_station_radius.csv (no re-extraction)")
    ap.add_argument("--fixed-radius-km", type=int, default=50,
                    help="Extraction radius (km) of the output metrics")
    ap.add_argument("--science-end", type=str, default=None,
                    help="End of the Science phase (YYYY-MM-DD); 2025-07-06 excludes the NRT segment")
    ap.add_argument("--operational-window", action="store_true",
                    help="Do not intersect my_miost with MY and MIOST-K (extended window)")
    a = ap.parse_args(argv)
    return Config(
        base_dir=a.base_dir, outdir=a.outdir,
        radii=list(range(a.rmin, a.rmax + 1, a.rstep)),
        dac_radius=a.dac_radius, box_margin=a.box_margin,
        common_grid=a.common_grid, regrid_method=a.regrid_method,
        export_ts=a.export_ts, emit_p1_metrics=not a.no_p1_metrics,
        from_csv=a.from_csv, fixed_radius_km=a.fixed_radius_km,
        science_end=a.science_end, triple_intersect=not a.operational_window,
    )


def main(argv=None):
    run(build_cfg_from_args(argv))


if __name__ == "__main__":
    main()