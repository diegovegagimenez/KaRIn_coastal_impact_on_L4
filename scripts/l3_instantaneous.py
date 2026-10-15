#!/usr/bin/env python3
"""Instantaneous L3 KaRIn vs tide gauge comparison (no daily mean, no LOESS),
CalVal, near-swath stations of the fixed panel.

- Tide gauge: hourly detided series (UTide), linearly interpolated to the pass
  time, minus the DAC of the L3 product (median of the valid pixels in the disc).
- L3: median of ssha_filtered with quality_flag == 0 within R km, per pass.
- L4 (MIOST-K, MY): daily value of the pass day (cached series).
- Daily reference: same passes, against the daily-mean tide gauge
  (daily detided tide gauge - daily DAC; tg_dac_cache.pkl).
- Joint 3-sigma filter and re-centring; RMSD and correlation per station.

Output: results/l3_near_swath_scale/l3_instantaneous_{samples,per_station,summary}.csv
"""
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

ART = Path(os.environ.get("KARIN_PROJECT_DIR", "data"))
HERE = Path(__file__).resolve().parent
D = HERE.parent / "results" / "l3_near_swath_scale"
sys.path.insert(0, str(ART / "PIPELINE_FINAL_LIMPIO"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import P1_altimetry_tidegauge_processing as P1  # noqa: E402

RADII = [10, 20, 50]
RE = 6371.0
T0, T1 = pd.Timestamp("2023-03-28"), pd.Timestamp("2023-07-06")

st = pd.read_csv(D / "near_panel_stations.csv").iloc[:, :3]
st.columns = ["station", "latitude", "longitude"]
reg = pd.read_csv(HERE / "per_station_radius_with_pixels.csv").drop_duplicates("station").set_index("station").region


def xyz(lat, lon):
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


def hourly_detided(sid):
    """Instantaneous L3 KaRIn vs tide gauge comparison (no daily mean, no LOESS),
CalVal, near-swath stations of the fixed panel.

- Tide gauge: hourly detided series (UTide), linearly interpolated to the pass
  time, minus the DAC of the L3 product (median of the valid pixels in the disc).
- L3: median of ssha_filtered with quality_flag == 0 within R km, per pass.
- L4 (MIOST-K, MY): daily value of the pass day (cached series).
- Daily reference: same passes, against the daily-mean tide gauge.
- Joint 3-sigma filter and re-centring; RMSD and correlation per station."""
    tg = pd.read_pickle(ART / "data/processed_NRT/tide_gauges_nrt" / f"{sid}.pkl").reset_index()
    if "TIME" in tg.columns:
        tg = tg.rename(columns={"TIME": "time_d"})
    tg["time_d"] = pd.to_datetime(tg["time_d"])
    lat = float(tg["LATITUDE"].dropna().iloc[0])
    tg = tg.dropna(subset=["SLEV"]).copy()
    if "SLEV_QC" in tg.columns:
        tg = tg[tg["SLEV_QC"].isin([1, 2])]
    tg = tg[(tg["SLEV"] > -15.0) & (tg["SLEV"] < 15.0)]
    tg = tg.set_index("time_d").sort_index()
    if tg.index.to_series().diff().median() < pd.Timedelta("1h"):
        tg = tg.resample("1h").mean(numeric_only=True)
    tg = tg.reset_index().dropna(subset=["SLEV"]).copy()
    det = P1.utide_detide(tg["time_d"], tg["SLEV"].values, lat)
    if det is None:
        return None
    s = pd.Series(np.asarray(det), index=pd.DatetimeIndex(tg["time_d"]))
    return s[(s.index >= T0 - pd.Timedelta("2D")) & (s.index <= T1 + pd.Timedelta("2D"))]


def build_tg():
    c = D / "tg_hourly_detided_cache.pkl"
    if c.exists():
        return pickle.load(open(c, "rb"))
    out = {}
    for i, sid in enumerate(st.station):
        try:
            out[sid] = hourly_detided(sid)
        except Exception as e:
            print("  TG fail", sid, e)
            out[sid] = None
        print(f"  TG {i + 1}/{len(st)} {sid}", flush=True)
    pickle.dump(out, open(c, "wb"))
    return out


def build_l3():
    c = D / "l3_instantaneous_cache.pkl"
    if c.exists():
        return pd.read_pickle(c)
    P = xyz(st.latitude.values, st.longitude.values)
    rows = []
    files = sorted((ART / "data/raw/swot_l3_expert").glob("SWOT_L3_LR_SSH_Expert_*.nc"))
    for i, f in enumerate(files):
        ds = xr.open_dataset(f)
        lat = ds.latitude.values
        lon = ((ds.longitude.values + 180) % 360) - 180
        ssha = ds.ssha_filtered.values
        dac = ds.dac.values
        ok = np.isfinite(ssha) & (ds.quality_flag.values == 0)
        tt = np.broadcast_to(ds.time.values.astype("int64")[:, None], lat.shape)
        ds.close()
        if not ok.any():
            continue
        la, lo, sv, dv, tv = lat[ok], lon[ok], ssha[ok], dac[ok], tt[ok]
        tree = cKDTree(xyz(la, lo))
        for r in RADII:
            idx = tree.query_ball_point(P, 2 * np.sin(r / RE / 2))
            for k, ii in enumerate(idx):
                if len(ii) == 0:
                    continue
                rows.append(dict(station=st.station[k], radius=r, npix=len(ii),
                                 time=pd.Timestamp(int(np.median(tv[ii]))),
                                 l3=float(np.median(sv[ii])), dac=float(np.nanmedian(dv[ii]))))
        if (i + 1) % 100 == 0:
            print(f"  L3 {i + 1}/{len(files)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_pickle(c)
    return df


def main():
    l3 = build_l3()
    tg = build_tg()
    L = lambda n: pickle.load(open(D / n, "rb"))
    l4, du, tgd = L("l4_cache.pkl"), L("duacs_cache.pkl"), L("tg_dac_cache.pkl")
    samples, rec = [], []
    for (sid, r), g in l3.groupby(["station", "radius"]):
        h = tg.get(sid)
        if h is None or sid not in l4 or sid not in du or sid not in tgd:
            continue
        g = g.sort_values("time").copy()
        # hourly tide gauge interpolated to the pass time (only if data within 1 h on each side)
        tnum = h.index.values.astype("int64")
        x = g.time.values.astype("int64")
        j = np.searchsorted(tnum, x)
        okb = (j > 0) & (j < len(tnum))
        gap = np.full(len(x), np.inf)
        gap[okb] = (tnum[j[okb]] - tnum[j[okb] - 1]) / 3.6e12
        g["tg_inst"] = np.where(gap <= 2.0, np.interp(x, tnum, h.values), np.nan) - g.dac.values
        g["date"] = g.time.dt.normalize()
        a = l4[sid][["time", f"SLA_L4_R{r}"]].rename(columns={"time": "date", f"SLA_L4_R{r}": "mk"})
        b = du[sid][["time", f"SLA_DUACS_R{r}"]].rename(columns={"time": "date", f"SLA_DUACS_R{r}": "my"})
        c = tgd[sid].rename(columns={"time": "date", "SLEV_detided_mdac": "tg_day"})
        g = g.merge(a, on="date").merge(b, on="date").merge(c, on="date")
        cols = ["tg_inst", "tg_day", "l3", "mk", "my"]
        g = g.dropna(subset=cols)
        if len(g) < 20:
            continue
        m = np.ones(len(g), bool)
        for col in cols:
            m &= (np.abs(g[col] - g[col].mean()) <= 3 * g[col].std()).values
        g = g[m].copy()
        if len(g) < 20:
            continue
        for col in cols:
            g[col] = (g[col] - g[col].mean()) * 100.0
        g["station"], g["radius"] = sid, r
        samples.append(g)
        out = dict(station=sid, region=reg.get(sid), radius=r, n=len(g), npix=g.npix.median(),
                   tg_std_inst=g.tg_inst.std(), tg_std_day=g.tg_day.std(),
                   inst_minus_day_std=(g.tg_inst - g.tg_day).std())
        for ref in ("inst", "day"):
            y = g[f"tg_{ref}"].values
            for p in ("l3", "mk", "my"):
                out[f"rmsd_{p}_{ref}"] = float(np.sqrt(np.mean((y - g[p].values) ** 2)))
                out[f"corr_{p}_{ref}"] = float(np.corrcoef(y, g[p].values)[0, 1])
        rec.append(out)
    S = pd.concat(samples)
    S.to_csv(D / "l3_instantaneous_samples.csv", index=False, float_format="%.4f")
    R = pd.DataFrame(rec)
    R = R[(R.filter(like="rmsd_") < 50).all(axis=1)]
    R.to_csv(D / "l3_instantaneous_per_station.csv", index=False, float_format="%.4f")

    pd.set_option("display.width", 250)

    def summ(x):
        o = dict(n_st=len(x), n_obs=x.n.median(), npix=x.npix.median(), tg_std=x.tg_std_inst.median(),
                 d_inst_day=x.inst_minus_day_std.median())
        for ref in ("inst", "day"):
            for p in ("l3", "mk", "my"):
                o[f"rmsd_{p}_{ref}"] = x[f"rmsd_{p}_{ref}"].median()
            for p in ("l3", "mk", "my"):
                o[f"r_{p}_{ref}"] = x[f"corr_{p}_{ref}"].median()
            o[f"pct_l3<my_{ref}"] = 100 * (x[f"rmsd_l3_{ref}"] < x[f"rmsd_my_{ref}"]).mean()
            o[f"pct_l3<mk_{ref}"] = 100 * (x[f"rmsd_l3_{ref}"] < x[f"rmsd_mk_{ref}"]).mean()
            o[f"pct_mk<my_{ref}"] = 100 * (x[f"rmsd_mk_{ref}"] < x[f"rmsd_my_{ref}"]).mean()
        return pd.Series(o)

    allr = R.assign(region="ALL")
    T = pd.concat([R, allr]).groupby(["radius", "region"]).apply(summ).round(2)
    T.to_csv(D / "l3_instantaneous_summary.csv")
    print(T.T.to_string())
    b = R[(R.region == "Baltic Sea") & (R.radius == 50)]
    print("\nBaltic stations, R=50 (instantaneous):")
    print(b[["station", "n", "npix", "tg_std_inst", "rmsd_l3_inst", "rmsd_mk_inst", "rmsd_my_inst",
             "corr_l3_inst", "corr_mk_inst", "corr_my_inst"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
