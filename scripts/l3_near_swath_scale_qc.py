#!/usr/bin/env python3
"""L3 SWOT KaRIn vs MIOST-K vs DUACS-MY vs tide gauges -- CalVal, near-swath panel,
radius sweep 10-100 km -- version con el L3 POR DEFECTO:

  - variable `ssha_filtered` (calibrada, editada y con reduccion de ruido U-Net),
  - enmascarando todo pixel con quality_flag != 0 (solo "good").

Sustituye a la extraccion L3 de l3_near_swath_scale.py (que usaba `ssha_unedited`
y solo excluia los flags 100-102). TG+DAC, MIOST-K y DUACS-MY se reutilizan de
las caches de ese script (no dependen del L3). Misma cadena de procesado:
mediana espacial por pasada dentro de R, merge de fechas comunes, 3-sigma
conjunto, recentrado, LOESS 15 d (low-pass) y residuo high-pass.

Salidas en results/l3_near_swath_scale/ con sufijo _qc:
  l3_cache_qc.pkl, per_station_radius_3products_qc{,_raw,_hp}.csv,
  summary_by_radius_3products_qc{,_raw,_hp}.csv, l3_pixel_counts_qc.csv
"""
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from statsmodels.nonparametric.smoothers_lowess import lowess

ART = Path(os.environ.get("KARIN_PROJECT_DIR", "data"))
D = Path(__file__).resolve().parent.parent / "results" / "l3_near_swath_scale"
RADII = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
RMSD_CAP = 50.0
LOESS_DAYS = 15.0


def hav(lon1, lat1, lon2, lat2):
    lon1, lon2, lat1, lat2 = map(np.deg2rad, [lon1, lon2, lat1, lat2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def extract_l3(st):
    files = sorted((ART / "data/raw/swot_l3_expert").glob("SWOT_L3_LR_SSH_Expert_*.nc"))
    print(f"[L3] {len(files)} ficheros; ssha_filtered + quality_flag == 0")
    rows = {s: [] for s in st.station}
    npix = []
    for i, f in enumerate(files):
        ds = xr.open_dataset(f)
        lat = ds.latitude.values
        lon = ((ds.longitude.values + 180.0) % 360.0) - 180.0
        ssha = ds.ssha_filtered.values
        ssha = np.where(ds.quality_flag.values == 0, ssha, np.nan)
        tvals = ds.time.values
        ds.close()
        for sid, la, lo in st.itertuples(index=False):
            bb = (np.abs(lat - la) < 1.0) & (np.abs(lon - lo) < 1.0 / np.cos(np.deg2rad(la)))
            if not bb.any():
                continue
            dist = np.full_like(lat, np.inf)
            dist[bb] = hav(lon[bb], lat[bb], lo, la)
            lines = np.where((dist <= max(RADII)).any(axis=1))[0]
            if len(lines) == 0:
                continue
            row = {"time": pd.Timestamp(np.median(tvals[lines].astype("int64")))}
            cnt = {"station": sid, "file": f.name}
            ok = False
            for r in RADII:
                v = ssha[dist <= r]
                v = v[np.isfinite(v)]
                cnt[f"n_R{r}"] = len(v)
                row[f"SLA_L3_R{r}"] = float(np.median(v)) if len(v) else np.nan
                ok |= len(v) > 0
            npix.append(cnt)
            if ok:
                rows[sid].append(row)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(files)}")
    out = {}
    for sid, recs in rows.items():
        if not recs:
            out[sid] = pd.DataFrame(columns=["time"] + [f"SLA_L3_R{r}" for r in RADII])
            continue
        df = pd.DataFrame(recs)
        df["time"] = pd.to_datetime(df.time).dt.normalize()
        out[sid] = df.groupby("time", as_index=False).median(numeric_only=True)
    pd.DataFrame(npix).to_csv(D / "l3_pixel_counts_qc.csv", index=False)
    return out


def prep(df, cols, mode):
    """mode: 'lp' (LOESS), 'raw' (sin filtro temporal), 'hp' (serie - LOESS)."""
    d = df.dropna(subset=cols).copy()
    if len(d) < 8:
        return None
    m = pd.Series(True, index=d.index)
    for c in cols:
        m &= (d[c] - d[c].mean()).abs() <= 3.0 * d[c].std()
    d = d[m].copy()
    if len(d) < 8:
        return None
    for c in cols:
        d[c] -= d[c].mean()
    if mode == "raw":
        return d
    t = (d.time - d.time.min()).dt.total_seconds() / 86400.0
    frac = min(LOESS_DAYS / (t.max() - t.min()), 1.0)
    for c in cols:
        lo = lowess(d[c].values, t.values, frac=frac, return_sorted=False)
        d[c] = lo if mode == "lp" else d[c].values - lo
    return d


def metrics(st, l3, tg, l4, du, mode):
    rec = []
    for sid in st.station:
        if any(sid not in x or x[sid].empty for x in (l3, tg, l4, du)):
            continue
        for r in RADII:
            c3, c4, cd = f"SLA_L3_R{r}", f"SLA_L4_R{r}", f"SLA_DUACS_R{r}"
            if c4 not in l4[sid] or cd not in du[sid]:
                continue
            m = (tg[sid].merge(l3[sid][["time", c3]], on="time")
                 .merge(l4[sid][["time", c4]], on="time").merge(du[sid][["time", cd]], on="time"))
            cols = ["SLEV_detided_mdac", c3, c4, cd]
            d = prep(m, cols, mode)
            if d is None:
                continue
            g, a, b, c = (d[x].values * 100.0 for x in cols)
            rm = lambda x: float(np.sqrt(np.mean((g - x) ** 2)))
            cc = lambda x: float(np.corrcoef(g, x)[0, 1])
            r3, r4, rd = rm(a), rm(b), rm(c)
            # tope de RMSD del articulo; se descartan tambien series degeneradas (RMSD nulo)
            if max(r3, r4, rd) > RMSD_CAP or min(r3, r4, rd) == 0:
                continue
            rec.append(dict(station=sid, radius=r, n=len(d), rmsd_l3=r3, rmsd_l4=r4, rmsd_duacs=rd,
                            corr_l3=cc(a), corr_l4=cc(b), corr_duacs=cc(c),
                            pi_l3_vs_duacs=100 * (rd - r3) / rd, pi_l4_vs_duacs=100 * (rd - r4) / rd,
                            pi_l3_vs_l4=100 * (r4 - r3) / r4))
    return pd.DataFrame(rec)


def summarize(df):
    g = df.groupby("radius")
    s = g.agg(n=("station", "size"), ndays=("n", "median"),
              median_rmsd_l3=("rmsd_l3", "median"), median_rmsd_l4=("rmsd_l4", "median"),
              median_rmsd_duacs=("rmsd_duacs", "median"),
              median_corr_l3=("corr_l3", "median"), median_corr_l4=("corr_l4", "median"),
              median_corr_duacs=("corr_duacs", "median"),
              median_pi_l4_vs_duacs=("pi_l4_vs_duacs", "median"),
              median_pi_l3_vs_duacs=("pi_l3_vs_duacs", "median"),
              median_pi_l3_vs_l4=("pi_l3_vs_l4", "median"))
    s["pct_l4_better_than_duacs"] = g.apply(lambda x: 100 * (x.rmsd_l4 < x.rmsd_duacs).mean())
    s["pct_l3_better_than_duacs"] = g.apply(lambda x: 100 * (x.rmsd_l3 < x.rmsd_duacs).mean())
    s["pct_l3_corr_better_than_duacs"] = g.apply(lambda x: 100 * (x.corr_l3 > x.corr_duacs).mean())
    return s


def main():
    st = pd.read_csv(D / "near_panel_stations.csv").iloc[:, :3]
    st.columns = ["station", "latitude", "longitude"]
    cache = D / "l3_cache_qc.pkl"
    if cache.exists():
        l3 = pickle.load(open(cache, "rb"))
    else:
        l3 = extract_l3(st)
        pickle.dump(l3, open(cache, "wb"))
    tg = pickle.load(open(D / "tg_dac_cache.pkl", "rb"))
    l4 = pickle.load(open(D / "l4_cache.pkl", "rb"))
    du = pickle.load(open(D / "duacs_cache.pkl", "rb"))
    pd.set_option("display.width", 250)
    for mode, suf in (("lp", ""), ("raw", "_raw"), ("hp", "_hp")):
        df = metrics(st, l3, tg, l4, du, mode)
        df.to_csv(D / f"per_station_radius_3products_qc{suf}.csv", index=False, float_format="%.5f")
        s = summarize(df)
        s.to_csv(D / f"summary_by_radius_3products_qc{suf}.csv", float_format="%.4f")
        print(f"\n===== {mode.upper()} (L3 ssha_filtered, QF==0) =====")
        print(s.round(3).to_string())
    n = pd.read_csv(D / "l3_pixel_counts_qc.csv")
    v = n[[c for c in n if c.startswith("n_R")]].replace(0, np.nan)
    print("\nMedian valid L3 pixels per overpass (QF==0):")
    print(v.median().round(0).to_string())


if __name__ == "__main__":
    main()
