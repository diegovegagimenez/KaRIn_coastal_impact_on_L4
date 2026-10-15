#!/usr/bin/env python3
"""Fig. 9: L3 KaRIn over (a) MY and (b) MIOST-K of the same day in the transition
zone, with the stations of the zone.

Also computes, for all CalVal overpasses of pass 003 over the zone, the RMS
difference and correlation between L3 and each L4 field (interpolated to the L3
pixels, means removed).
Output: figures/figO_l3_over_l4_transition_zone.{png,pdf}"""
import os
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature

BASE = Path(os.environ.get("KARIN_PROJECT_DIR", "data"))
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "figures"
RES = HERE.parent / "results" / "l3_near_swath_scale"
DATE = sys.argv[1] if len(sys.argv) > 1 else "20230429"
EXT = (9.0, 13.6, 53.8, 56.3)          # lon0, lon1, lat0, lat1
PAD = 0.3
CMAP = "RdBu_r"
FS = 12
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": FS,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})

MY = xr.open_dataset(BASE / "data/raw/duacs_l4_my/duacs_sla_eur_MY_0625deg_swot_period.nc").sla


def grid_my(date):
    return (MY.sel(time=pd.Timestamp(date), method="nearest")
            .sel(latitude=slice(EXT[2] - PAD, EXT[3] + PAD), longitude=slice(EXT[0] - PAD, EXT[1] + PAD)) * 100).load()


def grid_mk(date):
    f = glob.glob(str(BASE / f"data/raw/miost_l4/dt_global_allsat_phy_l4_{date}_*.nc"))
    if not f:
        return None
    ds = xr.open_dataset(f[0])
    g = (ds.sla.isel(time=0).sel(latitude=slice(EXT[2] - PAD, EXT[3] + PAD),
                                  longitude=slice(EXT[0] - PAD, EXT[1] + PAD)) * 100).load()
    ds.close()
    return g


def load_l3(f):
    ds = xr.open_dataset(f)
    lat = ds.latitude.values
    lon = ((ds.longitude.values + 180) % 360) - 180
    s = np.where(ds.quality_flag.values == 0, ds.ssha_filtered.values, np.nan) * 100
    ds.close()
    box = ((lat >= EXT[2] - PAD) & (lat <= EXT[3] + PAD) & (lon >= EXT[0] - PAD) & (lon <= EXT[1] + PAD))
    idx = np.where(box.any(axis=1))[0]
    return lon[idx], lat[idx], s[idx]


def compare(lon, lat, s, g):
    m = (lat >= EXT[2]) & (lat <= EXT[3]) & (lon >= EXT[0]) & (lon <= EXT[1]) & np.isfinite(s)
    if m.sum() < 200:
        return None
    v = g.interp(latitude=xr.DataArray(lat[m], dims="k"), longitude=xr.DataArray(lon[m], dims="k")).values
    a = s[m]
    ok = np.isfinite(v)
    a, v = a[ok], v[ok]
    if len(a) < 200:
        return None
    d = (a - a.mean()) - (v - v.mean())
    return dict(n=len(a), rms=float(np.sqrt(np.mean(d ** 2))), r=float(np.corrcoef(a, v)[0, 1]),
                std_l3=float(a.std()), std_l4=float(v.std()))


def main():
    files = sorted(glob.glob(str(BASE / "data/raw/swot_l3_expert/SWOT_L3_LR_SSH_Expert_*_003_*.nc")))
    rows = []
    for f in files:
        d = re.search(r"_(\d{8})T", f).group(1)
        lon, lat, s = load_l3(f)
        mk = grid_mk(d)
        if mk is None:
            continue
        a, b = compare(lon, lat, s, grid_my(d)), compare(lon, lat, s, mk)
        if a and b:
            rows.append(dict(date=d, n_my=a["n"], rms_my=a["rms"], r_my=a["r"], n_mk=b["n"], rms_mk=b["rms"],
                             r_mk=b["r"], std_l3=a["std_l3"], std_my=a["std_l4"], std_mk=b["std_l4"]))
    R = pd.DataFrame(rows)
    R.to_csv(RES / "l3_vs_l4_pass003_transition_zone.csv", index=False, float_format="%.3f")
    print(f"overpasses: {len(R)}")
    print(R[["rms_my", "rms_mk", "r_my", "r_mk", "std_l3", "std_my", "std_mk", "n_my"]].describe()
          .loc[["50%", "25%", "75%"]].round(2).to_string())
    print("MIOST-K closer to L3 than MY (RMS): %.0f%% of overpasses" % (100 * (R.rms_mk < R.rms_my).mean()))
    sel = R[R.date == DATE].iloc[0]
    print("selected", DATE, sel.drop("date").astype(float).round(2).to_dict())
    print("percentile of std_l3 on selected day: %.0f" % (100 * (R.std_l3 < sel.std_l3).mean()))

    f = [x for x in files if f"_{DATE}T" in x][0]
    lon, lat, s = load_l3(f)
    my, mk = grid_my(DATE), grid_mk(DATE)
    st = pd.read_csv(HERE / "station_regions_v2.csv")
    st = st[st.region_v2 == "Transition zone"]
    combo = np.concatenate([my.values[np.isfinite(my.values)], mk.values[np.isfinite(mk.values)], s[np.isfinite(s)]])
    vlim = float(np.nanpercentile(np.abs(combo - np.nanmedian(combo)), 98))
    c0 = float(np.nanmedian(s[np.isfinite(s)]))
    proj = ccrs.Mercator()
    fig, axes = plt.subplots(1, 2, figsize=(17, 8.2), subplot_kw=dict(projection=proj))
    for ax, lab, g, key in zip(axes, ("(a) SWOT KaRIn L3 over DUACS-MY", "(b) SWOT KaRIn L3 over MIOST-K"),
                               (my, mk), ("my", "mk")):
        ax.set_extent(EXT, crs=ccrs.PlateCarree())
        lo2, la2 = np.meshgrid(g.longitude.values, g.latitude.values)
        # the scene median of each field is subtracted to compare structure, not mean level
        pc = ax.pcolormesh(lo2, la2, g.values - np.nanmedian(g.values), cmap=CMAP, vmin=-vlim, vmax=vlim,
                           shading="auto", transform=ccrs.PlateCarree(), zorder=2, rasterized=True)
        ax.pcolormesh(lon, lat, s - c0, cmap=CMAP, vmin=-vlim, vmax=vlim, shading="nearest",
                      transform=ccrs.PlateCarree(), zorder=5, rasterized=True)
        ax.contour(lon, lat, np.isfinite(s).astype(float), levels=[0.5], colors="black", linewidths=0.8,
                   transform=ccrs.PlateCarree(), zorder=6)
        ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#3a3a3a", zorder=7)
        ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.6, zorder=8)
        ax.scatter(st.longitude, st.latitude, s=42, marker="^", facecolor="yellow", edgecolor="k", linewidth=0.7,
                   transform=ccrs.PlateCarree(), zorder=9, label="Tide gauges (transition zone)")
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.4, color="gray", linestyle="--", zorder=9)
        gl.top_labels = gl.right_labels = False
        gl.xlocator = mticker.FixedLocator(np.arange(9, 14.5, 1.0))
        gl.ylocator = mticker.FixedLocator(np.arange(54, 56.6, 0.5))
        ax.set_title(lab, fontweight="bold", fontsize=FS + 1)
        ax.text(0.02, 0.02, f"{DATE[:4]}-{DATE[4:6]}-{DATE[6:]}, pass 003\nL3 pixels: {int(sel['n_' + key])}\n"
                f"RMS(L3$-$L4) = {sel['rms_' + key]:.2f} cm\n$r$ = {sel['r_' + key]:.2f}",
                transform=ax.transAxes, fontsize=FS - 2, va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.4", alpha=0.9), zorder=10)
    axes[0].legend(loc="upper left", fontsize=FS - 2, framealpha=0.9).set_zorder(11)
    cb = fig.colorbar(pc, ax=axes, orientation="vertical", fraction=0.025, pad=0.02, extend="both")
    cb.set_label("Sea level anomaly relative to the scene median [cm]")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"figO_l3_over_l4_transition_zone.{ext}")
    print("[DONE]")


if __name__ == "__main__":
    main()
