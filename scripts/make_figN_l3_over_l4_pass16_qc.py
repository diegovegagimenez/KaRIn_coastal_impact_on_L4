#!/usr/bin/env python3
"""figN: instantanea SWOT L3 KaRIn (pase 016, 2023-04-22, CalVal) superpuesta a
(a) DUACS-MY (nadir-only) y (b) MIOST-K (nadir + KaRIn), Mediterraneo Occidental.

Igual que figM/figL pero:
  - L3 = ssha_filtered (producto por defecto) y solo pixeles con quality_flag == 0 ("good"): se eliminan los
    pixeles flaggeados cerca de costa (local/soft outliers 5/30,
    bad_quality_coast 10, ocean_unsure 18) ademas de tierra/no-dato;
  - ambos paneles comparten escala de color;
  - se imprimen metricas L3-vs-L4 dentro del swath (L4 interpolado
    bilinealmente a los pixeles L3 validos): RMS de la diferencia, correlacion
    y numero de pixeles L3 usados.
"""
import os
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
L3_FILE = BASE / "data/raw/swot_l3_expert/SWOT_L3_LR_SSH_Expert_498_016_20230422T070340_20230422T075444_v3.0.nc"
MY_FILE = BASE / "data/raw/duacs_l4_my/duacs_sla_eur_MY_0625deg_swot_period.nc"
MK_FILE = BASE / "data/raw/miost_l4/dt_global_allsat_phy_l4_20230422_20250112.nc"
OUT_DIR = Path(__file__).resolve().parent.parent / "figures"

DATE = pd.Timestamp("2023-04-22")
LAT_MIN, LAT_MAX = 38.95, 41.37
LON_MIN, LON_MAX = 0.06, 3.85
PAD = 0.35
QF_KEEP = [0]
CMAP = "RdBu_r"
FS = 13
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                     "font.size": FS, "savefig.dpi": 300, "savefig.bbox": "tight"})


def load_grid(path, sel):
    ds = xr.open_dataset(path)
    sla = ds["sla"].sel(time=DATE, method="nearest") if sel else ds["sla"].isel(time=0)
    sla = sla.sel(latitude=slice(LAT_MIN - PAD, LAT_MAX + PAD),
                  longitude=slice(LON_MIN - PAD, LON_MAX + PAD)) * 100.0
    sla = sla.load()
    ds.close()
    return sla


def load_l3():
    ds = xr.open_dataset(L3_FILE)
    lat = ds["latitude"].values
    lon = ((ds["longitude"].values + 180.0) % 360.0) - 180.0
    ssha = ds["ssha_filtered"].values * 100.0
    q = ds["quality_flag"].values
    ds.close()
    box = ((lat >= LAT_MIN - PAD) & (lat <= LAT_MAX + PAD) &
           (lon >= LON_MIN - PAD) & (lon <= LON_MAX + PAD))
    idx = np.where(box.any(axis=1))[0]
    lat, lon, ssha, q = lat[idx], lon[idx], ssha[idx], q[idx]
    ssha = np.where(np.isin(q, QF_KEEP), ssha, np.nan)
    return lon, lat, ssha


def l3_vs_l4(lon3, lat3, s3, grid):
    inbox = ((lat3 >= LAT_MIN) & (lat3 <= LAT_MAX) & (lon3 >= LON_MIN) & (lon3 <= LON_MAX)
             & np.isfinite(s3))
    pts_lat = xr.DataArray(lat3[inbox], dims="p")
    pts_lon = xr.DataArray(lon3[inbox], dims="p")
    g = grid.interp(latitude=pts_lat, longitude=pts_lon).values
    a = s3[inbox]
    ok = np.isfinite(g)
    a, g = a[ok], g[ok]
    # se elimina la media de cada campo (offset de referencia) antes de comparar
    d = (a - a.mean()) - (g - g.mean())
    return dict(n=int(ok.sum()), rms=float(np.sqrt(np.mean(d ** 2))),
                r=float(np.corrcoef(a, g)[0, 1]), std_l3=float(a.std()), std_l4=float(g.std()))


def main():
    my = load_grid(MY_FILE, sel=True)
    mk = load_grid(MK_FILE, sel=False)
    lon3, lat3, s3 = load_l3()
    print(f"L3 valid pixels (QF=0) in padded box: {np.isfinite(s3).sum()}")
    stats = {}
    for name, g in (("DUACS-MY", my), ("MIOST-K", mk)):
        stats[name] = l3_vs_l4(lon3, lat3, s3, g)
        print(name, {k: round(v, 3) for k, v in stats[name].items()},
              f"L4 cells in box: {int(np.isfinite(g.values).sum())}")

    combo = np.concatenate([my.values[np.isfinite(my.values)], mk.values[np.isfinite(mk.values)],
                            s3[np.isfinite(s3)]])
    vlim = float(np.nanpercentile(np.abs(combo), 98))

    proj = ccrs.Mercator()
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.6), subplot_kw=dict(projection=proj))
    for ax, (lab, g) in zip(axes, (("(a) SWOT KaRIn L3 over DUACS-MY", my),
                                   ("(b) SWOT KaRIn L3 over MIOST-K", mk))):
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
        lo2, la2 = np.meshgrid(g.longitude.values, g.latitude.values)
        pc = ax.pcolormesh(lo2, la2, g.values, cmap=CMAP, vmin=-vlim, vmax=vlim, shading="auto",
                           transform=ccrs.PlateCarree(), zorder=2, rasterized=True)
        ax.pcolormesh(lon3, lat3, s3, cmap=CMAP, vmin=-vlim, vmax=vlim, shading="nearest",
                      transform=ccrs.PlateCarree(), zorder=5, rasterized=True)
        ax.contour(lon3, lat3, np.isfinite(s3).astype(float), levels=[0.5], colors="black",
                   linewidths=1.0, transform=ccrs.PlateCarree(), zorder=6)
        ax.add_feature(cfeature.LAND, facecolor="#3a3a3a", zorder=7)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.9, zorder=8)
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.4, color="gray", linestyle="--", zorder=9)
        gl.top_labels = gl.right_labels = False
        gl.xlocator = mticker.FixedLocator(np.arange(0, 4.5, 1.0))
        gl.ylocator = mticker.FixedLocator(np.arange(39, 41.5, 0.5))
        ax.set_title(lab, fontweight="bold", fontsize=FS + 1)
        st = stats["DUACS-MY" if "MY" in lab else "MIOST-K"]
        ax.text(0.02, 0.02,
                f"2023-04-22, pass 016\nL3 pixels: {st['n']}\n"
                f"RMS(L3$-$L4) = {st['rms']:.2f} cm\n$r$ = {st['r']:.2f}",
                transform=ax.transAxes, fontsize=FS - 2, va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.4", alpha=0.9), zorder=10)
    cb = fig.colorbar(pc, ax=axes, orientation="vertical", fraction=0.025, pad=0.02, extend="both")
    cb.set_label("Sea level anomaly [cm]")
    out = OUT_DIR / "figN_l3_over_l4_pass16_qc"
    fig.savefig(out.with_suffix(".png"))
    fig.savefig(out.with_suffix(".pdf"))
    print("[DONE]", out)


if __name__ == "__main__":
    main()
