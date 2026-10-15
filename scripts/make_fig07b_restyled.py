#!/usr/bin/env python3
"""Fig. 2: RMSD maps of MY, MIOST-K and their difference (CalVal and Science),
R = 50 km, results_periodo_extendido/optimal_radius_fixed50.

Layout: 5-column GridSpec, floating colour bars after MIOST-K and after the
difference, statistics box in the upper-left corner, phase names only in the
titles."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

BASE = Path(os.environ.get("KARIN_PROJECT_DIR", "data") + "/PIPELINE_FINAL_LIMPIO/"
            "results_periodo_extendido/optimal_radius_fixed50")
OUT = Path(str(Path(__file__).resolve().parent.parent) + "/"
           "figures/fig07b_rmsd_combined_absolute_and_diff_maps_extended")

RMSD_OUTLIER_CAP = 50.0
LAT_MIN, LAT_MAX = 32, 68
LON_MIN, LON_MAX = -15.0, 42.0
EXTENT = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]
PHASES = {"CalVal": "CalVal", "Science": "Science"}
BASELINE_LABEL, TEST_LABEL = "MY", "MIOST-K"

FS = 16
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": FS,
})


def load(phase):
    df = pd.read_csv(BASE / f"overall_metrics_LOESS_PARALLEL_{phase}_v7.csv")
    df.columns = [c.replace("_loess", "") for c in df.columns]
    mask = (df[["rmsd_my", "rmsd_miost"]] > RMSD_OUTLIER_CAP).any(axis=1)
    return df[~mask].reset_index(drop=True)


def _make_merc_ax(fig, pos, title="", show_left=True, show_bottom=True):
    ax = fig.add_subplot(pos, projection=ccrs.Mercator(central_longitude=13))
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#d9d9d9", zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor="#c6e2f5", zorder=0)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="0.4", zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="0.2", zorder=3)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                       linewidth=0.4, color="0.5", alpha=0.7, linestyle="--", zorder=2)
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = show_left
    gl.bottom_labels = show_bottom
    gl.xlocator = mticker.FixedLocator(range(-15, 45, 15))
    gl.ylocator = mticker.FixedLocator(range(30, 75, 10))
    gl.xlabel_style = {"size": FS - 1}
    gl.ylabel_style = {"size": FS - 1}

    if title:
        ax.set_title(title, fontsize=FS, pad=4)
    return ax


def _scatter_map(ax, lons, lats, values, cmap, vmin, vmax, s=25):
    return ax.scatter(lons, lats, c=values, s=s, cmap=cmap, vmin=vmin, vmax=vmax,
                       edgecolors="k", linewidths=0.3, transform=ccrs.PlateCarree(), zorder=5)


def _add_floating_cbar(fig, sc, ax, label):
    cax = inset_axes(ax, width="4%", height="100%", loc="lower left",
                      bbox_to_anchor=(1.03, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    cb = fig.colorbar(sc, cax=cax, orientation="vertical", extend="both")
    cb.set_label(label, fontsize=FS - 1)
    cb.ax.tick_params(labelsize=FS - 2)
    return cb


datasets = [(load(ph), ph) for ph in PHASES]

all_rmsd = pd.concat([datasets[0][0]["rmsd_my"], datasets[0][0]["rmsd_miost"],
                       datasets[1][0]["rmsd_my"], datasets[1][0]["rmsd_miost"]])
abs_vmin, abs_vmax = 0, np.nanpercentile(all_rmsd, 95)
delta_vmin, delta_vmax = -2, 2

fig = plt.figure(figsize=(17.0, 11))
gs = gridspec.GridSpec(2, 5, figure=fig,
                        width_ratios=[1, 0.005, 1, 0.2, 1],
                        wspace=0.0, hspace=0.08,
                        left=0.03, right=0.92, top=0.94, bottom=0.02)
panel_labels = [["(a)", "(b)", "(c)"], ["(d)", "(e)", "(f)"]]

for row, (df, phase_key) in enumerate(datasets):
    lons, lats = df["longitude"].values, df["latitude"].values
    rmsd_my, rmsd_mk = df["rmsd_my"].values, df["rmsd_miost"].values
    delta_rmsd = rmsd_my - rmsd_mk

    panels = [
        {"vals": rmsd_my, "title": f"RMSD {BASELINE_LABEL}", "cmap": "RdYlGn_r",
         "vmin": abs_vmin, "vmax": abs_vmax, "label": "RMSD (cm)", "is_diff": False},
        {"vals": rmsd_mk, "title": f"RMSD {TEST_LABEL}", "cmap": "RdYlGn_r",
         "vmin": abs_vmin, "vmax": abs_vmax, "label": "RMSD (cm)", "is_diff": False},
        {"vals": delta_rmsd, "title": f"$\\Delta$RMSD ({BASELINE_LABEL} $-$ {TEST_LABEL})", "cmap": "seismic_r",
         "vmin": delta_vmin, "vmax": delta_vmax,
         "label": f"$\\Delta$RMSD (cm)\n(+) = {TEST_LABEL} better", "is_diff": True},
    ]

    for col, pconf in enumerate(panels):
        gs_col = col * 2
        title = f"{panel_labels[row][col]} {pconf['title']} – {PHASES[phase_key]}"
        ax = _make_merc_ax(fig, gs[row, gs_col], title=title,
                            show_left=(col == 0), show_bottom=(row == 1))
        sc = _scatter_map(ax, lons, lats, pconf["vals"], cmap=pconf["cmap"],
                           vmin=pconf["vmin"], vmax=pconf["vmax"])

        valid_vals = pconf["vals"][np.isfinite(pconf["vals"])]
        if len(valid_vals) > 0:
            mean_val = np.mean(valid_vals)
            if pconf["is_diff"]:
                n_better = (valid_vals > 0).sum()
                n_valid = len(valid_vals)
                pct_better = 100 * n_better / n_valid
                stats_txt = (f"Mean $\\Delta$: {mean_val:+.2f} cm\n"
                             f"{TEST_LABEL} better: {pct_better:.0f}% ({n_better}/{n_valid})")
            else:
                median_val = np.median(valid_vals)
                stats_txt = f"Mean: {mean_val:.2f} cm\nMedian: {median_val:.2f} cm"

            ax.text(0.03, 0.97, stats_txt, transform=ax.transAxes, fontsize=FS - 3,
                    va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.85),
                    zorder=10)

        if col != 0:
            _add_floating_cbar(fig, sc, ax, label=pconf["label"])

fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print(f"OK {OUT}.png / .pdf")
