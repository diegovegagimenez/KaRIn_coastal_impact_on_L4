#!/usr/bin/env python3
"""fig07c_corr_combined_absolute_and_diff_maps_extended: analoga a
fig07b (RMSD) pero para la correlacion con el mareografo (MY, MIOST-K, y su
diferencia). Misma estetica exacta que make_fig07b_restyled.py
(results/advanced_analysis/plot_combined_rmsd_maps.py): gridspec de 5
columnas, colorbars flotantes, titulos sin negrita, caja de stats arriba a
la izquierda, FS=16.

Convencion de signo: delta_corr = corr_miost - corr_my (positivo = MIOST-K
mejor, mayor correlacion), igual que en fig07b donde delta_rmsd = rmsd_my -
rmsd_miost es positivo cuando MIOST-K mejora. Con esto el mismo cmap
seismic_r y la misma leyenda "(+) = MIOST-K better" mantienen el mismo
significado de color (azul = MIOST-K mejor) en ambas figuras.
"""
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
           "figures/fig07c_corr_combined_absolute_and_diff_maps_extended")

RMSD_OUTLIER_CAP = 50.0  # mismo filtro de outliers que fig07b (por RMSD, no por corr)
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


def _add_floating_cbar(fig, sc, ax, label, extend="both"):
    cax = inset_axes(ax, width="4%", height="100%", loc="lower left",
                      bbox_to_anchor=(1.03, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    cb = fig.colorbar(sc, cax=cax, orientation="vertical", extend=extend)
    cb.set_label(label, fontsize=FS - 1)
    cb.ax.tick_params(labelsize=FS - 2)
    return cb


datasets = [(load(ph), ph) for ph in PHASES]

# La mayoria de estaciones caen entre 0.7 y 1.0 (mediana 0.87, P90 0.98); un
# vmin bajo (percentil 2 ~ -0.35) satura todo ese rango a verde oscuro. Se
# recorta a vmin=0.4 (~ P10 global) para que el degradado se reparta dentro
# del rango donde vive la red; las pocas estaciones peores quedan saturadas
# en el extremo rojo, marcadas con la flecha "extend=min" de la colorbar.
abs_vmin, abs_vmax = 0.4, 1.0
delta_vmin, delta_vmax = -0.1, 0.1

fig = plt.figure(figsize=(17.0, 11))
gs = gridspec.GridSpec(2, 5, figure=fig,
                        width_ratios=[1, 0.005, 1, 0.2, 1],
                        wspace=0.0, hspace=0.08,
                        left=0.03, right=0.92, top=0.94, bottom=0.02)
panel_labels = [["(a)", "(b)", "(c)"], ["(d)", "(e)", "(f)"]]

for row, (df, phase_key) in enumerate(datasets):
    lons, lats = df["longitude"].values, df["latitude"].values
    corr_my, corr_mk = df["corr_my"].values, df["corr_miost"].values
    delta_corr = corr_mk - corr_my

    panels = [
        {"vals": corr_my, "title": f"Corr. {BASELINE_LABEL}", "cmap": "viridis",
         "vmin": abs_vmin, "vmax": abs_vmax, "label": "Correlation", "is_diff": False},
        {"vals": corr_mk, "title": f"Corr. {TEST_LABEL}", "cmap": "viridis",
         "vmin": abs_vmin, "vmax": abs_vmax, "label": "Correlation", "is_diff": False},
        {"vals": delta_corr, "title": f"$\\Delta r$ ({TEST_LABEL} $-$ {BASELINE_LABEL})", "cmap": "seismic_r",
         "vmin": delta_vmin, "vmax": delta_vmax,
         "label": f"$\\Delta r$\n(+) = {TEST_LABEL} better", "is_diff": True},
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
                stats_txt = (f"Mean $\\Delta r$: {mean_val:+.3f}\n"
                             f"{TEST_LABEL} better: {pct_better:.0f}% ({n_better}/{n_valid})")
            else:
                median_val = np.median(valid_vals)
                stats_txt = f"Mean: {mean_val:.3f}\nMedian: {median_val:.3f}"

            ax.text(0.03, 0.97, stats_txt, transform=ax.transAxes, fontsize=FS - 3,
                    va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.85),
                    zorder=10)

        if col != 0:
            _add_floating_cbar(fig, sc, ax, label=pconf["label"],
                                extend="both" if pconf["is_diff"] else "min")

fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print(f"OK {OUT}.png / .pdf")
