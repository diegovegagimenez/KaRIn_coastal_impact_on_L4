#!/usr/bin/env python3
"""Fig. 5: low-pass filtered (LOESS 15 d) tide gauge, MY and MIOST-K series at
three near-swath stations of the transition zone, R = 50 km, CalVal.
Panel titles give PI_RMSD and the number of grid cells N_pix.

Input : daily series in time_series_optimal/CalVal (R = 50 km) and
per_station_radius_with_pixels_v2.csv.
Output: figures/figD_timeseries_example_v2.{pdf,png}"""
import os
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = Path(__file__).resolve().parent
TS = Path(os.environ.get("KARIN_PROJECT_DIR", "data") + "/PIPELINE_FINAL_LIMPIO/"
          "results_periodo_extendido/optimal_radius_fixed50/time_series_optimal/CalVal")
OUT = HERE.parent / "figures"

C_TG, C_MY, C_MK = "0.15", "#0072B2", "#D55E00"
STATIONS = [
    ("BO_TS_TG_Assens", "Assens (Little Belt)"),
    ("NO_TS_TG_SchleswigTG", "Schleswig (Schlei)"),
    ("NO_TS_TG_LangballigauTG", "Langballigau (Flensburg Fjord)"),
]

M = pd.read_csv(HERE / "per_station_radius_with_pixels_v2.csv")
M = M[(M.phase == "CalVal") & (M.radius == 50)].set_index("station")

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

fig, axes = plt.subplots(3, 1, figsize=(8.6, 8.4), sharex=True)
for ax, (sid, label) in zip(axes, STATIONS):
    df = pd.read_csv(TS / f"TS_data_{sid}_CalVal.csv", parse_dates=["time_d"])
    ax.plot(df.time_d, 100 * df.SLEV_detided_mdac, color="0.75", lw=0.8, zorder=1)
    ax.plot(df.time_d, 100 * df.SLEV_detided_mdac_loess, color=C_TG, lw=2.0, zorder=5, label="Tide gauge")
    ax.plot(df.time_d, 100 * df.sla_my_loess, color=C_MY, lw=1.8, ls="--", zorder=4, label="MY")
    ax.plot(df.time_d, 100 * df.sla_miost_loess, color=C_MK, lw=1.8, zorder=4, label="MIOST-K")
    pi, npix, reg = M.loc[sid, "pi"], int(M.loc[sid, "ncells_my"]), M.loc[sid, "reg"]
    assert reg == "Transition zone", (sid, reg)
    ax.set_title(f"{label}, near-swath, "
                 f"$\\mathrm{{PI}}_{{\\mathrm{{RMSD}}}}$ = {pi:+.1f}%, $N_{{\\mathrm{{pix}}}}$ = {npix}",
                 fontweight="bold")
    ax.set_ylabel("SLA [cm]")
    ax.grid(lw=0.4, alpha=0.4, zorder=0)

axes[0].legend(frameon=False, loc="upper right", ncol=3)
axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"figD_timeseries_example_v2.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print("OK figD_timeseries_example_v2")
print(M.loc[[s for s, _ in STATIONS], ["reg", "pi", "ncells_my", "near"]].round(2).to_string())
