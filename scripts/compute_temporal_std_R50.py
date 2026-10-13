#!/usr/bin/env python3
"""Variabilidad temporal REAL de cada producto a R = 50 km.

Desviacion estandar (cm) de la serie altimetrica filtrada (LOESS 15 d) extraida
en cada estacion, para MY y MIOST-K, y del mareografo detidado y corregido de
DAC. No confundir con las columnas std_* de los CSV del pipeline, que son la
desviacion estandar de la DIFERENCIA altimetria - mareografo (equivale al RMSD).

Entrada : series diarias guardadas por P1 (time_series_optimal/{CalVal,Science}).
Salida  : temporal_std_products_R50.csv (usado por make_fig07d_std.py).
"""
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
T = (os.environ.get("KARIN_PROJECT_DIR", "data") + "/PIPELINE_FINAL_LIMPIO/"
     "results_periodo_extendido/optimal_radius_fixed50/time_series_optimal/")
REG = pd.read_csv(HERE / "station_regions_v2.csv").set_index("station").region_v2

rows = []
for ph in ("CalVal", "Science"):
    for f in glob.glob(T + ph + "/*.csv"):
        sid = os.path.basename(f)[len("TS_data_"):-len(f"_{ph}.csv")]
        x = pd.read_csv(f).dropna(subset=["SLEV_detided_mdac_loess", "sla_my_loess", "sla_miost_loess"])
        if len(x) < 20:
            continue
        rows.append(dict(station=sid, phase=ph, reg=REG.get(sid),
                         tg=x.SLEV_detided_mdac_loess.std() * 100,
                         my=x.sla_my_loess.std() * 100, mk=x.sla_miost_loess.std() * 100))
r = pd.DataFrame(rows)
r["d"] = r.mk - r.my
r["rmy"], r["rmk"] = r.my / r.tg, r.mk / r.tg
r["closer"] = (r.mk - r.tg).abs() < (r.my - r.tg).abs()
r.to_csv(HERE / "temporal_std_products_R50.csv", index=False, float_format="%.4f")

rng = np.random.default_rng(0)


def ci(v, n=10000):
    v = np.asarray(v)
    m = v[rng.integers(0, len(v), (n, len(v)))].mean(axis=1)
    return v.mean(), *np.percentile(m, [2.5, 97.5])


for ph in ("CalVal", "Science"):
    for lab, x in (("network", r[r.phase == ph]), ("transition zone", r[(r.phase == ph) & (r.reg == "Transition zone")])):
        print(f"{ph:8s} {lab:16s} n={len(x):3d} median STD tg {x.tg.median():.2f} MY {x.my.median():.2f} "
              f"MK {x.mk.median():.2f} | ratio MY {x.rmy.median():.2f} MK {x.rmk.median():.2f} | "
              f"mean(MK-MY) {ci(x.d)[0]:+.3f} [{ci(x.d)[1]:+.3f},{ci(x.d)[2]:+.3f}] | "
              f"MK more variable {100 * (x.d > 0).mean():.0f}% | MK closer to TG {100 * x.closer.mean():.0f}%")
