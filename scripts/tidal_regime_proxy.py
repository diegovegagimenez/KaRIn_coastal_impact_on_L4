#!/usr/bin/env python3
"""Tidal regime indicator per station: standard deviation of the hourly tide
gauge series after subtracting its centred 25 h running mean (diurnal and
semidiurnal band), CalVal.
Output: tidal_regime_proxy.csv"""
import os
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
TG = Path(os.environ.get("KARIN_PROJECT_DIR", "data") + "/data/processed_NRT/tide_gauges_nrt")
st = pd.read_csv(HERE / "station_regions_v2.csv")
rows = []
for sid in st.station:
    f = TG / f"{sid}.pkl"
    if not f.exists():
        continue
    try:
        tg = pd.read_pickle(f).reset_index()
        tg = tg.rename(columns={"TIME": "time_d"}) if "TIME" in tg.columns else tg
        tg["time_d"] = pd.to_datetime(tg["time_d"])
        tg = tg.dropna(subset=["SLEV"])
        if "SLEV_QC" in tg.columns:
            tg = tg[tg.SLEV_QC.isin([1, 2])]
        tg = tg[(tg.SLEV > -15) & (tg.SLEV < 15)]
        s = tg.set_index("time_d").sort_index().SLEV.resample("1h").mean()
        s = s["2023-03-28":"2023-07-05"]
        hf = (s - s.rolling(25, center=True, min_periods=20).mean()).dropna()
        if len(hf) > 500:
            rows.append(dict(station=sid, tide_std_cm=100 * hf.std()))
    except Exception as e:
        print("fail", sid, e)
pd.DataFrame(rows).to_csv(HERE / "tidal_regime_proxy.csv", index=False)
print(len(rows))
