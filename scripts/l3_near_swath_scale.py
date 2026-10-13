#!/usr/bin/env python3
"""L3 SWOT (KaRIn) vs L4 MIOST-K vs Tide Gauges, CalVal only, near-swath stations,
radius sweep 10-100 km.

Pregunta: durante CalVal, para las estaciones "near" del swath (<=2 km de un
track CalVal, panel fijo 10-100 km ya usado en el articulo -> n=56, ver
nearfar_scale_stats.py / Tabla 9), es el propio L3 (nube de puntos SWOT KaRIn)
mejor que el L4 experimental (MIOST-K, que asimila ese mismo L3) frente al
mareografo?

Metodologia -- comparable con las Tablas 2-9 del articulo. Version corregida:
la primera version de este script usaba SLA_TG = SLEV - media (sin detidado
ni DAC), y reutilizaba TS_data_*.csv del pipeline principal para TG/L4 -- pero
esos ficheros de serie temporal SOLO se generaron y guardaron en disco para
R=50 km (el resto de radios solo tienen el CSV resumen por estacion, sin la
serie diaria). Esta version ya NO depende de esos TS_data: reconstruye TG y
L4 exactamente con las mismas funciones y parametros que usa
`P1_altimetry_tidegauge_processing.py` para producir las Tablas 2-9, para
cualquier radio:
  - TG: `P1_altimetry_tidegauge_processing.process_tg_detided` (detidado
    armonico UTide sobre el registro completo del mareografo) menos DAC
    (`P1_altimetry_tidegauge_processing.extract_dac_daily`, radio fijo 50 km,
    igual que el pipeline -- el DAC es suave y no se barre con el radio).
  - L4 (MIOST-K): mediana espacial de `sla` dentro de R km sobre la malla
    nativa MIOST (0.125°), extraida directamente de los ficheros MIOST L4
    locales. Es la misma operacion que
    `P1_altimetry_tidegauge_processing.extract_miost_multiradius`: con
    `common_grid="coarsest"` (el valor por defecto y el usado para las
    Tablas 2-9) la malla comun ES la malla nativa de MIOST, por lo que el
    regrid es la identidad y el calculo se reduce a una mediana por radio,
    igual que aqui.
  - L3: mediana espacial de `ssha_filtered` dentro de R km, por pasada (los 4
    pases CalVal 1/3/14/16), extraido directamente de los ficheros L3 locales.
  - Filtro conjunto 3-sigma + recentrado + suavizado LOESS de 15 dias
    (mismos `SIGMA_THRESHOLD=3.0` y `LOESS_WINDOW=15` que
    `P1_altimetry_tidegauge_processing.py`), aplicado por estacion y por radio
    sobre el merge {TG_detided_mdac, L3, MIOST-K} de fechas comunes (el join
    con L3 puede recortar el conjunto de dias respecto al usado en las Tablas
    2-9 para ese radio, porque L3 no cubre todos los dias a radios pequenos).
  - RMSD y correlacion respecto al TG, PI = 100*(RMSD_L4-RMSD_L3)/RMSD_L4
    (positivo = L3 mejor que L4).

Salidas en results/l3_near_swath_scale/:
  - per_station_radius.csv   (56 estaciones x 10 radios)
  - summary_by_radius.csv    (agregado: mediana/media/%mejor/test por radio)
  - l3_cache.pkl              (cache de la extraccion L3 cruda)
  - tg_dac_cache.pkl          (cache de TG detidado + DAC, no depende del radio)

Ejecutar:
    conda run -n swot_env python l3_near_swath_scale.py
"""
from __future__ import annotations

import os
import pickle
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================
ART_DIR = Path(os.environ.get("KARIN_PROJECT_DIR", "data"))
PIPE_DIR = ART_DIR / "PIPELINE_FINAL_LIMPIO"
OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "l3_near_swath_scale"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PIPE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import P1_altimetry_tidegauge_processing as P1  # noqa: E402  (mismo modulo que produce Tablas 2-9)

PATHS = {
    "tg_dir": ART_DIR / "data/processed_NRT/tide_gauges_nrt",
    "l3_dir": ART_DIR / "data/raw/swot_l3_expert",
    "l4_dir": ART_DIR / "data/raw/miost_l4",
    "dac_dir": ART_DIR / "data/raw/dac_l4",
    "duacs_my": ART_DIR / "data/raw/duacs_l4_my/duacs_sla_eur_MY_0625deg_swot_period.nc",
    "scale_dir": PIPE_DIR / "results_periodo_extendido",
    "prox_csv": PIPE_DIR / "results_P2_periodo_comun_50km" / "advanced_analysis_v2_fixed"
                / "miost_multivariate__vs__miost_multivariate_karin" / "swot_proximity_by_station.csv",
}

RADII = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
RMSD_CAP = 50.0           # cm, mismo criterio que el resto del articulo
NEAR_KM = 2.0              # umbral near-swath ya establecido en el articulo
CALVAL_START = pd.Timestamp("2023-03-28")
CALVAL_END = pd.Timestamp("2023-07-05")

LOESS_WINDOW_DAYS = 15.0
SIGMA_THRESHOLD = 3.0
DAC_RADIUS_KM = P1.DAC_RADIUS_DEFAULT     # 50 km, fijo (igual que Tablas 2-9)
BOX_MARGIN = 1.25                          # mismo margen por defecto del pipeline

# --- quality_flag de SWOT L3 ---------------------------------------------
# `ssha_filtered`/`ssha_unfiltered` llevan el editado por quality_flag YA
# incorporado (NaN donde el flag no es 0), lo que descarta de forma agresiva
# datos cerca de costa (bad_quality_coast=10, ocean_unsure=18) y en zonas de
# alta variabilidad (soft/local_outliers=30/5) -- comprobado: cerca de la
# estacion BO_TS_TG_Bagenkop (estrecho danes) el flag 30 por si solo
# descarta mas puntos que el flag 0 "good". Para poder usar datos cerca de
# costa se usa `ssha_unedited` (calibrado, SIN editar) y se aplica una
# mascara propia que solo excluye lo que no es dato oceanico real:
#   101 = not_on_sea (tierra)      102 = no_data (sin dato)
#   100 = bad_swath_extremities (borde de swath, casi sin dato real tampoco)
# Todo lo demas (incl. bad_quality_coast=10, ocean_unsure=18,
# soft/local_outliers=30/5, eclipse=3, ice=19/20, rain=25, extremes=50,
# mission_event=70) se CONSERVA.
QF_MEANING = {
    0: "good", 3: "eclipse", 5: "local_outliers", 10: "bad_quality_coast",
    18: "ocean_unsure", 19: "ice_unsure", 20: "ice", 25: "rain",
    30: "soft_outliers", 50: "extremes", 70: "mission_event",
    100: "bad_swath_extremities", 101: "not_on_sea", 102: "no_data",
}
QF_EXCLUDE = {100, 101, 102}


def haversine(lon1, lat1, lon2, lat2):
    lon1, lon2, lat1, lat2 = map(np.deg2rad, [lon1, lon2, lat1, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * 6371.0 * np.arcsin(np.sqrt(a))


# =============================================================================
# 1. PANEL FIJO DE ESTACIONES NEAR-SWATH (CalVal, 10-100 km) -- reproduce
#    exactamente el mismo filtro que scripts/nearfar_scale_stats.py
# =============================================================================

def build_near_panel() -> pd.DataFrame:
    sets = []
    for r in RADII:
        d = pd.read_csv(PATHS["scale_dir"] / f"optimal_radius_fixed{r}" / "overall_metrics_LOESS_PARALLEL_CalVal_v7.csv")
        d.columns = [c.replace("_loess", "") for c in d.columns]
        rc = [c for c in d.columns if c.startswith("rmsd_")]
        d = d[~(d[rc] > RMSD_CAP).any(axis=1)]
        sets.append(set(d.station))
    panel = set.intersection(*sets)

    prox = pd.read_csv(PATHS["prox_csv"])
    dist = dict(zip(prox.station, prox.swot_track_distance_km))
    near = sorted(s for s in panel if dist.get(s, np.inf) <= NEAR_KM)

    coords = pd.read_csv(PATHS["scale_dir"] / "optimal_radius_fixed50" / "overall_metrics_LOESS_PARALLEL_CalVal_v7.csv")
    coords = coords[coords.station.isin(near)][["station", "latitude", "longitude"]].drop_duplicates("station")
    coords = coords.reset_index(drop=True)
    print(f"[PANEL] near-swath CalVal, panel fijo 10-100km: n={len(coords)} "
          f"(de {len(panel)} en panel fijo total, umbral {NEAR_KM} km)")
    return coords


# =============================================================================
# 2. EXTRACCION L3 (un solo paso por archivo, todos los radios y estaciones)
# =============================================================================

def extract_l3(stations: pd.DataFrame) -> dict[str, pd.DataFrame]:
    l3_dir = PATHS["l3_dir"]
    files = sorted(l3_dir.glob("SWOT_L3_LR_SSH_Expert_*.nc"))
    print(f"[L3] {len(files)} ficheros locales a procesar (1 pasada por estacion+radio, "
          f"pases CalVal 1/3/14/16)")
    print(f"  variable: ssha_unedited (SIN editado por quality_flag) + mascara manual "
          f"excluyendo solo {sorted(QF_EXCLUDE)} ({', '.join(QF_MEANING[f] for f in sorted(QF_EXCLUDE))})")

    rows: dict[str, list[dict]] = {sid: [] for sid in stations.station}
    max_r = max(RADII)

    for i, nc_path in enumerate(files):
        try:
            ds = xr.open_dataset(nc_path)
            lat = ds["latitude"].values
            lon = ((ds["longitude"].values + 180.0) % 360.0) - 180.0
            ssha = ds["ssha_unedited"].values
            qflag = ds["quality_flag"].values
            time_vals = ds["time"].values
            ds.close()
        except Exception as e:
            print(f"  [WARN] {nc_path.name}: {e}")
            continue

        # solo se excluyen tierra (not_on_sea) y no_data/bordes de swath sin dato
        # real; TODOS los flags "de calidad" relacionados con la costa
        # (bad_quality_coast, ocean_unsure, soft/local_outliers, etc.) se
        # mantienen, para poder usar datos cerca de costa.
        ssha = np.where(np.isin(qflag, list(QF_EXCLUDE)), np.nan, ssha)

        for sid, lat_tg, lon_tg in stations.itertuples(index=False):
            lat_diff = np.abs(lat - lat_tg)
            lon_diff = np.abs(lon - lon_tg)
            bbox = (lat_diff < 1.0) & (lon_diff < 1.0 / np.cos(np.deg2rad(lat_tg)))
            if not np.any(bbox):
                continue

            dist = np.full_like(lat, np.inf)
            dist[bbox] = haversine(lon[bbox], lat[bbox], lon_tg, lat_tg)

            line_mask_max = (dist <= max_r).any(axis=1)
            line_idx = np.where(line_mask_max)[0]
            if len(line_idx) == 0:
                continue

            rep_time = pd.Timestamp(np.median(time_vals[line_idx].astype("int64")))
            row = {"time": rep_time}
            any_valid = False
            for r in RADII:
                m = dist <= r
                vals = ssha[m]
                valid = vals[np.isfinite(vals)]
                if len(valid) > 0:
                    row[f"SLA_L3_R{r}"] = float(np.nanmedian(valid))
                    any_valid = True
                else:
                    row[f"SLA_L3_R{r}"] = np.nan
            if any_valid:
                rows[sid].append(row)

        if (i + 1) % 50 == 0:
            print(f"  [L3] {i + 1}/{len(files)} ficheros procesados")

    out = {}
    for sid, recs in rows.items():
        if not recs:
            out[sid] = pd.DataFrame(columns=["time"] + [f"SLA_L3_R{r}" for r in RADII])
            continue
        df = pd.DataFrame(recs).sort_values("time").reset_index(drop=True)
        df["time"] = pd.to_datetime(df["time"]).dt.normalize()
        # si hay >1 pasada el mismo dia, mediana
        df = df.groupby("time", as_index=False).median(numeric_only=True)
        out[sid] = df
    return out


# =============================================================================
# 3a. TG detidado (UTide, registro completo) + DAC (radio fijo 50 km) --
#     mismas funciones que usa P1_altimetry_tidegauge_processing.py para las
#     Tablas 2-9. Independiente del radio de extraccion altimetrica.
# =============================================================================

class _FilteredDacDir:
    """Envoltorio de PATHS['dac_dir'] cuyo .glob() solo devuelve los ficheros
    DAC (6-horarios) dentro de la ventana CalVal, para no escanear ~4200
    ficheros de todo el registro cuando solo hace falta ~1/10 de ellos."""

    def __init__(self, real_dir: Path, start: pd.Timestamp, end: pd.Timestamp):
        self.real_dir = real_dir
        epoch = pd.Timestamp("1950-01-01")
        self.jday_min = (start - epoch).days - 1
        self.jday_max = (end - epoch).days + 1

    def glob(self, pattern):
        out = []
        for f in sorted(self.real_dir.glob(pattern)):
            m = re.match(r"dac_dif_(\d+)_(\d+)", f.stem)
            if not m:
                continue
            jday = int(m.group(1))
            if self.jday_min <= jday <= self.jday_max:
                out.append(f)
        return out


def build_tg_dac(stations: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Devuelve {station: DataFrame(time, SLEV_detided_mdac)} SIN centrar
    (el centrado se hace en p4_filter_smooth, tras intersectar con L3/L4,
    igual que hace metrics_station_radius en el pipeline principal)."""
    tg_coords = {}
    detided = {}
    for sid, lat_tg, lon_tg in stations.itertuples(index=False):
        pkl_path = PATHS["tg_dir"] / f"{sid}.pkl"
        res, info = P1.process_tg_detided(pkl_path)
        if res is None:
            print(f"  [WARN] TG detide fallo para {sid}: {info}")
            continue
        detided[sid] = res
        tg_coords[sid] = (lat_tg, lon_tg)

    print(f"[TG] {len(detided)}/{len(stations)} estaciones detidadas (UTide) correctamente")

    dac_dir = _FilteredDacDir(PATHS["dac_dir"], CALVAL_START, CALVAL_END)
    dac = P1.extract_dac_daily(dac_dir, tg_coords, DAC_RADIUS_KM, BOX_MARGIN)

    out = {}
    for sid, tg_df in detided.items():
        d = tg_df.copy()
        dac_df = dac.get(sid)
        if dac_df is not None and len(dac_df):
            d = pd.merge(d, dac_df, on="time_d", how="left")
        if "dac" in d.columns:
            d["SLEV_detided_mdac"] = d["SLEV_detided"] - d["dac"]
        else:
            d["SLEV_detided_mdac"] = d["SLEV_detided"]
        d = d.rename(columns={"time_d": "time"})
        d["time"] = pd.to_datetime(d["time"]).dt.normalize()
        d = d[(d.time >= CALVAL_START) & (d.time <= CALVAL_END)]
        out[sid] = d[["time", "SLEV_detided_mdac"]]
    return out


# =============================================================================
# 3b. EXTRACCION L4 MIOST-K (mediana en R km, malla nativa 0.125° -- identica
#     a extract_miost_multiradius con common_grid='coarsest', el valor usado
#     para las Tablas 2-9)
# =============================================================================

def extract_l4(stations: pd.DataFrame) -> dict[str, pd.DataFrame]:
    l4_dir = PATHS["l4_dir"]
    all_files = sorted(l4_dir.glob("dt_global_allsat_phy_l4_*.nc"))

    def file_date(p: Path):
        m = re.search(r"phy_l4_(\d{8})_", p.name)
        return pd.Timestamp(m.group(1)) if m else None

    files = [(f, file_date(f)) for f in all_files]
    files = [(f, d) for f, d in files if d is not None and CALVAL_START <= d <= CALVAL_END]
    print(f"[L4] {len(files)} ficheros MIOST-K en ventana CalVal "
          f"({CALVAL_START.date()} -> {CALVAL_END.date()})")

    if not files:
        raise RuntimeError("No se encontraron ficheros MIOST L4 en la ventana CalVal")

    ds_ref = xr.open_dataset(files[0][0])
    lat1d = ds_ref["latitude"].values
    lon1d = ds_ref["longitude"].values
    ds_ref.close()

    max_r = max(RADII)
    station_idx: dict[str, tuple] = {}
    for sid, lat_tg, lon_tg in stations.itertuples(index=False):
        deg_pad = max_r / 111.0 + 0.2
        lat_mask = np.abs(lat1d - lat_tg) <= deg_pad
        lon_diff = np.abs(((lon1d - lon_tg + 180.0) % 360.0) - 180.0)
        lon_mask = lon_diff <= deg_pad / np.cos(np.deg2rad(lat_tg))
        lat_idx = np.where(lat_mask)[0]
        lon_idx = np.where(lon_mask)[0]
        sub_lat = lat1d[lat_idx]
        sub_lon = lon1d[lon_idx]
        LON, LAT = np.meshgrid(sub_lon, sub_lat)
        dist = haversine(LON, LAT, lon_tg, lat_tg)
        station_idx[sid] = (lat_idx, lon_idx, dist)

    rows: dict[str, list[dict]] = {sid: [] for sid in stations.station}

    for i, (fpath, date) in enumerate(files):
        try:
            ds = xr.open_dataset(fpath)
            sla_full = ds["sla"].isel(time=0).values
            ds.close()
        except Exception as e:
            print(f"  [WARN] {fpath.name}: {e}")
            continue

        for sid, (lat_idx, lon_idx, dist) in station_idx.items():
            sub = sla_full[np.ix_(lat_idx, lon_idx)]
            row = {"time": date}
            any_valid = False
            for r in RADII:
                m = dist <= r
                vals = sub[m]
                valid = vals[np.isfinite(vals)]
                if len(valid) > 0:
                    row[f"SLA_L4_R{r}"] = float(np.nanmedian(valid))
                    any_valid = True
                else:
                    row[f"SLA_L4_R{r}"] = np.nan
            if any_valid:
                rows[sid].append(row)

        if (i + 1) % 20 == 0:
            print(f"  [L4] {i + 1}/{len(files)} ficheros procesados")

    out = {}
    for sid, recs in rows.items():
        if not recs:
            out[sid] = pd.DataFrame(columns=["time"] + [f"SLA_L4_R{r}" for r in RADII])
            continue
        df = pd.DataFrame(recs).sort_values("time").reset_index(drop=True)
        df["time"] = pd.to_datetime(df["time"]).dt.normalize()
        out[sid] = df
    return out


# =============================================================================
# 3c. EXTRACCION DUACS MY (nadir puro, SIN KaRIn) -- REUTILIZA
#     P1_altimetry_tidegauge_processing.extract_single_nc_multiradius, la
#     misma funcion que produce la columna sla_my de las Tablas 2-9: regrid
#     conservativo (area-weighted) de la malla nativa DUACS (0.0625°) a la
#     malla comun MIOST (0.125°, common_grid="coarsest") y despues mediana
#     por radio sobre esa malla comun.
# =============================================================================

def extract_duacs_my(stations: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ref_file = sorted(PATHS["l4_dir"].glob("dt_global_allsat_phy_l4_*.nc"))[0]
    ds_ref = xr.open_dataset(ref_file)
    tgt_lat = ds_ref["latitude"].values
    tgt_lon = ds_ref["longitude"].values
    ds_ref.close()

    tg_coords = {sid: (lat, lon) for sid, lat, lon in stations.itertuples(index=False)}

    print(f"[DUACS-MY] {PATHS['duacs_my'].name}: extract_single_nc_multiradius "
          f"(regrid conservativo -> malla MIOST, igual que Tablas 2-9) ...")
    out_raw, _ncells = P1.extract_single_nc_multiradius(
        PATHS["duacs_my"], "sla_my", tg_coords, RADII, BOX_MARGIN,
        tgt_lat=tgt_lat, tgt_lon=tgt_lon, method="conservative")

    out = {}
    for sid, df in out_raw.items():
        d = df.rename(columns={"time_d": "time"})
        d["time"] = pd.to_datetime(d["time"]).dt.normalize()
        d = d[(d.time >= CALVAL_START) & (d.time <= CALVAL_END)]
        rename = {f"sla_my|{r}": f"SLA_DUACS_R{r}" for r in RADII}
        d = d.rename(columns=rename)
        keep_cols = ["time"] + [c for c in rename.values() if c in d.columns]
        out[sid] = d[keep_cols]
    print(f"  -> {len(out)} estaciones con datos DUACS-MY")
    return out


# =============================================================================
# 5. P4: filtro 3-sigma conjunto (+ recentrado) [+ LOESS 15 dias, OPCIONAL]
#
# `smooth=True`  -> metodologia identica a las Tablas 2-9 (3-sigma + LOESS).
# `smooth=False` -> "sin filtrado temporal": se mantiene el filtro 3-sigma
#   conjunto (es un filtro de amplitud/outliers, no temporal -- sin el, un
#   solo dia con un outlier de posicionamiento/nube dispara RMSD espurios que
#   no reflejan la señal oceanica) y el recentrado, pero NO se aplica el
#   suavizado LOESS de 15 dias: se comparan los valores DIARIOS crudos de TG,
#   L3 y MIOST-K entre si, sin paso alguno de baja frecuencia.
# =============================================================================

def p4_filter_smooth(df: pd.DataFrame, cols: list[str], smooth: bool = True) -> pd.DataFrame:
    d = df.dropna(subset=cols).copy()
    if len(d) < 8:
        return d.iloc[0:0]

    mask = pd.Series(True, index=d.index)
    for c in cols:
        mu, sd = d[c].mean(), d[c].std()
        mask &= (np.abs(d[c] - mu) <= SIGMA_THRESHOLD * sd)
    d = d[mask].copy()
    if len(d) < 8:
        return d.iloc[0:0]

    for c in cols:
        d[c] = d[c] - d[c].mean()

    if not smooth:
        return d

    t_num = (d["time"] - d["time"].min()).dt.total_seconds() / 86400.0
    t_range = t_num.max() - t_num.min()
    if t_range > 0:
        frac = min(LOESS_WINDOW_DAYS / t_range, 1.0)
        for c in cols:
            d[c] = lowess(d[c].values, t_num.values, frac=frac, return_sorted=False)
    return d


# =============================================================================
# 6. PER STATION / PER RADIUS METRICS
# =============================================================================

def compute_metrics(stations: pd.DataFrame, l3: dict, tg_dac: dict, l4: dict,
                    smooth: bool = True) -> pd.DataFrame:
    records = []
    for sid, lat_tg, lon_tg in stations.itertuples(index=False):
        df_l3 = l3.get(sid, pd.DataFrame())
        df_tg = tg_dac.get(sid, pd.DataFrame())
        df_l4 = l4.get(sid, pd.DataFrame())
        if df_l3.empty or df_tg.empty or df_l4.empty:
            continue

        for r in RADII:
            l3_col = f"SLA_L3_R{r}"
            l4_col = f"SLA_L4_R{r}"
            if l4_col not in df_l4.columns:
                continue

            merged = pd.merge(df_tg, df_l3[["time", l3_col]], on="time", how="inner")
            merged = pd.merge(merged, df_l4[["time", l4_col]], on="time", how="inner")
            if merged.empty:
                continue

            cols = ["SLEV_detided_mdac", l3_col, l4_col]
            d = p4_filter_smooth(merged, cols, smooth=smooth)
            if len(d) < 8:
                continue

            tg = d["SLEV_detided_mdac"].values * 100.0
            l3v = d[l3_col].values * 100.0
            l4v = d[l4_col].values * 100.0

            rmsd_l3 = float(np.sqrt(np.mean((tg - l3v) ** 2)))
            rmsd_l4 = float(np.sqrt(np.mean((tg - l4v) ** 2)))
            corr_l3 = float(np.corrcoef(tg, l3v)[0, 1])
            corr_l4 = float(np.corrcoef(tg, l4v)[0, 1])

            # sanidad: mismo tope que build_near_panel()/resto del articulo.
            # Con los flags de costa desactivados, unas pocas estaciones de
            # alta latitud (hielo en el Golfo de Botnia: Furuogrund,
            # Stromoren) muestran SSH espurio de decenas de metros -- el
            # filtro 3-sigma no lo detecta porque esta calculado sobre la
            # propia serie ya contaminada. Se descartan esas filas en vez de
            # dejarlas distorsionar la media/PI.
            if rmsd_l3 > RMSD_CAP or rmsd_l4 > RMSD_CAP:
                continue

            records.append(dict(
                station=sid, latitude=lat_tg, longitude=lon_tg, radius=r, n=len(d),
                rmsd_l3=rmsd_l3, rmsd_l4=rmsd_l4,
                corr_l3=corr_l3, corr_l4=corr_l4,
                delta_rmsd=rmsd_l4 - rmsd_l3,
                pi_rmsd=100.0 * (rmsd_l4 - rmsd_l3) / rmsd_l4 if rmsd_l4 != 0 else np.nan,
            ))

    return pd.DataFrame(records)


def compute_metrics_3products(stations: pd.DataFrame, l3: dict, tg_dac: dict, l4: dict,
                              duacs: dict, smooth: bool = True) -> pd.DataFrame:
    """Igual que compute_metrics, pero añadiendo DUACS-MY (nadir puro, sin
    KaRIn) como tercer producto. El filtro 3-sigma conjunto y el LOESS se
    aplican sobre las CUATRO series {TG, L3, MIOST-K, DUACS-MY} a la vez."""
    records = []
    for sid, lat_tg, lon_tg in stations.itertuples(index=False):
        df_l3 = l3.get(sid, pd.DataFrame())
        df_tg = tg_dac.get(sid, pd.DataFrame())
        df_l4 = l4.get(sid, pd.DataFrame())
        df_du = duacs.get(sid, pd.DataFrame())
        if df_l3.empty or df_tg.empty or df_l4.empty or df_du.empty:
            continue

        for r in RADII:
            l3_col = f"SLA_L3_R{r}"
            l4_col = f"SLA_L4_R{r}"
            du_col = f"SLA_DUACS_R{r}"
            if l4_col not in df_l4.columns or du_col not in df_du.columns:
                continue

            merged = pd.merge(df_tg, df_l3[["time", l3_col]], on="time", how="inner")
            merged = pd.merge(merged, df_l4[["time", l4_col]], on="time", how="inner")
            merged = pd.merge(merged, df_du[["time", du_col]], on="time", how="inner")
            if merged.empty:
                continue

            cols = ["SLEV_detided_mdac", l3_col, l4_col, du_col]
            d = p4_filter_smooth(merged, cols, smooth=smooth)
            if len(d) < 8:
                continue

            tg = d["SLEV_detided_mdac"].values * 100.0
            l3v = d[l3_col].values * 100.0
            l4v = d[l4_col].values * 100.0
            duv = d[du_col].values * 100.0

            rmsd_l3 = float(np.sqrt(np.mean((tg - l3v) ** 2)))
            rmsd_l4 = float(np.sqrt(np.mean((tg - l4v) ** 2)))
            rmsd_duacs = float(np.sqrt(np.mean((tg - duv) ** 2)))
            corr_l3 = float(np.corrcoef(tg, l3v)[0, 1])
            corr_l4 = float(np.corrcoef(tg, l4v)[0, 1])
            corr_duacs = float(np.corrcoef(tg, duv)[0, 1])

            if max(rmsd_l3, rmsd_l4, rmsd_duacs) > RMSD_CAP:
                continue

            records.append(dict(
                station=sid, latitude=lat_tg, longitude=lon_tg, radius=r, n=len(d),
                rmsd_l3=rmsd_l3, rmsd_l4=rmsd_l4, rmsd_duacs=rmsd_duacs,
                corr_l3=corr_l3, corr_l4=corr_l4, corr_duacs=corr_duacs,
                # PI = 100*(baseline - test)/baseline, positivo = test mejor
                pi_l3_vs_duacs=100.0 * (rmsd_duacs - rmsd_l3) / rmsd_duacs if rmsd_duacs else np.nan,
                pi_l4_vs_duacs=100.0 * (rmsd_duacs - rmsd_l4) / rmsd_duacs if rmsd_duacs else np.nan,
                pi_l3_vs_l4=100.0 * (rmsd_l4 - rmsd_l3) / rmsd_l4 if rmsd_l4 else np.nan,
            ))

    return pd.DataFrame(records)


def summarize_3products(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in RADII:
        d = df[df.radius == r]
        if len(d) < 5:
            continue
        rows.append(dict(
            radius=r, n=len(d),
            median_rmsd_l3=d.rmsd_l3.median(), median_rmsd_l4=d.rmsd_l4.median(),
            median_rmsd_duacs=d.rmsd_duacs.median(),
            mean_rmsd_l3=d.rmsd_l3.mean(), mean_rmsd_l4=d.rmsd_l4.mean(),
            mean_rmsd_duacs=d.rmsd_duacs.mean(),
            median_corr_l3=d.corr_l3.median(), median_corr_l4=d.corr_l4.median(),
            median_corr_duacs=d.corr_duacs.median(),
            median_pi_l4_vs_duacs=d.pi_l4_vs_duacs.median(),   # valor anadido KaRIn (L4 vs DUACS)
            median_pi_l3_vs_duacs=d.pi_l3_vs_duacs.median(),   # L3 crudo vs DUACS
            median_pi_l3_vs_l4=d.pi_l3_vs_l4.median(),         # L3 vs L4 (ya reportado antes)
            pct_l4_better_than_duacs=100.0 * (d.rmsd_l4 < d.rmsd_duacs).mean(),
            pct_l3_better_than_duacs=100.0 * (d.rmsd_l3 < d.rmsd_duacs).mean(),
        ))
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in RADII:
        d = df[df.radius == r]
        if len(d) < 5:
            continue
        pi = d.pi_rmsd.dropna()
        w_stat, w_p = (np.nan, np.nan)
        if len(d) >= 5:
            try:
                w_stat, w_p = stats.wilcoxon(d.rmsd_l4, d.rmsd_l3)
            except Exception:
                pass
        rows.append(dict(
            radius=r, n=len(d),
            median_rmsd_l3=d.rmsd_l3.median(), median_rmsd_l4=d.rmsd_l4.median(),
            mean_rmsd_l3=d.rmsd_l3.mean(), mean_rmsd_l4=d.rmsd_l4.mean(),
            median_pi=pi.median(), mean_pi=pi.mean(),
            pct_l3_better=100.0 * (d.rmsd_l3 < d.rmsd_l4).mean(),
            wilcoxon_p=w_p,
        ))
    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def _load_or_build(cache_name: str, builder, *args):
    cache_path = OUT_DIR / cache_name
    if cache_path.exists():
        print(f"  -> usando cache existente ({cache_path})")
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)
    result = builder(*args)
    with open(cache_path, "wb") as fh:
        pickle.dump(result, fh)
    print(f"  -> cache guardada en {cache_path}")
    return result


def main():
    stations = build_near_panel()
    stations.to_csv(OUT_DIR / "near_panel_stations.csv", index=False)

    print("\n[STEP] Extraccion L3 (esto puede tardar) ...")
    l3 = _load_or_build("l3_cache.pkl", extract_l3, stations)

    print("\n[STEP] TG detidado (UTide) + DAC (radio fijo 50 km) ...")
    tg_dac = _load_or_build("tg_dac_cache.pkl", build_tg_dac, stations)

    print("\n[STEP] Extraccion L4 MIOST-K (mediana en R km, malla nativa) ...")
    l4 = _load_or_build("l4_cache.pkl", extract_l4, stations)

    pd.set_option("display.width", 200)

    print("\n[STEP] Merge + filtro P4 (3-sigma) + LOESS 15d + metricas por estacion/radio ...")
    per_sr = compute_metrics(stations, l3, tg_dac, l4, smooth=True)
    per_sr.to_csv(OUT_DIR / "per_station_radius.csv", index=False, float_format="%.5f")
    print(f"  -> {len(per_sr)} filas (estacion x radio) guardadas")

    summary = summarize(per_sr)
    summary.to_csv(OUT_DIR / "summary_by_radius.csv", index=False, float_format="%.4f")
    print("\n" + "=" * 100)
    print("RESUMEN [LOESS 15d, comparable a Tablas 2-9]: L3 vs L4 vs TG -- CalVal, near-swath")
    print("=" * 100)
    print(summary.to_string(index=False))

    print("\n[STEP] Version SIN filtrado temporal (solo 3-sigma + recentrado, sin LOESS) ...")
    per_sr_raw = compute_metrics(stations, l3, tg_dac, l4, smooth=False)
    per_sr_raw.to_csv(OUT_DIR / "per_station_radius_raw.csv", index=False, float_format="%.5f")
    print(f"  -> {len(per_sr_raw)} filas (estacion x radio) guardadas")

    summary_raw = summarize(per_sr_raw)
    summary_raw.to_csv(OUT_DIR / "summary_by_radius_raw.csv", index=False, float_format="%.4f")
    print("\n" + "=" * 100)
    print("RESUMEN [SIN LOESS, valores diarios crudos]: L3 vs L4 vs TG -- CalVal, near-swath")
    print("=" * 100)
    print(summary_raw.to_string(index=False))

    print("\n[STEP] Extraccion DUACS-MY (nadir puro, sin KaRIn) ...")
    duacs = _load_or_build("duacs_cache.pkl", extract_duacs_my, stations)

    print("\n[STEP] 3 productos {L3, MIOST-K, DUACS-MY} + filtro P4 (3-sigma) + LOESS 15d ...")
    per_sr_3p = compute_metrics_3products(stations, l3, tg_dac, l4, duacs, smooth=True)
    per_sr_3p.to_csv(OUT_DIR / "per_station_radius_3products.csv", index=False, float_format="%.5f")
    print(f"  -> {len(per_sr_3p)} filas (estacion x radio) guardadas")

    summary_3p = summarize_3products(per_sr_3p)
    summary_3p.to_csv(OUT_DIR / "summary_by_radius_3products.csv", index=False, float_format="%.4f")
    print("\n" + "=" * 100)
    print("RESUMEN [LOESS 15d, comparable a Tablas 2-9]: L3 vs MIOST-K vs DUACS-MY vs TG -- "
          "CalVal, near-swath")
    print("=" * 100)
    print(summary_3p.to_string(index=False))

    print(f"\n[DONE] Resultados en: {OUT_DIR}")


if __name__ == "__main__":
    main()
