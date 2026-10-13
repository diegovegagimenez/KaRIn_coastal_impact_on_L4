#!/usr/bin/env python3
"""
P1_altimetry_tidegauge_processing.py
================================================================================
USO PARA EL ARTÍCULO (dos escenarios; mismo script, distintos argumentos)
--------------------------------------------------------------------------------
  A) PERIODO COMÚN (eje central; los 3 productos sobre los mismos días).
     Repetir para 30/50/70 km, cada uno en su carpeta:

       python P1_altimetry_tidegauge_processing.py --fixed-radius-km 50 \\
              --skip-per-station --skip-per-region

     (--science-end por defecto = abierto; la triple intersección está ACTIVA por
      defecto, así que Science se cierra por la cobertura común de MY-MIOST.)

  B) PERIODO EXTENDIDO sin NRT (solo 50 km; comparación operacional MY vs
     MIOST_KaRIn hasta el fin del segmento DT):

       python P1_altimetry_tidegauge_processing.py --fixed-radius-km 50 \\
              --skip-per-station --skip-per-region \\
              --science-end 2025-07-06 --operational-window

     --science-end 2025-07-06  -> trunca Science al último día DT (excluye NRT).
     --operational-window      -> my_miost entra con join externo y NO recorta a
                                  MY/MIOST_KaRIn, permitiéndoles cubrir la ventana
                                  larga. La comparación válida en esta corrida es
                                  total_effect (MY -> MIOST_KaRIn); karin_effect
                                  queda limitado por la cobertura corta de MY-MIOST.

VERIFICACIÓN tras cada corrida: comprobar con check_coverage_periods.py que en
Science las fechas de fin de los tres productos son las esperadas (idénticas en A;
MIOST_KaRIn hasta 2025-07-06 en B).
================================================================================
(Renombrado y extendido a partir de P0c_radius_sensitivity_commongrid_Cipollini_r400.py)

Genera las series temporales procesadas de altimetría y mareógrafo (TG), y
las métricas RMSD/correlación resultantes, para TRES metodologías de radio
de extracción, en una sola ejecución:

  (1) RADIO FIJO       — un único radio (por defecto 50 km, igual que el P1
                          original v7 FINAL) aplicado a TODAS las estaciones.
                          No requiere selección: es simplemente la fila del
                          barrido de sensibilidad en radius=--fixed-radius-km.
  (2) RADIO POR ESTACIÓN — método de Cipollini et al. (2017): para CADA
                          mareógrafo, el radio que minimiza la RMSD frente
                          al mareógrafo (ya implementado en la versión
                          original de este script).
  (3) RADIO POR REGIÓN  — NUEVO. Cada estación usa la MEDIANA REGIONAL
                          (región CMEMS, vía shapefile IHO v2) de los radios
                          óptimos per-station de su región, snapeada al grid
                          de 5 km del barrido de sensibilidad. Estaciones
                          cuya región no se puede asignar (fuera de las 6
                          regiones CMEMS mapeadas) se EXCLUYEN de esta salida.

Las tres metodologías se calculan a partir del MISMO barrido de sensibilidad
(rows / merged_store), sin re-extraer datos ni re-procesar mareógrafos —
por defecto SE EMITEN LAS TRES en la misma ejecución (usa --skip-fixed,
--skip-per-station, --skip-per-region para desactivar alguna).

Versión unificada con DOS características configurables e independientes:
  (A) REGRIDDING a malla común: todos los productos se llevan a la misma resolución
      espacial antes de extraer, para que la comparación KaRIn (MY-MIOST vs
      MIOST_KaRIn) no mezcle sensor con resolución de malla (0.0625° vs 0.125°).
      Por defecto: downsampling de los finos a 0.125° por MEDIA DE ÁREA conservativa.
      Opciones: --common-grid {coarsest|finest|<paso°>}, --regrid-method
      {conservative|bilinear|nearest}.
  (B) SELECCIÓN del radio "cabecera" (la que aparece en optimal_radius.json /
      consola como resumen ejecutivo): por defecto per_station (Cipollini
      2017); alternativamente global (un único radio auto-optimizado — NO
      confundir con "radio fijo", que es un valor impuesto, no optimizado)
      o per_region. Esto NO afecta a qué métricas se EMITEN (ver arriba: las
      tres se emiten por defecto), solo a cuál se imprime como "la elegida".

Para cada estación genera la serie temporal de cada producto barriendo radios de
5 a 400 km (paso 5), calcula las métricas (RMSD, correlación; raw y LOESS) frente
al mareógrafo, agrega por radio y elige el radio óptimo de forma defendible.

  · Réplica FIEL del pipeline de P1 v7 FINAL (QC estricto del TG, UTide, DAC,
    filtro 3σ conjunto, re-centrado, LOESS 15 d, métricas en cm) para que el radio
    elegido sea directamente utilizable en P1→P2.
  · CORRECCIÓN respecto al P1 original: usa una caja fija de ±0.5° alrededor del TG, que
    TRUNCA el disco para radios grandes (>~55 km) y, en longitud, a latitudes altas
    incluso a 50 km. Aquí la caja se dimensiona según el radio MÁXIMO y la latitud
    (consciente de cos(lat)), de modo que el disco nunca se recorta.
  · EFICIENCIA: cada dataset se lee UNA sola vez; para cada estación se extrae el
    cubo de la caja y las distancias una vez, y cada radio sólo cambia el umbral de
    la máscara (mediana espacial barata).
  · Radios pequeños sin píxeles: se detectan (n_celdas=0) y la estación queda sin
    dato a ese radio (no se inventan valores); la cobertura por radio se reporta.
  · NOTA: como la extracción promedia (mediana) sobre un radio (~20–50 km) muy
    superior a ambas mallas, el regridding apenas cambia los números → sirve como
    PRUEBA DE ROBUSTEZ. La comparación del mapeo (MY vs MY-MIOST) ya estaba en malla
    común 0.0625° y no se ve afectada.

CRITERIO DE SELECCIÓN "POR ESTACIÓN" Y "POR REGIÓN" (Cipollini et al. 2017,
Surv. Geophys. 38, 33–57): para CADA mareógrafo se elige el radio que MINIMIZA
la RMSD frente al mareógrafo (referencia = --select-product, por defecto MY →
no se sesga hacia SWOT), y dentro de ese radio se toma la MEDIANA de la
altimetría (robusta a outliers). "Por región" agrega esos radios óptimos
individuales tomando la MEDIANA dentro de cada región CMEMS (asignada vía
shapefile IHO v2 + mapeo REGION_MAP, idéntico al usado en P2/P3).

ENTRADAS (mismas rutas que P1 v7 FINAL; ajusta --base-dir o el bloque CONFIG):
  data/processed_NRT/tide_gauges_nrt/*.pkl        (mareógrafos NRT)
  data/raw/duacs_l4_my/...0625deg...nc            (MY, MIOST estándar/univariante, 0.0625°)
  data/raw/duacs_l4_my_miost/...0625deg...nc       (MY-MIOST, MIOST nadir, 0.0625°)
  data/raw/miost_l4/dt_global_allsat_phy_l4_*.nc  (MIOST_KaRIn, 0.125°, diarios)
  data/raw/dac_l4/dac_dif_*.nc                     (DAC 6-horario, 0.25°) [opcional]
  World_Seas_IHO_v2.shp                            (shapefile IHO, para radio por región)

SALIDAS (en --outdir):
  sensitivity_per_station_radius.csv   long: station×phase×radius×producto
  sensitivity_aggregate.csv            por (phase,radio,producto): medianas/IQR + cobertura
  optimal_radius.json / .txt           resumen del método "cabecera" (--selection-mode)

  optimal_radius_fixed{N}/             ← RADIO FIJO (N=--fixed-radius-km, def. 50)
      overall_metrics_[LOESS_]PARALLEL_{phase}_v7.csv
      time_series_optimal/{phase}/TS_data_{station}_{phase}.csv (si --export-ts)

  optimal_radius_per_station/          ← RADIO POR ESTACIÓN (Cipollini 2017)
      overall_metrics_[LOESS_]PARALLEL_{phase}_v7.csv
      optimal_radius_per_station.csv   radio óptimo de cada mareógrafo
      fig_radius_per_station.png/.pdf
      time_series_optimal/{phase}/TS_data_{station}_{phase}.csv (si --export-ts)

  optimal_radius_per_region/           ← RADIO POR REGIÓN (NUEVO)
      overall_metrics_[LOESS_]PARALLEL_{phase}_v7.csv
      optimal_radius_per_region.csv    región, radio mediano, n estaciones
      radius_assignment_per_region.csv estación → región → radio asignado
      fig_radius_per_region.png/.pdf
      time_series_optimal/{phase}/TS_data_{station}_{phase}.csv (si --export-ts)

Aplicación: usa las overall_metrics_* de la carpeta que corresponda como entrada
de P2 (--radius-mode {fixed|per_station|per_region}).

Dependencias: numpy, pandas, xarray, matplotlib, statsmodels (LOESS), utide (detido),
geopandas, shapely (asignación de región para el modo per_region).
Autor: análisis para D. Vega-Gimenez / SL_TAC, 2026 (réplica de P1 v7 FINAL).
Método de selección por estación/región: Cipollini, Calafat, Jevrejeva, Melet & Prandi (2017).
================================================================================
"""
from __future__ import annotations

import os
import argparse
import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

try:                                   # barra de progreso opcional
    from tqdm import tqdm
except Exception:                      # pragma: no cover
    def tqdm(x, **k):
        return x

try:                                   # necesario solo para --selection-mode/emit per_region
    import geopandas as gpd
    from shapely.geometry import Point
    _HAS_GEOPANDAS = True
except ImportError:                    # pragma: no cover
    _HAS_GEOPANDAS = False

# ============================================================================
# CONFIG (espejo de P1 v7 FINAL). Edita BASE_DIR o pásalo por --base-dir.
# ============================================================================
BASE_DIR_DEFAULT = Path(os.environ.get("KARIN_PROJECT_DIR", "data"))

PHASES = {                              # idénticas a P1
    "CalVal":  ("2023-03-28", "2023-07-05"),
    "Science": ("2023-07-06", None),
}
PRODUCTS = ["my", "my_miost", "miost"]
PRODUCT_SLA_COL = {"my": "sla_my", "my_miost": "sla_my_miost", "miost": "sla_miost"}

TIDAL_CONSTITUENTS = ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1"]
MIN_SPAN_YEARS = 0.25
MAX_GAP_DAYS = 15
MIN_COVERAGE = 0.90
SIGMA_THRESHOLD = 3.0
LOESS_WINDOW = 15
RMSD_PLAUSIBILITY_CAP_CM = 100.0
LAT_MIN, LAT_MAX = 27.0, 72.0
LON_MIN, LON_MAX = -15.0, 42.0
DAC_EPOCH = pd.Timestamp("1950-01-01")
EARTH_R_KM = 6371.0

RADII_DEFAULT = list(range(5, 401, 5))   # 5,10,...,400 km
DAC_RADIUS_DEFAULT = 50.0                # DAC es suave → radio fijo (no se barre)

PRODUCT_STYLE = {                        # color-blind friendly (Wong)
    "my":       {"color": "#E69F00", "label": "MY (MIOST std., nadir)"},
    "my_miost": {"color": "#0072B2", "label": "MY-MIOST (MIOST multivariate, nadir)"},
    "miost":    {"color": "#009E73", "label": "MIOST_KaRIn (MIOST multivariate, +SWOT)"},
}


@dataclass
class Config:
    base_dir: Path = BASE_DIR_DEFAULT
    outdir: Path = Path("results_radius_sensitivity_commongrid_cipollini_r400")
    radii: list = field(default_factory=lambda: list(RADII_DEFAULT))
    dac_radius: float = DAC_RADIUS_DEFAULT
    box_margin: float = 1.25
    # malla común (regridding)
    common_grid: str = "coarsest"   # 'coarsest' (0.125°, recomendado) | 'finest' | float (°)
    regrid_method: str = "conservative"  # 'conservative' (downsample) | 'bilinear' | 'nearest'
    # selección
    selection_mode: str = "per_station"  # 'per_station' (Cipollini 2017) o 'global'
    select_product: str = "my"   # 'my'/'my_miost'/'miost' o 'mean' (promedio de los 3)
    select_on: str = "loess"     # 'loess' (low-pass, por defecto en P3) o 'raw'
    select_metric: str = "rmsd"  # 'rmsd' (minimizar) o 'corr' (maximizar)
    select_phase: str = "both"   # 'CalVal' | 'Science' | 'both'
    tol_frac: float = 0.0        # tolerancia de parsimonia (solo modo global; 0 = mínimo puro)
    min_coverage_frac: float = 0.90
    # salidas
    export_ts: str = "optimal"   # 'none' | 'optimal' | 'all'
    emit_p1_metrics: bool = True
    from_csv: Path = None        # reanudar desde el CSV por-estación (sin re-extracción)
    # --- NUEVO: qué metodologías emitir en esta ejecución (independiente de
    #     selection_mode, que solo decide cuál se imprime como "cabecera") ---
    emit_fixed: bool = True
    emit_per_station: bool = True
    emit_per_region: bool = True
    fixed_radius_km: int = 50
    iho_shp: Path = None          # si None, usa el default de abajo
    # --- Ventana temporal y alineación de los tres productos ---
    # science_end: fecha de fin de la fase Science (YYYY-MM-DD) o None (abierta).
    #   · None                -> Science se cierra por la cobertura común de los
    #                            productos (comportamiento del periodo COMÚN).
    #   · "2025-07-06"        -> trunca Science al último día del segmento DT,
    #                            excluyendo el segmento NRT (periodo EXTENDIDO).
    science_end: str = None
    # triple_intersect: si True, los TRES productos se evalúan sobre EXACTAMENTE
    #   los mismos días (join interno también para my_miost). Es el modo correcto
    #   para el eje central (aislamiento limpio de KaRIn) y evita el artefacto de
    #   ventana que aparece cuando my_miost (demo congelado) cubre menos días que
    #   MY/MIOST_KaRIn. Si False, my_miost entra con join externo (left) y NO
    #   recorta a MY/MIOST_KaRIn -> se usa para la comparación operacional
    #   MY vs MIOST_KaRIn sobre la ventana extendida, donde my_miost es más corto.
    triple_intersect: bool = True

    @property
    def phases(self) -> dict:
        """Definición de fases derivada de la config (permite truncar Science
        sin editar la constante global PHASES)."""
        sci_end = self.science_end if self.science_end else None
        return {
            "CalVal":  ("2023-03-28", "2023-07-05"),
            "Science": ("2023-07-06", sci_end),
        }

    @property
    def paths(self) -> dict:
        b = self.base_dir
        return {
            "tg_dir":         b / "data/processed_NRT/tide_gauges_nrt",
            "duacs_my":       b / "data/raw/duacs_l4_my/duacs_sla_eur_MY_0625deg_swot_period.nc",
            "duacs_my_miost": b / "data/raw/duacs_l4_my_miost/duacs_sla_eur_MY_MIOST_0625deg_swot_period.nc",
            "miost_dir":      b / "data/raw/miost_l4",
            "dac_dir":        b / "data/raw/dac_l4",
            "iho_shp":        self.iho_shp or Path(
                os.environ.get("KARIN_IHO_SHP", os.environ.get("KARIN_PROJECT_DIR", "data") + "/World_Seas_IHO_v2/World_Seas_IHO_v2.shp")),
        }


# ============================================================================
# HELPERS (copiados de P1 v7 FINAL para consistencia exacta)
# ============================================================================
def haversine(lon1, lat1, lon2, lat2):
    lon1, lon2, lat1, lat2 = map(np.deg2rad, [lon1, lon2, lat1, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))


def calculate_metrics(ground_truth, predicted):
    """RMSD (cm), VAR (cm², ×10000), STD (cm), correlación de Pearson."""
    diff = ground_truth - predicted
    N = np.sum(~np.isnan(diff))
    if N < 10:
        return np.nan, np.nan, np.nan, np.nan
    rmsd = np.sqrt(np.nanmean(diff ** 2)) * 100
    var = np.nanvar(diff) * 10_000
    std = np.nanstd(diff) * 100
    if np.nanstd(ground_truth) == 0 or np.nanstd(predicted) == 0:
        corr = np.nan
    else:
        mask = ~np.isnan(ground_truth) & ~np.isnan(predicted)
        corr = np.corrcoef(ground_truth[mask], predicted[mask])[0, 1]
    return rmsd, var, std, corr


def check_data_continuity(times, min_years=MIN_SPAN_YEARS,
                          max_gap_days=MAX_GAP_DAYS, min_coverage=MIN_COVERAGE):
    if len(times) < 2:
        return False
    ts = pd.Series(times).sort_values()
    span = (ts.max() - ts.min()).days / 365.25
    if span < min_years:
        return False
    gaps = ts.diff().dt.days.dropna()
    if gaps.max() > max_gap_days:
        return False
    expected = (ts.max() - ts.min()).days + 1
    if len(ts) / expected < min_coverage:
        return False
    return True


def apply_loess_filter(series, time_numeric, window_days=LOESS_WINDOW):
    """LOESS low-pass (fracción f = W/(T_max−T_min)). Fallback: serie sin filtrar."""
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
    except Exception:
        return series.copy()
    t_range = time_numeric.max() - time_numeric.min()
    if t_range <= 0:
        return series.copy()
    frac = min(max(window_days / t_range, 0.01), 1.0)
    smoothed = lowess(series.values, time_numeric.values, frac=frac, return_sorted=False)
    return pd.Series(smoothed, index=series.index)


def utide_detide(time_vals, slev_vals, latitude):
    """Análisis armónico UTide (filtro 3σ local previo) → SLEV sin marea."""
    try:
        from utide import solve, reconstruct
    except Exception as e:
        raise RuntimeError("Se requiere 'utide' para el detido del mareógrafo.") from e
    try:
        slev = slev_vals.copy().astype(float)
        mu, sig = np.nanmean(slev), np.nanstd(slev)
        slev[np.abs(slev - mu) > SIGMA_THRESHOLD * sig] = np.nan
        t_num = (time_vals - pd.Timestamp("2000-01-01")) / pd.Timedelta("1D")
        coef = solve(t_num.values, slev, lat=latitude, nodal=False, trend=False,
                     constit=TIDAL_CONSTITUENTS, method="ols", conf_int="linear")
        tide = reconstruct(t_num.values, coef)
        return slev_vals - tide.h
    except Exception as e:
        print(f"  [UTide] Error: {e}")
        return None


def box_halfwidths(lat_deg: float, r_km: float, margin: float):
    """Semianchos de la caja (grados) para contener un disco de r_km a esa latitud."""
    dlat = r_km / 111.32 * margin
    coslat = max(np.cos(np.radians(abs(lat_deg))), np.cos(np.radians(80.0)))
    dlon = r_km / (111.32 * coslat) * margin
    return dlat, dlon


# ============================================================================
# PROCESADO DEL MAREÓGRAFO (espejo de process_single_station 4.1–4.6)
# ============================================================================
def process_tg_detided(pkl_file: Path):
    """Devuelve (df[time_d, SLEV_detided] diario, lat, lon) o (None, motivo)."""
    sid = Path(pkl_file).stem
    try:
        tg = pd.read_pickle(pkl_file).reset_index()
        if "TIME" in tg.columns:
            tg = tg.rename(columns={"TIME": "time_d"})
        tg["time_d"] = pd.to_datetime(tg["time_d"])
        if "DEPTH" in tg.columns:
            tg = tg.drop(columns=["DEPTH"])
        lat = float(tg["LATITUDE"].dropna().iloc[0])
        lon = float(tg["LONGITUDE"].dropna().iloc[0])
        tg = tg.dropna(subset=["SLEV"]).copy()
        if "SLEV_QC" in tg.columns:
            tg = tg[tg["SLEV_QC"].isin([1, 2])]
        tg = tg[(tg["SLEV"] > -15.0) & (tg["SLEV"] < 15.0)]
        if len(tg) < 48:
            return None, f"Strict QC (<48): {sid}"
        tg = tg.set_index("time_d").sort_index()
        if tg.index.to_series().diff().median() < pd.Timedelta("1h"):
            tg = tg.resample("1h").mean(numeric_only=True)
        tg = tg.reset_index().dropna(subset=["SLEV"]).copy()
        detided = utide_detide(tg["time_d"], tg["SLEV"].values, lat)
        if detided is None:
            return None, f"UTide failed: {sid}"
        tg["SLEV_detided"] = detided
        tg = (tg.set_index("time_d").resample("1D").mean(numeric_only=True)
                .dropna(subset=["SLEV_detided"]).reset_index())
        if not check_data_continuity(tg["time_d"]):
            return None, f"Continuity (span/gaps/coverage): {sid}"
        return tg[["time_d", "SLEV_detided"]], (lat, lon)
    except Exception as e:
        return None, f"Error {sid}: {e}"


# ============================================================================
# EXTRACCIÓN ALTIMÉTRICA MULTI-RADIO (una sola pasada de lectura por dataset)
# ============================================================================
def build_target_grid(cfg: "Config", paths: dict):
    """Coordenadas 1D (lat, lon) de la malla común destino, recortadas al dominio.
    'coarsest' → malla nativa de MIOST (~0.125°, RECOMENDADO: downsampling de los finos).
    'finest'   → malla nativa de DUACS (~0.0625°). float → malla regular a ese paso."""
    opt = str(cfg.common_grid).strip().lower()
    if opt in ("coarsest", "coarse", "miost"):
        files = sorted(paths["miost_dir"].glob("dt_global_allsat_phy_l4_*.nc"))
        if not files:
            raise SystemExit("common_grid='coarsest' requiere los archivos MIOST.")
        ds = xr.open_dataset(files[0]); tlat = ds["latitude"].values; tlon = ds["longitude"].values; ds.close()
        label = "MIOST native (~0.125°)"
    elif opt in ("finest", "fine", "duacs"):
        ds = xr.open_dataset(paths["duacs_my"]); tlat = ds["latitude"].values; tlon = ds["longitude"].values; ds.close()
        label = "DUACS native (~0.0625°)"
    else:
        sp = float(cfg.common_grid)
        tlat = np.arange(LAT_MIN, LAT_MAX + 1e-9, sp)
        tlon = np.arange(LON_MIN, LON_MAX + 1e-9, sp)
        label = f"regular {sp}°"
    tlat = np.sort(np.asarray(tlat, float)); tlon = np.sort(np.asarray(tlon, float))
    tlat = tlat[(tlat >= LAT_MIN - 1) & (tlat <= LAT_MAX + 1)]
    tlon = tlon[(tlon >= LON_MIN - 1) & (tlon <= LON_MAX + 1)]
    print(f"  Malla común destino: {label}  ({tlat.size}×{tlon.size} celdas, método={cfg.regrid_method})")
    return tlat, tlon


def _assign_idx(src_centers, tgt_centers):
    """Para cada centro fuente, índice del centro destino más cercano (por puntos medios)."""
    if len(tgt_centers) == 1:
        return np.zeros(len(src_centers), dtype=int)
    edges = (tgt_centers[:-1] + tgt_centers[1:]) / 2.0
    return np.clip(np.searchsorted(edges, src_centers), 0, len(tgt_centers) - 1)


def regrid_box(src_lat, src_lon, cube, tgt_lat, tgt_lon, method="conservative"):
    """Regrid del cubo (time,nlat,nlon) de la malla fuente a la destino del box.
    'conservative' = media de área (nanmean de celdas fuente en cada celda destino),
    apropiado para DOWNSAMPLING (ignora NaN de tierra, sin erosión costera).
    'nearest'/'bilinear' = interpolación (para upsampling)."""
    if (src_lat.shape == tgt_lat.shape and src_lon.shape == tgt_lon.shape
            and np.allclose(src_lat, tgt_lat) and np.allclose(src_lon, tgt_lon)):
        return cube                                            # malla idéntica → identidad
    if method == "conservative":
        ilat = _assign_idx(src_lat, tgt_lat)
        ilon = _assign_idx(src_lon, tgt_lon)
        T, ntl, nto = cube.shape[0], len(tgt_lat), len(tgt_lon)
        valid = np.isfinite(cube)
        src0 = np.where(valid, cube, 0.0)
        sum_lat = np.zeros((T, ntl, cube.shape[2])); cnt_lat = np.zeros_like(sum_lat)
        np.add.at(sum_lat, (slice(None), ilat, slice(None)), src0)
        np.add.at(cnt_lat, (slice(None), ilat, slice(None)), valid.astype(float))
        sum_t = np.zeros((T, ntl, nto)); cnt_t = np.zeros_like(sum_t)
        np.add.at(sum_t, (slice(None), slice(None), ilon), sum_lat)
        np.add.at(cnt_t, (slice(None), slice(None), ilon), cnt_lat)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(cnt_t > 0, sum_t / np.where(cnt_t > 0, cnt_t, 1.0), np.nan)
    # nearest / bilinear (sobre el box pequeño)
    from scipy.interpolate import RegularGridInterpolator
    kind = "nearest" if method == "nearest" else "linear"
    Tlon, Tlat = np.meshgrid(tgt_lon, tgt_lat)
    pts = np.column_stack([Tlat.ravel(), Tlon.ravel()])
    out = np.empty((cube.shape[0], len(tgt_lat), len(tgt_lon)))
    for t in range(cube.shape[0]):
        f = RegularGridInterpolator((src_lat, src_lon), cube[t], method=kind,
                                    bounds_error=False, fill_value=np.nan)
        out[t] = f(pts).reshape(len(tgt_lat), len(tgt_lon))
    return out


def _target_box(tgt_lat, tgt_lon, lat_tg, lon_tg, r_max, margin):
    """Celdas de la malla común dentro del box del TG + sus distancias Haversine."""
    dlat, dlon = box_halfwidths(lat_tg, r_max, margin)
    tla = np.where(np.abs(tgt_lat - lat_tg) <= dlat)[0]
    tlo = np.where(np.abs(tgt_lon - lon_tg) <= dlon)[0]
    if tla.size == 0 or tlo.size == 0:
        return None
    blat, blon = tgt_lat[tla], tgt_lon[tlo]
    lon2d, lat2d = np.meshgrid(blon, blat)
    return blat, blon, haversine(lon2d, lat2d, lon_tg, lat_tg)


def _series_from_box(sub_cube, dist2d, radii):
    """sub_cube: (time, nlat, nlon). Devuelve (dict r->serie temporal, dict r->n_celdas)."""
    nt = sub_cube.shape[0]
    flat = sub_cube.reshape(nt, -1)
    dflat = dist2d.ravel()
    series, ncells = {}, {}
    for r in radii:
        m = dflat <= r
        ncells[r] = int(m.sum())
        if m.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                series[r] = np.nanmedian(flat[:, m], axis=1)
        else:
            series[r] = np.full(nt, np.nan)
    return series, ncells


def extract_single_nc_multiradius(nc_path: Path, sla_col: str, tg_coords: dict,
                                  radii: list, margin: float,
                                  tgt_lat=None, tgt_lon=None, method="conservative"):
    """DUACS en NetCDF único. Cada box se REGRILLA a la malla común (tgt_lat/tgt_lon)
    antes de la mediana por radio, de modo que todos los productos comparten resolución."""
    if not nc_path.exists():
        print(f"    [WARN] No existe {sla_col}: {nc_path}")
        return {}, {}
    ds = xr.open_dataset(nc_path)
    da = ds["sla"]
    if set(("time", "latitude", "longitude")).issubset(da.dims):
        da = da.transpose("time", "latitude", "longitude")
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    times = pd.to_datetime(ds["time"].values).normalize()
    r_max = max(radii)
    out, ncells = {}, {}
    for sid, (lat_tg, lon_tg) in tqdm(tg_coords.items(), desc=f"  {sla_col}", ncols=80):
        tb = _target_box(tgt_lat, tgt_lon, lat_tg, lon_tg, r_max, margin)
        if tb is None:
            continue
        blat, blon, dist = tb
        dlat, dlon = box_halfwidths(lat_tg, r_max, margin)
        la = np.where(np.abs(lat - lat_tg) <= dlat)[0]
        lo = np.where(np.abs(lon - lon_tg) <= dlon)[0]
        if la.size == 0 or lo.size == 0:
            continue
        sub = da.isel(latitude=la, longitude=lo).values          # (time,nla,nlo) malla fuente
        cube = regrid_box(lat[la], lon[lo], sub, blat, blon, method)   # → malla común
        series, nc = _series_from_box(cube, dist, radii)
        df = pd.DataFrame({f"{sla_col}|{r}": series[r] for r in radii})
        df["time_d"] = times.values
        out[sid] = df
        ncells[sid] = nc
    ds.close()
    print(f"    -> {len(out)} estaciones con datos {sla_col}")
    return out, ncells


def extract_miost_multiradius(miost_dir: Path, sla_col: str, tg_coords: dict,
                              radii: list, margin: float,
                              tgt_lat=None, tgt_lon=None, method="conservative"):
    """MIOST (0.125°, archivos diarios). Si la malla común ES la de MIOST (caso por
    defecto), el regrid es identidad; si no, cada box se regrilla a la malla común."""
    files = sorted(miost_dir.glob("dt_global_allsat_phy_l4_*.nc"))
    if not files:
        print(f"    [WARN] No hay archivos {sla_col} en {miost_dir}")
        return {}, {}
    ds0 = xr.open_dataset(files[0])
    ref_lat = ds0["latitude"].values
    ref_lon = ds0["longitude"].values
    ds0.close()
    r_max = max(radii)
    blocks, src_coords, tboxes, ncells = {}, {}, {}, {}
    for sid, (lat_tg, lon_tg) in tg_coords.items():
        tb = _target_box(tgt_lat, tgt_lon, lat_tg, lon_tg, r_max, margin)
        if tb is None:
            continue
        blat, blon, dist = tb
        dlat, dlon = box_halfwidths(lat_tg, r_max, margin)
        la = np.where(np.abs(ref_lat - lat_tg) <= dlat)[0]
        lo = np.where(np.abs(ref_lon - lon_tg) <= dlon)[0]
        if la.size == 0 or lo.size == 0:
            continue
        la0, la1, lo0, lo1 = la[0], la[-1] + 1, lo[0], lo[-1] + 1
        s_lat, s_lon = ref_lat[la0:la1], ref_lon[lo0:lo1]
        identity = (s_lat.shape == blat.shape and s_lon.shape == blon.shape
                    and np.allclose(s_lat, blat) and np.allclose(s_lon, blon))
        blocks[sid] = (la0, la1, lo0, lo1)
        src_coords[sid] = (s_lat, s_lon)
        tboxes[sid] = (blat, blon, dist, identity)
        ncells[sid] = {r: int((dist.ravel() <= r).sum()) for r in radii}
    recs = {sid: {"time": [], **{r: [] for r in radii}} for sid in blocks}
    for f in tqdm(files, desc=f"  {sla_col} files", ncols=80):
        m = re.search(r"phy_l4_(\d{8})_", f.name)
        if not m:
            continue
        date = pd.Timestamp(m.group(1))
        try:
            ds = xr.open_dataset(f)
            da = ds["sla"].isel(time=0)
            if set(("latitude", "longitude")).issubset(da.dims):
                da = da.transpose("latitude", "longitude")
            full = da.values
            for sid, (la0, la1, lo0, lo1) in blocks.items():
                blk = full[la0:la1, lo0:lo1]
                blat, blon, dist, identity = tboxes[sid]
                if identity:
                    cell_vals = blk
                else:
                    s_lat, s_lon = src_coords[sid]
                    cell_vals = regrid_box(s_lat, s_lon, blk[None, ...], blat, blon, method)[0]
                flat, dflat = cell_vals.ravel(), dist.ravel()
                recs[sid]["time"].append(date)
                for r in radii:
                    mask = dflat <= r
                    if mask.any() and np.any(np.isfinite(flat[mask])):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            recs[sid][r].append(np.nanmedian(flat[mask]))
                    else:
                        recs[sid][r].append(np.nan)
            ds.close()
        except Exception:
            pass
    out = {}
    for sid, d in recs.items():
        if d["time"]:
            df = pd.DataFrame({f"{sla_col}|{r}": d[r] for r in radii})
            df["time_d"] = pd.to_datetime(d["time"])
            out[sid] = df
    print(f"    -> {len(out)} estaciones con datos {sla_col}")
    return out, ncells


def extract_dac_daily(dac_dir: Path, tg_coords: dict, dac_radius: float, margin: float):
    """DAC (6-horario, 0.25°) a radio FIJO → media diaria por estación. Espejo de P1."""
    files = sorted(dac_dir.glob("dac_dif_*.nc"))
    if not files:
        print("    [WARN] Sin archivos DAC (se continúa sin corrección DAC).")
        return {}
    ds0 = xr.open_dataset(files[0])
    ref_lat = ds0["latitude"].values
    raw_lon = ds0["longitude"].values
    ref_lon = ((raw_lon + 180) % 360) - 180
    sort_idx = np.argsort(ref_lon)
    ref_lon = ref_lon[sort_idx]
    ds0.close()
    blocks, dists = {}, {}
    for sid, (lat_tg, lon_tg) in tg_coords.items():
        dlat, dlon = box_halfwidths(lat_tg, dac_radius, margin)
        la = np.where(np.abs(ref_lat - lat_tg) <= dlat)[0]
        lo = np.where(np.abs(ref_lon - lon_tg) <= dlon)[0]
        if la.size == 0 or lo.size == 0:
            continue
        la0, la1, lo0, lo1 = la[0], la[-1] + 1, lo[0], lo[-1] + 1
        lon2d, lat2d = np.meshgrid(ref_lon[lo0:lo1], ref_lat[la0:la1])
        dist = haversine(lon2d, lat2d, lon_tg, lat_tg)
        blocks[sid] = (la0, la1, lo0, lo1)
        dists[sid] = (dist.ravel() <= dac_radius)
    recs = {sid: [] for sid in blocks}
    for f in tqdm(files, desc="  DAC files", ncols=80):
        parts = f.stem.split("_")
        try:
            jday, hour = int(parts[2]), int(parts[3])
        except Exception:
            continue
        date = DAC_EPOCH + pd.Timedelta(days=jday) + pd.Timedelta(hours=hour)
        try:
            ds = xr.open_dataset(f)
            full = ds["dac"].values[:, sort_idx]
            for sid, (la0, la1, lo0, lo1) in blocks.items():
                blk = full[la0:la1, lo0:lo1].ravel()
                mask = dists[sid]
                val = np.nanmedian(blk[mask]) if (mask.any() and np.any(np.isfinite(blk[mask]))) else np.nan
                recs[sid].append({"datetime": date, "dac": val})
            ds.close()
        except Exception:
            pass
    out = {}
    for sid, r in recs.items():
        if r:
            d = pd.DataFrame(r).set_index("datetime").resample("1D").mean().reset_index()
            d = d.rename(columns={"datetime": "time_d"})
            d["time_d"] = pd.to_datetime(d["time_d"]).dt.normalize()
            out[sid] = d[["time_d", "dac"]]
    print(f"    -> {len(out)} estaciones con DAC")
    return out


# ============================================================================
# MÉTRICAS POR (ESTACIÓN, RADIO, FASE)  (espejo de 4.7–4.12)
# ============================================================================
def metrics_station_radius(tg_df, alt_by_prod, dac_df, phase_dates,
                           triple_intersect=True):
    """alt_by_prod: dict prod->Series temporal (a un radio). Devuelve dict métricas o None.

    triple_intersect:
      True  -> my_miost se une con join INTERNO: los tres productos quedan sobre
               exactamente los mismos días (periodo común, aislamiento de KaRIn).
      False -> my_miost se une con join EXTERNO (left): NO recorta a MY/MIOST_KaRIn.
               Se usa en la ventana extendida (comparación operacional MY vs
               MIOST_KaRIn), donde my_miost cubre menos días.
    """
    present = [p for p in PRODUCTS if p in alt_by_prod and alt_by_prod[p] is not None]
    if "my" not in present or "miost" not in present:
        return None  # criterio de P1: requerir al menos MY y MIOST
    # ensamblar altimetría: inner my∩miost siempre; my_miost inner|left según flag
    base = pd.DataFrame({"time_d": alt_by_prod["my"].index,
                         "sla_my": alt_by_prod["my"].values})
    mi = pd.DataFrame({"time_d": alt_by_prod["miost"].index,
                       "sla_miost": alt_by_prod["miost"].values})
    alt = pd.merge(base, mi, on="time_d", how="inner")
    if "my_miost" in present:
        mm = pd.DataFrame({"time_d": alt_by_prod["my_miost"].index,
                           "sla_my_miost": alt_by_prod["my_miost"].values})
        how_mm = "inner" if triple_intersect else "left"
        alt = pd.merge(alt, mm, on="time_d", how=how_mm)
    merged = pd.merge(tg_df[["time_d", "SLEV_detided"]], alt, on="time_d", how="inner")
    if merged.empty:
        return None
    if dac_df is not None and len(dac_df):
        merged = pd.merge(merged, dac_df, on="time_d", how="left")
    if "dac" in merged.columns:
        merged["SLEV_detided_mdac"] = merged["SLEV_detided"] - merged["dac"]
    else:
        merged["SLEV_detided_mdac"] = merged["SLEV_detided"]
    merged["SLEV_detided_mdac"] -= merged["SLEV_detided_mdac"].mean()

    start, end = phase_dates
    mask = merged["time_d"] >= pd.Timestamp(start)
    if end is not None:
        mask &= merged["time_d"] <= pd.Timestamp(end)
    merged = merged.loc[mask].copy()
    if len(merged) < 30:
        return None

    sla_cols = [c for c in ["sla_my", "sla_my_miost", "sla_miost"] if c in merged.columns]
    for c in sla_cols:
        merged[c] -= merged[c].mean()
    jm = pd.Series(True, index=merged.index)
    for c in ["SLEV_detided_mdac"] + sla_cols:
        mu, sig = merged[c].mean(), merged[c].std()
        if sig > 0:
            jm &= ~((merged[c] < mu - SIGMA_THRESHOLD * sig) |
                    (merged[c] > mu + SIGMA_THRESHOLD * sig))
    merged = merged.loc[jm].copy()
    for c in ["SLEV_detided_mdac"] + sla_cols:
        merged[c] -= merged[c].mean()
    if len(merged) < 20:
        return None

    out = {"N": len(merged),
           "tg_std": np.std(merged["SLEV_detided_mdac"].values) * 100,
           "tg_var": np.var(merged["SLEV_detided_mdac"].values) * 10_000}
    rmsd_vals = []
    for p in PRODUCTS:
        col = f"sla_{p}"
        if col in merged.columns:
            rmsd, var, std, corr = calculate_metrics(
                merged["SLEV_detided_mdac"].values, merged[col].values)
            out[f"rmsd_{p}"], out[f"var_{p}"], out[f"std_{p}"], out[f"corr_{p}"] = rmsd, var, std, corr
            if not np.isnan(rmsd):
                rmsd_vals.append(rmsd)
        else:
            out[f"rmsd_{p}"] = out[f"var_{p}"] = out[f"std_{p}"] = out[f"corr_{p}"] = np.nan
    if rmsd_vals and max(rmsd_vals) > RMSD_PLAUSIBILITY_CAP_CM:
        return None  # desalineamiento, no variabilidad real

    merged = merged.sort_values("time_d").copy()
    merged["time_num"] = (merged["time_d"] - merged["time_d"].min()).dt.days.astype(float)
    for c in ["SLEV_detided_mdac"] + sla_cols:
        try:
            merged[f"{c}_loess"] = apply_loess_filter(merged[c], merged["time_num"])
        except Exception:
            merged[f"{c}_loess"] = merged[c]
    out["tg_std_loess"] = np.std(merged["SLEV_detided_mdac_loess"].values) * 100
    out["tg_var_loess"] = np.var(merged["SLEV_detided_mdac_loess"].values) * 10_000
    for p in PRODUCTS:
        col = f"sla_{p}_loess"
        if col in merged.columns:
            rmsd, var, std, corr = calculate_metrics(
                merged["SLEV_detided_mdac_loess"].values, merged[col].values)
            out[f"rmsd_{p}_loess"], out[f"var_{p}_loess"], out[f"std_{p}_loess"], out[f"corr_{p}_loess"] = \
                rmsd, var, std, corr
        else:
            out[f"rmsd_{p}_loess"] = out[f"var_{p}_loess"] = out[f"std_{p}_loess"] = out[f"corr_{p}_loess"] = np.nan
    out["_merged"] = merged   # para exportar series en el radio óptimo
    return out


# ============================================================================
# SELECCIÓN DEL RADIO ÓPTIMO
# ============================================================================
# ============================================================================
# ASIGNACIÓN DE REGIÓN CMEMS (idéntica a P2/P3, para consistencia entre scripts)
# Solo se usa si selection_mode/emit_per_region requieren región.
# ============================================================================
REGION_MAP = {
    'Mediterranean Sea - Western Basin': 'Mediterranean Sea',
    'Balearic (Iberian Sea)': 'Mediterranean Sea',
    'Alboran Sea': 'Mediterranean Sea',
    'Ligurian Sea': 'Mediterranean Sea',
    'Tyrrhenian Sea': 'Mediterranean Sea',
    'Strait of Gibraltar': 'Mediterranean Sea',
    'Mediterranean Sea - Eastern Basin': 'Mediterranean Sea',
    'Adriatic Sea': 'Mediterranean Sea',
    'Ionian Sea': 'Mediterranean Sea',
    'Aegean Sea': 'Mediterranean Sea',
    'Sea of Marmara': 'Mediterranean Sea',
    'Baltic Sea': 'Baltic Sea',
    'Gulf of Finland': 'Baltic Sea',
    'Gulf of Bothnia': 'Baltic Sea',
    'Gulf of Riga': 'Baltic Sea',
    'North Sea': 'North West Shelf',
    'Skagerrak': 'North West Shelf',
    'Kattegat': 'North West Shelf',
    'English Channel': 'North West Shelf',
    "Irish Sea and St. George's Channel": 'North West Shelf',
    'Bristol Channel': 'North West Shelf',
    'Inner Seas off the West Coast of Scotland': 'North West Shelf',
    'Celtic Sea': 'Iberia-Biscay-Ireland',
    'Bay of Biscay': 'Iberia-Biscay-Ireland',
    'North Atlantic Ocean': 'Iberia-Biscay-Ireland',
    'Norwegian Sea': 'Arctic',
    'Greenland Sea': 'Arctic',
    'Barentsz Sea': 'Arctic',
    'Black Sea': 'Black Sea',
}
REGION_ORDER = ["Arctic", "Baltic Sea", "Black Sea", "Iberia-Biscay-Ireland",
               "Mediterranean Sea", "North West Shelf"]
UNKNOWN_REGION_LABEL = "Unknown/Outside"

_GDF_IHO = None  # cache del shapefile, cargado una sola vez


def _load_iho_shapefile(iho_shp: Path):
    global _GDF_IHO
    if _GDF_IHO is None:
        if not _HAS_GEOPANDAS:
            raise ImportError(
                "geopandas/shapely no disponibles pero se pidió asignación "
                "de región (per_region). Instala con: pip install geopandas shapely"
            )
        print("  [INFO] Cargando shapefile IHO v2...")
        _GDF_IHO = gpd.read_file(iho_shp).to_crs("EPSG:4326")
    return _GDF_IHO


def assign_regions_batch(df: pd.DataFrame, iho_shp: Path) -> pd.Series:
    """Asigna la región CMEMS a TODAS las estaciones de un DataFrame de una vez
    (sjoin_nearest, índice espacial R-tree). Idéntico método al usado en P2/P3.
    Estaciones cuyo polígono IHO más cercano no está en REGION_MAP reciben la
    etiqueta UNKNOWN_REGION_LABEL en vez del nombre IHO crudo (para que el
    downstream sepa que deben excluirse del modo per_region).
    """
    gdf_seas = _load_iho_shapefile(iho_shp)
    pts = gpd.GeoDataFrame(
        df[["latitude", "longitude"]].copy(),
        geometry=[Point(xy) for xy in zip(df["longitude"], df["latitude"])],
        crs="EPSG:4326",
    )
    joined = gpd.sjoin_nearest(
        pts, gdf_seas[["geometry", "NAME"]], how="left", distance_col="_dist_sea"
    )
    joined = joined[~joined.index.duplicated(keep="first")]
    regions = joined["NAME"].map(lambda n: REGION_MAP.get(n, UNKNOWN_REGION_LABEL))
    return regions.reindex(df.index)


def _metric_col(cfg: Config) -> str:
    suf = "_loess" if cfg.select_on == "loess" else ""
    return f"{cfg.select_metric}_{cfg.select_product}{suf}"


def select_optimal_radius(df: pd.DataFrame, cfg: Config) -> dict:
    col = _metric_col(cfg)
    maximize = (cfg.select_metric == "corr")
    phases = cfg.phases.keys() if cfg.select_phase == "both" else [cfg.select_phase]
    phases = [p for p in phases if p in df["phase"].unique()]
    radii = sorted(int(r) for r in df["radius"].unique())

    # cobertura por (fase, radio) = nº estaciones con métrica finita
    def coverage(ph, r):
        sub = df[(df["phase"] == ph) & (df["radius"] == r)]
        return sub[col].notna().sum()

    # radios elegibles: cobertura >= frac * max cobertura, en TODAS las fases usadas
    eligible = []
    for r in radii:
        ok = True
        for ph in phases:
            covs = [coverage(ph, rr) for rr in radii]
            if max(covs) == 0 or coverage(ph, r) < cfg.min_coverage_frac * max(covs):
                ok = False
                break
        if ok:
            eligible.append(r)
    if not eligible:
        eligible = radii  # fallback

    # panel = estaciones con métrica finita en TODOS los radios elegibles y todas las fases usadas
    panel = None
    for ph in phases:
        for r in eligible:
            sub = df[(df["phase"] == ph) & (df["radius"] == r)]
            valid = set(sub.loc[sub[col].notna(), "station"])
            panel = valid if panel is None else (panel & valid)
    panel = panel or set()

    # objetivo por radio = media (sobre fases) de la mediana del panel
    objective = {}
    for r in eligible:
        vals_per_phase = []
        for ph in phases:
            sub = df[(df["phase"] == ph) & (df["radius"] == r) & (df["station"].isin(panel))]
            if len(sub):
                vals_per_phase.append(np.nanmedian(sub[col].values))
        if vals_per_phase:
            objective[r] = float(np.mean(vals_per_phase))
    if not objective:
        raise RuntimeError("No se pudo evaluar el objetivo de selección (panel vacío).")

    best_val = (max if maximize else min)(objective.values())
    # parsimonia: radios a < tol del óptimo → elegir el más pequeño
    tol = abs(best_val) * cfg.tol_frac
    if maximize:
        near = [r for r, v in objective.items() if v >= best_val - tol]
    else:
        near = [r for r, v in objective.items() if v <= best_val + tol]
    r_opt = min(near)

    tie = "pure minimum" if cfg.tol_frac == 0 else f"parsimony tol={cfg.tol_frac:.0%}"
    return {
        "optimal_radius_km": int(r_opt),
        "criterion": f"{'max' if maximize else 'min'} median {col} ({tie})",
        "select_product": cfg.select_product, "select_on": cfg.select_on,
        "select_metric": cfg.select_metric, "select_phase": cfg.select_phase,
        "panel_n_stations": int(len(panel)),
        "eligible_radii_km": [int(x) for x in eligible],
        "objective_by_radius": {int(k): float(round(v, 4)) for k, v in objective.items()},
        "best_objective_value": float(round(best_val, 4)),
    }


# ============================================================================
# AGREGACIÓN Y FIGURA
# ============================================================================
def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ph in df["phase"].unique():
        for r in sorted(df["radius"].unique()):
            sub = df[(df["phase"] == ph) & (df["radius"] == r)]
            rec = {"phase": ph, "radius": r, "n_stations_my": int(sub["rmsd_my"].notna().sum())}
            for p in PRODUCTS:
                for base in ["rmsd", "corr"]:
                    for suf in ["", "_loess"]:
                        c = f"{base}_{p}{suf}"
                        if c in sub.columns:
                            v = sub[c].dropna().values
                            rec[f"{c}_median"] = np.nanmedian(v) if len(v) else np.nan
                            rec[f"{c}_mean"] = np.nanmean(v) if len(v) else np.nan
                            rec[f"{c}_q25"] = np.nanpercentile(v, 25) if len(v) else np.nan
                            rec[f"{c}_q75"] = np.nanpercentile(v, 75) if len(v) else np.nan
            rows.append(rec)
    return pd.DataFrame(rows)


def make_figure(agg: pd.DataFrame, sel: dict, cfg: Config, outdir: Path):
    phases = [p for p in PHASES.keys() if p in agg["phase"].unique()]
    metrics = [("rmsd", "RMSD [cm]"), ("corr", "Correlation $r$")]
    suf = "_loess" if cfg.select_on == "loess" else ""
    r_opt = sel["optimal_radius_km"]
    fig, axes = plt.subplots(len(metrics), len(phases),
                             figsize=(5.2 * len(phases), 3.6 * len(metrics)),
                             squeeze=False)
    for i, (met, ylab) in enumerate(metrics):
        for j, ph in enumerate(phases):
            ax = axes[i][j]
            sub = agg[agg["phase"] == ph].sort_values("radius")
            for p in PRODUCTS:
                c = f"{met}_{p}{suf}_median"
                if c in sub.columns:
                    ax.plot(sub["radius"], sub[c], marker="o", ms=4,
                            color=PRODUCT_STYLE[p]["color"], label=PRODUCT_STYLE[p]["label"])
            ax.axvline(r_opt, color="0.3", ls="--", lw=1.2)
            ax.set_xlabel("Extraction radius [km]")
            ax.set_ylabel(f"median {ylab}" + (" (low-pass)" if suf else " (raw)"))
            if i == 0:
                ax.set_title(ph, fontweight="bold")
            ax.grid(True, lw=0.4, alpha=0.5)
            # cobertura en eje gemelo
            axt = ax.twinx()
            axt.plot(sub["radius"], sub["n_stations_my"], color="0.6", lw=1.0, ls=":")
            axt.set_ylabel("n stations (coverage)", color="0.5", fontsize=8)
            axt.tick_params(axis="y", labelsize=7, colors="0.5")
            if i == 0 and j == 0:
                ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"Radius sensitivity — optimal = {r_opt} km "
                 f"(min median {_metric_col(cfg)}, parsimony)", fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = outdir / "fig_radius_sensitivity.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out.name} (+ .pdf)")


def emit_p1_metrics(df_full_rows: list, r_opt: int, outdir: Path):
    """Escribe overall_metrics_[LOESS_]PARALLEL_{phase}_v7.csv al radio óptimo."""
    sub = [r for r in df_full_rows if r["radius"] == r_opt]
    by_phase = {}
    for r in sub:
        by_phase.setdefault(r["phase"], []).append(r)
    mdir = outdir / "optimal_radius"
    mdir.mkdir(parents=True, exist_ok=True)
    for phase, rows in by_phase.items():
        raw_cols = ["station", "latitude", "longitude", "phase", "N", "tg_std", "tg_var"]
        loe_cols = ["station", "latitude", "longitude", "phase", "N", "tg_std_loess", "tg_var_loess"]
        for p in PRODUCTS:
            raw_cols += [f"rmsd_{p}", f"var_{p}", f"std_{p}", f"corr_{p}"]
            loe_cols += [f"rmsd_{p}_loess", f"var_{p}_loess", f"std_{p}_loess", f"corr_{p}_loess"]
        df_raw = pd.DataFrame(rows)[[c for c in raw_cols if c in rows[0]]]
        df_loe = pd.DataFrame(rows)[[c for c in loe_cols if c in rows[0]]]
        df_raw.to_csv(mdir / f"overall_metrics_PARALLEL_{phase}_v7.csv", index=False, float_format="%.6f")
        df_loe.to_csv(mdir / f"overall_metrics_LOESS_PARALLEL_{phase}_v7.csv", index=False, float_format="%.6f")
    print(f"  ✓ overall_metrics_*_v7.csv al radio {r_opt} km → {mdir}")


def export_timeseries(merged_store: dict, r_target, outdir: Path):
    """Exporta TS_data_{station}_{phase}.csv (formato P1) al radio dado."""
    base = outdir / "time_series_optimal"
    n = 0
    for (sid, phase, r), merged in merged_store.items():
        if r != r_target:
            continue
        d = base / phase
        d.mkdir(parents=True, exist_ok=True)
        cols = [c for c in merged.columns if c != "time_num"]
        merged[cols].to_csv(d / f"TS_data_{sid}_{phase}.csv", index=False)
        n += 1
    print(f"  ✓ {n} series temporales exportadas al radio {r_target} km")


# ============================================================================
# ORQUESTACIÓN
# ============================================================================
def _json_default(o):
    """Coacciona tipos numpy a nativos para json.dumps (evita 'int64 not serializable')."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def select_per_station_radii(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Cipollini et al. (2017): para CADA mareógrafo, el radio que minimiza la RMSD
    (referencia = cfg.select_product) frente al mareógrafo, agrupando las fases
    indicadas. Devuelve un radio óptimo por estación (distinto en cada una)."""
    suf = "_loess" if cfg.select_on == "loess" else ""
    maximize = (cfg.select_metric == "corr")
    df = df.copy()
    if cfg.select_product == "mean":
        cols = [f"{cfg.select_metric}_{p}{suf}" for p in PRODUCTS
                if f"{cfg.select_metric}_{p}{suf}" in df.columns]
        df["_selobj"] = df[cols].mean(axis=1, skipna=True)
    else:
        df["_selobj"] = df[f"{cfg.select_metric}_{cfg.select_product}{suf}"]
    if cfg.select_phase != "both":
        df = df[df["phase"] == cfg.select_phase]
    recs = []
    for sid, g in df.groupby("station"):
        pooled = g.groupby("radius")["_selobj"].mean().dropna()   # media sobre fases por radio
        if pooled.empty:
            continue
        r_opt = int(pooled.idxmax() if maximize else pooled.idxmin())
        recs.append({"station": sid,
                     "latitude": float(g["latitude"].iloc[0]),
                     "longitude": float(g["longitude"].iloc[0]),
                     "optimal_radius_km": r_opt,
                     "sel_value": float(pooled.loc[r_opt]),
                     "n_candidate_radii": int(pooled.size),
                     "n_phases": int(g["phase"].nunique())})
    return pd.DataFrame(recs).sort_values("station").reset_index(drop=True)


def summarize_per_station(per: pd.DataFrame, cfg: Config) -> dict:
    r = per["optimal_radius_km"].values.astype(float)
    sufd = "_loess" if cfg.select_on == "loess" else ""
    refname = "mean-of-products" if cfg.select_product == "mean" else cfg.select_product
    return {
        "method": "Cipollini et al. (2017): per-tide-gauge optimal search radius",
        "criterion": f"per-station argmin of {cfg.select_metric}_{refname}{sufd} "
                     f"(median of altimetry within the radius; phases pooled = {cfg.select_phase})",
        "reference_product": cfg.select_product, "select_on": cfg.select_on,
        "select_metric": cfg.select_metric, "select_phase": cfg.select_phase,
        "n_stations": int(len(per)),
        "radius_median_km": float(np.median(r)),
        "radius_mean_km": float(np.mean(r)),
        "radius_q25_km": float(np.percentile(r, 25)),
        "radius_q75_km": float(np.percentile(r, 75)),
        "radius_min_km": int(np.min(r)),
        "radius_max_km": int(np.max(r)),
        "radius_values_km": [int(x) for x in sorted(per["optimal_radius_km"].unique())],
    }


def make_figure_perstation(agg: pd.DataFrame, per: pd.DataFrame, cfg: Config, outdir: Path):
    suf = "_loess" if cfg.select_on == "loess" else ""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    radii_sorted = sorted(per["optimal_radius_km"].unique())
    step = (radii_sorted[1] - radii_sorted[0]) if len(radii_sorted) > 1 else 5
    bins = np.arange(min(radii_sorted) - step / 2, max(radii_sorted) + step, step)
    ax1.hist(per["optimal_radius_km"], bins=bins, color="#0072B2", alpha=0.85, edgecolor="white")
    med = float(np.median(per["optimal_radius_km"]))
    ax1.axvline(med, color="k", ls="--", lw=1.4, label=f"median = {med:.0f} km")
    ax1.set_xlabel("Per-station optimal radius [km]")
    ax1.set_ylabel("Number of tide gauges")
    ax1.set_title(f"Per-station optimal radius (Cipollini 2017), n={len(per)}",
                  fontsize=10, fontweight="bold")
    ax1.legend(fontsize=9); ax1.grid(True, axis="y", lw=0.4, alpha=0.5)
    for p in PRODUCTS:
        c = f"rmsd_{p}{suf}_median"
        if c in agg.columns:
            sub = agg.groupby("radius")[c].mean()
            ax2.plot(sub.index, sub.values, marker="o", ms=4,
                     color=PRODUCT_STYLE[p]["color"], label=PRODUCT_STYLE[p]["label"])
    ax2.axvline(med, color="k", ls="--", lw=1.2)
    ax2.set_xlabel("Extraction radius [km]")
    ax2.set_ylabel("network-median RMSD [cm]" + (" (low-pass)" if suf else " (raw)"))
    ax2.set_title("Context: skill vs radius (whole network)", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8); ax2.grid(True, lw=0.4, alpha=0.5)
    fig.tight_layout()
    out = outdir / "fig_radius_per_station.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out.name} (+ .pdf)")


def emit_p1_metrics_per_station(rows: list, per: pd.DataFrame, outdir: Path):
    """overall_metrics_[LOESS_]PARALLEL_{phase}_v7.csv con CADA estación en SU radio óptimo."""
    ropt = dict(zip(per["station"], per["optimal_radius_km"]))
    by_phase = {}
    for r in rows:
        sid = r["station"]
        if sid in ropt and r["radius"] == ropt[sid]:
            rr = dict(r); rr["radius_km"] = int(ropt[sid])
            by_phase.setdefault(r["phase"], []).append(rr)
    mdir = outdir / "optimal_radius_per_station"
    mdir.mkdir(parents=True, exist_ok=True)
    for phase, prows in by_phase.items():
        raw_cols = ["station", "latitude", "longitude", "phase", "radius_km", "N", "tg_std", "tg_var"]
        loe_cols = ["station", "latitude", "longitude", "phase", "radius_km", "N", "tg_std_loess", "tg_var_loess"]
        for p in PRODUCTS:
            raw_cols += [f"rmsd_{p}", f"var_{p}", f"std_{p}", f"corr_{p}"]
            loe_cols += [f"rmsd_{p}_loess", f"var_{p}_loess", f"std_{p}_loess", f"corr_{p}_loess"]
        pd.DataFrame(prows)[[c for c in raw_cols if c in prows[0]]].to_csv(
            mdir / f"overall_metrics_PARALLEL_{phase}_v7.csv", index=False, float_format="%.6f")
        pd.DataFrame(prows)[[c for c in loe_cols if c in prows[0]]].to_csv(
            mdir / f"overall_metrics_LOESS_PARALLEL_{phase}_v7.csv", index=False, float_format="%.6f")
    print(f"  ✓ overall_metrics_*_v7.csv (radio por estación) → {mdir}")


def export_timeseries_per_station(merged_store: dict, per: pd.DataFrame, outdir: Path):
    ropt = dict(zip(per["station"], per["optimal_radius_km"]))
    base = outdir / "time_series_optimal"; n = 0
    for (sid, phase, r), merged in merged_store.items():
        if sid in ropt and r == ropt[sid]:
            d = base / phase; d.mkdir(parents=True, exist_ok=True)
            cols = [c for c in merged.columns if c != "time_num"]
            merged[cols].to_csv(d / f"TS_data_{sid}_{phase}.csv", index=False)
            n += 1
    print(f"  ✓ {n} series temporales exportadas (radio por estación)")


# ============================================================================
# NUEVO — RADIO ÓPTIMO POR REGIÓN (mediana regional del radio per-station)
# ============================================================================
RADIUS_GRID_STEP_KM = 5   # resolución del barrido (cfg.radii por defecto va de 5 en 5)


def _snap_to_grid(value_km: float, step: int = RADIUS_GRID_STEP_KM) -> int:
    """Redondea al múltiplo de `step` km más cercano (el radio óptimo por
    región puede caer entre dos valores del grid si el nº de estaciones de
    la región es par)."""
    return int(round(value_km / step) * step)


def select_per_region_radii(per_station: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """A partir del DataFrame de radios óptimos POR ESTACIÓN (Cipollini 2017,
    salida de select_per_station_radii), asigna la región CMEMS de cada
    estación y calcula el radio de asignación REGIONAL = mediana de los
    radios óptimos individuales de esa región, snapeada al grid de 5 km.

    Devuelve un DataFrame con una fila por estación (excepto las de región
    no mapeada, ver UNKNOWN_REGION_LABEL) con columnas:
        station, latitude, longitude, CMEMS_Region,
        own_optimal_radius_km   (el radio individual de Cipollini, informativo)
        region_radius_km        (el radio que se USA — mediana regional)
        n_stations_in_region
    """
    per = per_station.copy()
    per["CMEMS_Region"] = assign_regions_batch(per, cfg.paths["iho_shp"])

    n_unknown = int((per["CMEMS_Region"] == UNKNOWN_REGION_LABEL).sum())
    if n_unknown:
        print(f"  [per_region] {n_unknown} estaciones con región no mapeada "
              f"('{UNKNOWN_REGION_LABEL}') — se EXCLUIRÁN del modo per_region.")

    per_valid = per[per["CMEMS_Region"] != UNKNOWN_REGION_LABEL].copy()

    region_median = (
        per_valid.groupby("CMEMS_Region")["optimal_radius_km"]
        .median()
        .apply(_snap_to_grid)
        .rename("region_radius_km")
    )

    print("  [per_region] Radio mediano por región (snap-to-5km):")
    for reg, r in region_median.sort_values().items():
        n = int((per_valid["CMEMS_Region"] == reg).sum())
        print(f"      {reg:25s}  n={n:3d}  radius = {r} km")

    out = per_valid.merge(region_median, on="CMEMS_Region", how="left")
    out = out.rename(columns={"optimal_radius_km": "own_optimal_radius_km"})
    out["n_stations_in_region"] = out.groupby("CMEMS_Region")["station"].transform("count")
    keep_cols = ["station", "latitude", "longitude", "CMEMS_Region",
                "own_optimal_radius_km", "region_radius_km", "n_stations_in_region"]
    return out[keep_cols].sort_values("station").reset_index(drop=True)


def summarize_per_region(per_region: pd.DataFrame, cfg: Config) -> dict:
    r = per_region["region_radius_km"].values.astype(float)
    return {
        "method": "Median of per-station Cipollini (2017) optimal radii, "
                 "grouped by CMEMS region (IHO v2 shapefile), snapped to "
                 f"the nearest {RADIUS_GRID_STEP_KM} km grid point.",
        "n_stations": int(len(per_region)),
        "n_regions": int(per_region["CMEMS_Region"].nunique()),
        "radius_median_km_station_weighted": float(np.median(r)),
        "by_region": {
            reg: {
                "n_stations": int(len(g)),
                "region_radius_km": float(g["region_radius_km"].iloc[0]),
                "own_radius_median_km": float(g["own_optimal_radius_km"].median()),
                "own_radius_iqr_km": [
                    float(np.percentile(g["own_optimal_radius_km"], 25)),
                    float(np.percentile(g["own_optimal_radius_km"], 75)),
                ],
            }
            for reg, g in per_region.groupby("CMEMS_Region")
        },
    }


def make_figure_perregion(per_region: pd.DataFrame, outdir: Path):
    """Boxplot de radios óptimos individuales por región + el radio regional
    (mediana, snap 5km) elegido para cada una, marcado con una línea."""
    regions_sorted = (
        per_region.groupby("CMEMS_Region")["region_radius_km"]
        .first().sort_values().index.tolist()
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    data = [per_region.loc[per_region["CMEMS_Region"] == r,
                           "own_optimal_radius_km"].values for r in regions_sorted]
    bp = ax.boxplot(data, tick_labels=regions_sorted, patch_artist=True, widths=0.6)
    for patch in bp["boxes"]:
        patch.set_facecolor("#0072B2"); patch.set_alpha(0.6)

    region_radii = [per_region.loc[per_region["CMEMS_Region"] == r,
                                   "region_radius_km"].iloc[0] for r in regions_sorted]
    ax.scatter(range(1, len(regions_sorted) + 1), region_radii, marker="D",
              color="red", s=70, zorder=10, label="Radio regional usado (mediana snap-5km)")

    ax.set_ylabel("Radio óptimo individual [km]")
    ax.set_title("Radio óptimo por estación agrupado por región CMEMS\n"
                 "(◆ rojo = radio finalmente asignado a la región)",
                 fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, axis="y", alpha=0.4, linewidth=0.5)
    plt.tight_layout()
    out = outdir / "fig_radius_per_region.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out.name} (+ .pdf)")


def emit_p1_metrics_per_region(rows: list, per_region: pd.DataFrame, outdir: Path):
    """overall_metrics_[LOESS_]PARALLEL_{phase}_v7.csv con cada estación en el
    radio MEDIANO de SU REGIÓN (no en su propio óptimo individual)."""
    ropt = dict(zip(per_region["station"], per_region["region_radius_km"]))
    by_phase = {}
    for r in rows:
        sid = r["station"]
        if sid in ropt and r["radius"] == ropt[sid]:
            rr = dict(r); rr["radius_km"] = int(ropt[sid])
            by_phase.setdefault(r["phase"], []).append(rr)
    mdir = outdir / "optimal_radius_per_region"
    mdir.mkdir(parents=True, exist_ok=True)
    for phase, prows in by_phase.items():
        raw_cols = ["station", "latitude", "longitude", "phase", "radius_km", "N", "tg_std", "tg_var"]
        loe_cols = ["station", "latitude", "longitude", "phase", "radius_km", "N", "tg_std_loess", "tg_var_loess"]
        for p in PRODUCTS:
            raw_cols += [f"rmsd_{p}", f"var_{p}", f"std_{p}", f"corr_{p}"]
            loe_cols += [f"rmsd_{p}_loess", f"var_{p}_loess", f"std_{p}_loess", f"corr_{p}_loess"]
        pd.DataFrame(prows)[[c for c in raw_cols if c in prows[0]]].to_csv(
            mdir / f"overall_metrics_PARALLEL_{phase}_v7.csv", index=False, float_format="%.6f")
        pd.DataFrame(prows)[[c for c in loe_cols if c in prows[0]]].to_csv(
            mdir / f"overall_metrics_LOESS_PARALLEL_{phase}_v7.csv", index=False, float_format="%.6f")
    print(f"  ✓ overall_metrics_*_v7.csv (radio por región) → {mdir}")

    # Auditoría estación → región → radio asignado
    per_region.to_csv(mdir / "radius_assignment_per_region.csv", index=False)
    print(f"  ✓ radius_assignment_per_region.csv (auditoría)")


def export_timeseries_per_region(merged_store: dict, per_region: pd.DataFrame, outdir: Path):
    ropt = dict(zip(per_region["station"], per_region["region_radius_km"]))
    base = outdir / "time_series_optimal"; n = 0
    for (sid, phase, r), merged in merged_store.items():
        if sid in ropt and r == ropt[sid]:
            d = base / phase; d.mkdir(parents=True, exist_ok=True)
            cols = [c for c in merged.columns if c != "time_num"]
            merged[cols].to_csv(d / f"TS_data_{sid}_{phase}.csv", index=False)
            n += 1
    print(f"  ✓ {n} series temporales exportadas (radio por región)")


def emit_fixed_radius_metrics(rows: list, r_fixed: int, outdir: Path):
    """overall_metrics_[LOESS_]PARALLEL_{phase}_v7.csv con TODAS las estaciones
    al mismo radio fijo (por defecto 50 km, igual que P1 v7 FINAL original).
    Reutiliza emit_p1_metrics() (idéntica lógica, solo cambia el nombre de la
    carpeta de salida para que las tres metodologías convivan sin pisarse)."""
    mdir = outdir / f"optimal_radius_fixed{r_fixed}"
    emit_p1_metrics(rows, r_fixed, outdir)
    # emit_p1_metrics() escribe en outdir/"optimal_radius" — renombramos/movemos
    # a la carpeta con nombre explícito del radio fijo para que sea inequívoco.
    src = outdir / "optimal_radius"
    if src.exists() and src != mdir:
        mdir.parent.mkdir(parents=True, exist_ok=True)
        if mdir.exists():
            import shutil
            shutil.rmtree(mdir)
        src.rename(mdir)
    print(f"  ✓ overall_metrics_*_v7.csv (radio fijo {r_fixed} km) → {mdir}")


def export_timeseries_fixed(merged_store: dict, r_fixed: int, outdir: Path):
    export_timeseries(merged_store, r_fixed, outdir / f"optimal_radius_fixed{r_fixed}")


def _finalize(df: pd.DataFrame, rows: list, merged_store: dict, cfg: Config):
    """Agregación + las TRES metodologías de radio (fijo / per-station / per-region).

    Por defecto SE EMITEN LAS TRES (cfg.emit_fixed/emit_per_station/emit_per_region,
    todas True), cada una en su propia carpeta bajo cfg.outdir, a partir del MISMO
    barrido de sensibilidad (rows/merged_store) — sin re-extracción.

    cfg.selection_mode ('per_station' | 'per_region' | 'global') decide únicamente
    cuál de ellas se escribe también como "resumen cabecera" en optimal_radius.json
    / optimal_radius.txt en la raíz de cfg.outdir (útil para un vistazo rápido).
    """
    agg = aggregate(df)
    agg.to_csv(cfg.outdir / "sensitivity_aggregate.csv", index=False, float_format="%.6f")
    print("  ✓ sensitivity_aggregate.csv")

    headline = None       # se guarda el summary de la metodología "cabecera"
    headline_kind = None

    # ---- (1) RADIO FIJO ----
    if cfg.emit_fixed:
        print(f"\n{'='*70}\n  METODOLOGÍA 1/3: RADIO FIJO ({cfg.fixed_radius_km} km)\n{'='*70}")
        if cfg.fixed_radius_km not in cfg.radii:
            print(f"  [warn] fixed_radius_km={cfg.fixed_radius_km} no está en el "
                  f"barrido {cfg.radii[0]}-{cfg.radii[-1]} (paso "
                  f"{cfg.radii[1]-cfg.radii[0]}); se omite radio fijo.")
        else:
            if cfg.emit_p1_metrics:
                emit_fixed_radius_metrics(rows, cfg.fixed_radius_km, cfg.outdir)
            if cfg.export_ts != "none":
                export_timeseries_fixed(merged_store, cfg.fixed_radius_km, cfg.outdir)

    # ---- (2) RADIO POR ESTACIÓN (Cipollini 2017) ----
    per_station = None
    if cfg.emit_per_station or cfg.emit_per_region or cfg.selection_mode in ("per_station", "per_region"):
        # per_region necesita per_station como insumo, así que se calcula
        # igualmente aunque emit_per_station=False (solo no se emite el CSV).
        per_station = select_per_station_radii(df, cfg)
        if per_station.empty:
            raise RuntimeError("Ninguna estación tiene un radio válido; revisa datos/radios.")

    if cfg.emit_per_station:
        print(f"\n{'='*70}\n  METODOLOGÍA 2/3: RADIO POR ESTACIÓN (Cipollini 2017)\n{'='*70}")
        per_station.to_csv(cfg.outdir / "optimal_radius_per_station.csv",
                           index=False, float_format="%.4f")
        summary_station = summarize_per_station(per_station, cfg)
        print(f"  >>> mediana {summary_station['radius_median_km']:.0f} km, IQR "
              f"[{summary_station['radius_q25_km']:.0f}, "
              f"{summary_station['radius_q75_km']:.0f}] km, rango "
              f"[{summary_station['radius_min_km']}, {summary_station['radius_max_km']}] km "
              f"(n={summary_station['n_stations']})")
        make_figure_perstation(agg, per_station, cfg, cfg.outdir)
        if cfg.emit_p1_metrics:
            emit_p1_metrics_per_station(rows, per_station, cfg.outdir)
        if cfg.export_ts != "none":
            export_timeseries_per_station(merged_store, per_station, cfg.outdir)
        if cfg.selection_mode == "per_station":
            headline, headline_kind = summary_station, "per_station"

    # ---- (3) RADIO POR REGIÓN (mediana regional del óptimo per-station) ----
    if cfg.emit_per_region:
        print(f"\n{'='*70}\n  METODOLOGÍA 3/3: RADIO POR REGIÓN (mediana regional, "
              f"snap {RADIUS_GRID_STEP_KM} km)\n{'='*70}")
        per_region = select_per_region_radii(per_station, cfg)
        summary_region = summarize_per_region(per_region, cfg)
        print(f"  >>> {summary_region['n_regions']} regiones, "
              f"{summary_region['n_stations']} estaciones asignadas "
              f"(radio mediano ponderado por estación: "
              f"{summary_region['radius_median_km_station_weighted']:.0f} km — "
              f"ver desglose por región en optimal_radius_per_region.json)")
        make_figure_perregion(per_region, cfg.outdir)
        (cfg.outdir / "optimal_radius_per_region.json").write_text(
            json.dumps(summary_region, indent=2, ensure_ascii=False, default=_json_default))
        if cfg.emit_p1_metrics:
            emit_p1_metrics_per_region(rows, per_region, cfg.outdir)
        if cfg.export_ts != "none":
            export_timeseries_per_region(merged_store, per_region,
                                         cfg.outdir / "optimal_radius_per_region")
        if cfg.selection_mode == "per_region":
            headline, headline_kind = summary_region, "per_region"

    # ---- headline opcional: modo 'global' (radio único auto-optimizado,
    #      DISTINTO de 'radio fijo' — este SÍ minimiza una métrica) ----
    if cfg.selection_mode == "global":
        sel = select_optimal_radius(df, cfg)
        make_figure(agg, sel, cfg, cfg.outdir)
        print(f"\n  >>> RADIO ÓPTIMO (global, auto-seleccionado) = "
              f"{sel['optimal_radius_km']} km  ({sel['criterion']})")
        if cfg.emit_p1_metrics:
            emit_p1_metrics(rows, sel["optimal_radius_km"], cfg.outdir / "_global_headline")
        headline, headline_kind = sel, "global"

    # ---- resumen cabecera en la raíz de outdir ----
    if headline is not None:
        (cfg.outdir / "optimal_radius.json").write_text(
            json.dumps({"headline_kind": headline_kind, **headline}, indent=2,
                       ensure_ascii=False, default=_json_default))
        (cfg.outdir / "optimal_radius.txt").write_text(
            f"CABECERA (--selection-mode={cfg.selection_mode}):\n"
            f"{json.dumps(headline, indent=2, ensure_ascii=False, default=_json_default)}\n\n"
            f"Nota: esto es solo un resumen ejecutivo. Las TRES metodologías\n"
            f"(fijo / per_station / per_region) se han emitido por separado en\n"
            f"sus respectivas carpetas bajo {cfg.outdir}, salvo que se hayan\n"
            f"desactivado explícitamente con --skip-fixed/--skip-per-station/\n"
            f"--skip-per-region.\n")
        print(f"\n  ✓ optimal_radius.json/.txt (cabecera = {headline_kind})")

    return {"headline_kind": headline_kind, "headline": headline}


def run(cfg: Config):
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    paths = cfg.paths
    print("=" * 70)
    print(f"  Sensibilidad del radio · {cfg.radii[0]}–{cfg.radii[-1]} km (paso "
          f"{cfg.radii[1]-cfg.radii[0] if len(cfg.radii) > 1 else '-'})")
    print(f"  base_dir = {cfg.base_dir}")
    print("=" * 70)

    if cfg.from_csv is not None:
        print(f"\n[resume] Cargando métricas desde {cfg.from_csv} (sin re-extracción)...")
        df = pd.read_csv(cfg.from_csv)
        if cfg.export_ts != "none":
            print("  (--export-ts requiere ejecución completa; se omite en modo --from-csv)")
            cfg.export_ts = "none"
        return _finalize(df, df.to_dict("records"), {}, cfg)

    # 1) Estaciones
    pkls = sorted(paths["tg_dir"].glob("*.pkl"))
    print(f"\n[1] Mareógrafos: {len(pkls)} pickles")
    tg_coords, tg_detided = {}, {}
    failed = []
    for pk in tqdm(pkls, desc="  TG QC+detide", ncols=80):
        try:
            df = pd.read_pickle(pk)
            lat = float(df["LATITUDE"].dropna().iloc[0])
            lon = float(df["LONGITUDE"].dropna().iloc[0])
        except Exception:
            continue
        if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
            continue
        res, info = process_tg_detided(pk)
        if res is None:
            failed.append(info)
            continue
        sid = pk.stem
        tg_detided[sid] = res
        tg_coords[sid] = info  # (lat, lon)
    print(f"    estaciones válidas (TG): {len(tg_coords)}  (descartadas: {len(failed)})")
    if not tg_coords:
        raise SystemExit("No hay estaciones válidas; revisa --base-dir y los datos.")

    # 2) DAC (radio fijo)
    print(f"\n[2] DAC (radio fijo {cfg.dac_radius:.0f} km)...")
    dac = extract_dac_daily(paths["dac_dir"], tg_coords, cfg.dac_radius, cfg.box_margin)

    # 3) Altimetría multi-radio (todas en malla común)
    print("\n[3] Altimetría multi-radio (regridding a malla común)...")
    tgt_lat, tgt_lon = build_target_grid(cfg, paths)
    alt_my, nc_my = extract_single_nc_multiradius(paths["duacs_my"], "sla_my", tg_coords, cfg.radii, cfg.box_margin, tgt_lat, tgt_lon, cfg.regrid_method)
    alt_mm, nc_mm = extract_single_nc_multiradius(paths["duacs_my_miost"], "sla_my_miost", tg_coords, cfg.radii, cfg.box_margin, tgt_lat, tgt_lon, cfg.regrid_method)
    alt_mi, nc_mi = extract_miost_multiradius(paths["miost_dir"], "sla_miost", tg_coords, cfg.radii, cfg.box_margin, tgt_lat, tgt_lon, cfg.regrid_method)

    # 4) Métricas por (estación, fase, radio)
    print("\n[4] Métricas por estación/fase/radio...")
    rows, merged_store = [], {}
    for sid in tqdm(tg_coords, desc="  metrics", ncols=80):
        lat, lon = tg_coords[sid]
        dac_df = dac.get(sid)
        for phase, pd_dates in cfg.phases.items():
            for r in cfg.radii:
                alt_by_prod = {}
                for prod, store, ncs in [("my", alt_my, nc_my),
                                         ("my_miost", alt_mm, nc_mm),
                                         ("miost", alt_mi, nc_mi)]:
                    if sid in store and ncs.get(sid, {}).get(r, 0) > 0:
                        col = f"{PRODUCT_SLA_COL[prod]}|{r}"
                        s = store[sid][["time_d", col]].dropna()
                        if len(s):
                            alt_by_prod[prod] = pd.Series(s[col].values,
                                                          index=pd.to_datetime(s["time_d"]).dt.normalize())
                m = metrics_station_radius(tg_detided[sid], alt_by_prod, dac_df, pd_dates,
                                           triple_intersect=cfg.triple_intersect)
                if m is None:
                    continue
                merged_store[(sid, phase, r)] = m.pop("_merged")
                row = {"station": sid, "latitude": lat, "longitude": lon,
                       "phase": phase, "radius": r,
                       "ncells_my": nc_my.get(sid, {}).get(r, 0),
                       "ncells_my_miost": nc_mm.get(sid, {}).get(r, 0),
                       "ncells_miost": nc_mi.get(sid, {}).get(r, 0)}
                row.update(m)
                rows.append(row)
    if not rows:
        raise SystemExit("No se produjo ninguna métrica; revisa datos/fases/radios.")
    df = pd.DataFrame(rows)
    keep = [c for c in df.columns if c != "_merged"]
    df[keep].to_csv(cfg.outdir / "sensitivity_per_station_radius.csv", index=False, float_format="%.6f")
    print(f"\n  ✓ sensitivity_per_station_radius.csv ({len(df)} filas)")

    return _finalize(df, rows, merged_store, cfg)


def build_cfg_from_args(argv=None) -> Config:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-dir", type=Path, default=BASE_DIR_DEFAULT)
    ap.add_argument("--outdir", type=Path, default=Path("results_radius_sensitivity_commongrid_cipollini_r400"))
    ap.add_argument("--rmin", type=int, default=5)
    ap.add_argument("--rmax", type=int, default=400)
    ap.add_argument("--rstep", type=int, default=5)
    ap.add_argument("--dac-radius", type=float, default=DAC_RADIUS_DEFAULT)
    ap.add_argument("--box-margin", type=float, default=1.25)
    ap.add_argument("--common-grid", default="coarsest",
                    help="Malla común: 'coarsest' (0.125°, recomendado), 'finest', o paso en grados")
    ap.add_argument("--regrid-method", default="conservative",
                    choices=["conservative", "bilinear", "nearest"],
                    help="conservative=media de área (downsampling); bilinear/nearest=interpolación")
    ap.add_argument("--selection-mode", default="per_station",
                    choices=["per_station", "per_region", "global"],
                    help="Cuál metodología se imprime como 'cabecera' en "
                         "optimal_radius.json/.txt. Esto NO controla qué se "
                         "EMITE (ver --skip-*): por defecto las tres "
                         "(fijo/per_station/per_region) se generan siempre.")
    ap.add_argument("--select-product", default="my", choices=PRODUCTS + ["mean"],
                    help="referencia para elegir el radio: un producto o 'mean' (promedio de los 3)")
    ap.add_argument("--select-on", default="loess", choices=["loess", "raw"])
    ap.add_argument("--select-metric", default="rmsd", choices=["rmsd", "corr"])
    ap.add_argument("--select-phase", default="both", choices=["CalVal", "Science", "both"])
    ap.add_argument("--parsimony-tol", type=float, default=0.0,
                    help="(solo modo global) tolerancia de parsimonia; 0 = mínimo puro (Cipollini)")
    ap.add_argument("--min-coverage-frac", type=float, default=0.90)
    ap.add_argument("--export-ts", default="optimal", choices=["none", "optimal", "all"])
    ap.add_argument("--no-p1-metrics", action="store_true")
    ap.add_argument("--from-csv", type=Path, default=None,
                    help="Reanudar desde sensitivity_per_station_radius.csv (sin re-extracción)")
    # --- NUEVO: control de qué metodologías se EMITEN (por defecto, las 3) ---
    ap.add_argument("--fixed-radius-km", type=int, default=50,
                    help="Radio (km) para la metodología 'radio fijo' (def. 50, "
                         "igual que P1 v7 FINAL original)")
    ap.add_argument("--skip-fixed", action="store_true",
                    help="No emitir la metodología de radio fijo")
    ap.add_argument("--skip-per-station", action="store_true",
                    help="No emitir la metodología de radio óptimo por estación")
    ap.add_argument("--skip-per-region", action="store_true",
                    help="No emitir la metodología de radio óptimo por región "
                         "(requiere geopandas/shapely + shapefile IHO v2)")
    ap.add_argument("--iho-shp", type=Path, default=None,
                    help="Ruta al shapefile IHO v2 (solo si difiere del default "
                         "hardcodeado; necesario para per_region)")
    # --- Ventana temporal y alineación de productos ---
    ap.add_argument("--science-end", type=str, default=None,
                    help="Fin de la fase Science (YYYY-MM-DD). Por defecto abierta "
                         "(la cierra la cobertura común). Usa 2025-07-06 para el "
                         "periodo EXTENDIDO sin NRT (último día del segmento DT).")
    ap.add_argument("--operational-window", action="store_true",
                    help="Desactiva la triple intersección: my_miost entra con "
                         "join externo (left) y NO recorta a MY/MIOST_KaRIn. Úsalo "
                         "SOLO para la comparación operacional MY vs MIOST_KaRIn "
                         "en la ventana extendida (con --science-end 2025-07-06). "
                         "Por defecto (sin este flag) los tres productos se "
                         "intersectan sobre los mismos días -> periodo común.")
    a = ap.parse_args(argv)
    return Config(
        base_dir=a.base_dir, outdir=a.outdir,
        radii=list(range(a.rmin, a.rmax + 1, a.rstep)),
        dac_radius=a.dac_radius, box_margin=a.box_margin,
        common_grid=a.common_grid, regrid_method=a.regrid_method,
        selection_mode=a.selection_mode,
        select_product=a.select_product, select_on=a.select_on,
        select_metric=a.select_metric, select_phase=a.select_phase,
        tol_frac=a.parsimony_tol, min_coverage_frac=a.min_coverage_frac,
        export_ts=a.export_ts, emit_p1_metrics=not a.no_p1_metrics,
        from_csv=a.from_csv,
        fixed_radius_km=a.fixed_radius_km,
        emit_fixed=not a.skip_fixed,
        emit_per_station=not a.skip_per_station,
        emit_per_region=not a.skip_per_region,
        iho_shp=a.iho_shp,
        science_end=a.science_end,
        triple_intersect=not a.operational_window,
    )


def main(argv=None):
    run(build_cfg_from_args(argv))


if __name__ == "__main__":
    main()