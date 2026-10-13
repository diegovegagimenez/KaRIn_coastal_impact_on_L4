#!/usr/bin/env python3
"""
P2_advanced_value_added_analysis.py
================================================================================
USO PARA EL ARTÍCULO (mismo script para 30/50/70 km y para la ventana extendida)
--------------------------------------------------------------------------------
  Fijar SIEMPRE la variable de entorno P2_FIXED_RADIUS_KM al mismo valor que
  --fixed-radius-km (PATHS se construye al importar el módulo, antes de leer
  argparse):

    P2_FIXED_RADIUS_KM=50 python P2_advanced_value_added_analysis.py \\
        --radius-mode fixed --fixed-radius-km 50

  Genera las 3 comparaciones (interpolation_effect, karin_effect, total_effect).
  Para el eje central se usan karin_effect (aislado) y total_effect (control);
  en la corrida de ventana extendida el resultado operacional es total_effect.

CORRECCIONES DE FIGURAS INCLUIDAS EN ESTA VERSIÓN (sin scripts de parche externos):
  · Eliminado el banner _stamp_comparison de todas las figuras.
  · Título de fig07 (baja frecuencia) ya NO dice "HP"; la versión HIGH-PASS sí.
  · Añadida caja "Mean | Median" arriba-izquierda en fig07 (LF), fig07 HIGHPASS
    y fig15 (mapa combinado).
  · Títulos/leyendas que decían "MY"/"MIOST" fijos ahora usan BASELINE_LABEL /
    TEST_LABEL (Taylor, ΔRMSD maps, summary table, difference maps).
  · El "(50 km radius)" hardcodeado en las figuras de variabilidad espacial ahora
    usa FIXED_RADIUS_KM, correcto a 30/50/70.
================================================================================
(Renombrado a partir de P3_advanced_value_added_analysis_v2_THREE_COMPARISONS_optimized.py)

Análisis de valor añadido MY / MY-MIOST / MIOST_KaRIn, PARAMETRIZADO para
las TRES metodologías de radio de extracción que emite P1
(P1_altimetry_tidegauge_processing.py):

    --radius-mode fixed        radio fijo (por defecto 50 km)
    --radius-mode per_station  radio óptimo por estación (Cipollini 2017)
    --radius-mode per_region   radio óptimo por región (mediana regional)

Cada modo apunta PATHS["metrics"] a la carpeta correspondiente que P1 genera
(optimal_radius_fixed{N} / optimal_radius_per_station / optimal_radius_per_region)
y escribe sus resultados en una carpeta de salida separada, de modo que las
tres corridas conviven sin pisarse:

    results_latest/advanced_analysis_v2/{comparison}/                (fixed)
    results_latest/advanced_analysis_v2_per_station/{comparison}/
    results_latest/advanced_analysis_v2_per_region/{comparison}/

INCLUYE (heredado sin cambios del script base, confirmado funcional):
  · Las DOS comparaciones científicas (interpolation_effect, karin_effect) +
    total_effect.
  · Análisis de PROXIMIDAD A TRAZAS SWOT durante CalVal (Near vs Far, umbral
    SWOT_TRACK_DISTANCE_KM=2 km): boxplots PI_RMSD/PI_Corr global
    (plot_calval_swot_impact, vía el parche P3_patch_swot_analysis.py que
    persiste swot_proximity_tests.csv y swot_proximity_by_station.csv) y
    DESGLOSADO POR REGIÓN (plot_calval_swot_impact_by_region).
  · Todo el resto del pipeline P3 original (Taylor diagrams, mapas PI,
    análisis regional, high-pass, etc.)

DEPENDENCIA: P3_patch_swot_analysis.py debe estar en el mismo directorio
(se importa dinámicamente desde plot_calval_swot_impact()).

USO
---
  python P2_advanced_value_added_analysis.py --radius-mode fixed
  python P2_advanced_value_added_analysis.py --radius-mode per_station
  python P2_advanced_value_added_analysis.py --radius-mode per_region

  # Rutas custom (si P1 se corrió con --outdir distinto del default):
  python P2_advanced_value_added_analysis.py --radius-mode per_region \\
      --p1-outdir /ruta/a/results_radius_sensitivity_commongrid_cipollini_r400

================================================================================
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
import xarray as xr
import json
import geopandas as gpd
from shapely.geometry import Point
import concurrent.futures
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# NOMENCLATURA CANÓNICA DE PRODUCTOS
# =============================================================================
# Nombre interno (clave de columna en los CSV)  ->  Nombre para mostrar en
# TODAS las figuras y tablas. Única fuente de verdad para las etiquetas.
PRODUCT_NAMES = {
    "my":       "MY",            # DUACS MY  (mapeo MIOST estándar/univariante, solo nadir)
    "my_miost": "MY-MIOST",      # DUACS MY-MIOST (interpolación MIOST, solo nadir)
    "miost":    "MIOST_KaRIn",   # MIOST (interpolación MIOST, nadir + KaRIn/SWOT)
}

def product_label(internal_name: str) -> str:
    """Devuelve el nombre canónico de un producto a partir de su clave interna."""
    return PRODUCT_NAMES.get(internal_name, internal_name)



# =============================================================================
# CONFIGURACIÓN DE COMPARACIONES (el corazón de la parametrización)
# =============================================================================
# Cada comparación define: baseline, test, etiquetas legibles y carpeta.
# El pipeline completo se ejecuta una vez por cada entrada de COMPARISONS.

# --- NOMENCLATURA (2026-08-11) ---------------------------------------------
# Las claves/carpetas se nombran por el PAR de productos que compara cada
# entrada, no por el "efecto" abstracto. Equivalencia con la nomenclatura
# anterior y con las claves internas de columna:
#
#   producto interno | etiqueta figuras | producto (nomenclatura MIOST)
#   -----------------|------------------|------------------------------
#   my               | MY               | MIOST estándar (univariante), nadir
#   my_miost         | MY-MIOST         | MIOST multivariante, nadir
#   miost            | MIOST_KaRIn      | MIOST multivariante + KaRIn (SWOT)
#
#   nombre nuevo (carpeta)                            | nombre antiguo
#   --------------------------------------------------|--------------------
#   miost__vs__miost_multivariate                      | interpolation_effect
#   miost_multivariate__vs__miost_multivariate_karin   | karin_effect
#   miost__vs__miost_multivariate_karin                | total_effect
#
# P4/P5 (copias en esta misma carpeta) ya apuntan a los nombres nuevos.
COMPARISONS = {
    "miost__vs__miost_multivariate": {
        "baseline": "my",
        "test": "my_miost",
        "baseline_label": "MY",
        "test_label": "MY-MIOST",
        "title": "Mapping configuration: MIOST vs MIOST multivariate (both nadir-only)",
        "short": "Mapping",
        "outdir": "miost__vs__miost_multivariate",
    },
    "miost_multivariate__vs__miost_multivariate_karin": {
        "baseline": "my_miost",
        "test": "miost",
        "baseline_label": "MY-MIOST",
        "test_label": "MIOST_KaRIn",
        "title": "KaRIn (SWOT) added value: MIOST multivariate vs MIOST multivariate + KaRIn",
        "short": "KaRIn",
        "outdir": "miost_multivariate__vs__miost_multivariate_karin",
    },
    "miost__vs__miost_multivariate_karin": {
        "baseline": "my",
        "test": "miost",
        "baseline_label": "MY",
        "test_label": "MIOST_KaRIn",
        "title": "Total change: MIOST vs MIOST multivariate + KaRIn",
        "short": "Total",
        "outdir": "miost__vs__miost_multivariate_karin",
    },
}
    
# Producto de reference común para la variante de PI "vs MY"
COMMON_BASELINE = "my"

# Variables globales reconfiguradas por run_comparison() antes de cada pasada.
# Todo el resto del código las lee en lugar de hardcodear 'my'/'miost'.
BASELINE = "my"
TEST = "my_miost"
BASELINE_LABEL = "MY"
TEST_LABEL = "MY-MIOST"
COMPARISON_TITLE = "Interpolation method effect: MY vs MY-MIOST"
COMPARISON_KEY = "interpolation_effect"

# =============================================================================
# CONFIGURACIÓN DE RUTAS — PARAMETRIZADA POR METODOLOGÍA DE RADIO
# =============================================================================
# Se usan variables de entorno (no solo argparse) porque este script lanza
# subprocesos con multiprocessing 'spawn' (una comparación por proceso); cada
# subproceso reimporta el módulo desde cero, así que la única forma robusta
# de propagar la configuración a los hijos es a través de os.environ (que
# spawn hereda automáticamente), no de sys.argv. El bloque `__main__` al
# final del fichero parsea argparse y, si el usuario pasó --radius-mode /
# --fixed-radius-km / --p1-outdir, actualiza os.environ ANTES de lanzar
# main() (que es cuando se crean los subprocesos), de modo que tanto el
# proceso padre como los hijos vean la misma configuración.
import os

RADIUS_MODE = os.environ.get("P2_RADIUS_MODE", "fixed")            # fixed|per_station|per_region
FIXED_RADIUS_KM = int(os.environ.get("P2_FIXED_RADIUS_KM", "50"))
P1_OUTDIR = Path(os.environ.get(
    "P2_P1_OUTDIR",
    str(Path(__file__).resolve().parent /
        "results_radius_sensitivity_commongrid_cipollini_r400")))

# --- BUG FIX (2026-07-03) ---
# swot_l3 y coverage_csv NO deben derivarse de dónde vive el script P2
# (Path(__file__).resolve().parent) sino del --base-dir que usó P1 para
# descargar/extraer los datos crudos (data/raw/swot_l3_expert vive ahí, NO
# junto al script P2 si P2 se guarda en un subdirectorio como PIPELINE_COMPLETO/).
# Antes: BASE_DIR = Path(__file__).resolve().parent  -> ruta rota si P2 no
# vive en el mismo directorio que fue --base-dir de P1.
DATA_BASE_DIR = Path(os.environ.get(
    "P2_DATA_BASE_DIR",
    os.environ.get("KARIN_PROJECT_DIR", "data")))  # == BASE_DIR_DEFAULT de P1

_RADIUS_MODE_SUBDIR = {
    "fixed": f"optimal_radius_fixed{FIXED_RADIUS_KM}",
    "per_station": "optimal_radius_per_station",
    "per_region": "optimal_radius_per_region",
}
if RADIUS_MODE not in _RADIUS_MODE_SUBDIR:
    raise ValueError(f"RADIUS_MODE inválido: {RADIUS_MODE!r}. "
                     f"Usa: {list(_RADIUS_MODE_SUBDIR)}")

BASE_DIR = Path(__file__).resolve().parent   # solo para output_root (dónde escribir resultados)

# --- NUEVO: Leer variable de entorno para outdir (o usar 'results_latest' por defecto) ---
P2_OUTDIR = Path(os.environ.get("P2_OUTDIR", str(BASE_DIR / "results_latest")))

PATHS = {
    "metrics": P1_OUTDIR / _RADIUS_MODE_SUBDIR[RADIUS_MODE],
    "swot_l3": DATA_BASE_DIR / "data" / "raw" / "swot_l3_expert",
    "ts_data": P1_OUTDIR / _RADIUS_MODE_SUBDIR[RADIUS_MODE] / "time_series_optimal",
    "coverage_csv": DATA_BASE_DIR / "results" / "metrics" / "product_temporal_coverage_v7.csv",
    # --- MODIFICADO: Usar P2_OUTDIR dinámico en lugar del string fijo 'results_latest' ---
    "output_root": P2_OUTDIR / f"advanced_analysis_v2_{RADIUS_MODE}",
    "iho_shp": Path(os.environ.get(
        "P2_IHO_SHP",
        os.environ.get("KARIN_IHO_SHP", os.environ.get("KARIN_PROJECT_DIR", "data") + "/World_Seas_IHO_v2/World_Seas_IHO_v2.shp"))),
}

if not PATHS["swot_l3"].exists():
    print(f"  [WARN] swot_l3 no encontrado en {PATHS['swot_l3']} — el análisis "
          f"de proximidad SWOT (Near/Far) se OMITIRÁ silenciosamente en "
          f"run_comparison() si esto no se corrige. Si tus datos SWOT L3 "
          f"están en otra ruta, usa --data-base-dir o la variable de entorno "
          f"P2_DATA_BASE_DIR.")

if not PATHS["metrics"].exists():
    raise FileNotFoundError(
        f"[FATAL] No se encontró la carpeta de métricas para "
        f"--radius-mode={RADIUS_MODE}:\n"
        f"        {PATHS['metrics']}\n"
        f"        Ejecuta primero P1_altimetry_tidegauge_processing.py "
        f"(que por defecto ya emite las tres metodologías), o pasa "
        f"--p1-outdir si usaste una carpeta de salida distinta en P1."
    )

PATHS["output_root"].mkdir(parents=True, exist_ok=True)

# PATHS['output'] y PATHS['figs_maps'] se fijan dinámicamente en run_comparison()
PATHS["output"] = PATHS["output_root"] / "interpolation_effect"
PATHS["figs_maps"] = PATHS["output"] / "maps"

# Diccionario global para almacenar resultados numéricos (se reinicia por comparación)
RESULTS = {}

def _reset_results():
    global RESULTS
    RESULTS = {
        "comparison": {},
        "improvement_metrics": {},
        "rmsd_analysis": {},
        "improvement_stats": {},
        "regional_stats": {},
        "swot_analysis": {},
        "variability_analysis": {},
        "combined_analysis": {},
        "regression_analysis": {},
    }

_reset_results()

# Fases
PHASES = ["CalVal", "Science"]
PHASE_LABELS = {
    "CalVal": "CalVal (Mar–Jul 2023)",
    "Science": "Science (Jul 2023 onwards)",
}

# Región geográfica
EXTENT = [-12, 42, 30, 67]  # [lon_min, lon_max, lat_min, lat_max]

# Límites de color
PI_CLIM = (-20, 20)        # Percentage Improvement limits: ±20%
RMSD_OUTLIER_CAP = 50.0    # cm

# Colores por región (color-blind friendly)
REGION_COLORS = {
    "Arctic": "#4E79A7",
    "Baltic Sea": "#F28E2B",
    "Black Sea": "#E15759",
    "Iberia-Biscay-Ireland": "#76B7B2",
    "Mediterranean Sea": "#59A14F",
    "North West Shelf": "#B07AA1",
}
REGION_ORDER = list(REGION_COLORS.keys())

# Mapeo explícito de polígonos IHO a Regiones CMEMS agregadas (P7)
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

SERIOUS_COLORS = {
    'Mediterranean Sea': '#1b4f72',
    'Baltic Sea': '#117a65',
    'North West Shelf': '#b9770e',
    'Iberia-Biscay-Ireland': '#7b241c',
    'Arctic': '#515a5a',
    'Black Sea': '#4a235a',
}

# Estética global
FS = 12
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": FS,
    "axes.titlesize": FS + 1,
    "axes.titleweight": "bold",
    "axes.labelsize": FS,
    "xtick.labelsize": FS,
    "ytick.labelsize": FS,
    "legend.fontsize": FS - 1,
    "legend.framealpha": 0.85,
    "figure.facecolor": "white",
})

# Tracks SWOT CalVal
SWOT_CALVAL_PASSES = [1, 3, 14, 16]
SWOT_TRACK_DISTANCE_KM = 2.0
def haversine(lon1, lat1, lon2, lat2):
    """Distancia en km entre puntos."""
    lon1, lon2, lat1, lat2 = map(np.deg2rad, [lon1, lon2, lat1, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * 6371.0 * np.arcsin(np.sqrt(a))


# Variable global para cachear el GeoDataFrame
_GDF_IHO = None

def _load_iho_shapefile():
    global _GDF_IHO
    if _GDF_IHO is None:
        print("[INFO] Cargando shapefile IHO v2...")
        _GDF_IHO = gpd.read_file(PATHS["iho_shp"]).to_crs("EPSG:4326")
    return _GDF_IHO

def assign_region(lat: float, lon: float) -> str:
    """Asigna región CMEMS a un único punto (Point-in-polygon + Nearest fallback).
    Se conserva por compatibilidad, pero para DataFrames usar assign_regions_batch,
    que es ~70x más rápido (índice espacial vectorizado en lugar de bucle Python).
    """
    gdf = _load_iho_shapefile()
    point = Point(lon, lat)

    for idx, row in gdf.iterrows():
        if row['geometry'].contains(point):
            iho_name = row['NAME']
            cmems_region = REGION_MAP.get(iho_name)
            return cmems_region if cmems_region else iho_name

    distances = gdf.geometry.distance(point)
    nearest_idx = distances.idxmin()
    nearest_iho_name = gdf.loc[nearest_idx, 'NAME']

    cmems_region = REGION_MAP.get(nearest_iho_name)
    return cmems_region if cmems_region else nearest_iho_name


def assign_regions_batch(df: pd.DataFrame) -> pd.Series:
    """Asigna la región CMEMS a TODAS las estaciones de un DataFrame de una vez,
    usando sjoin_nearest (índice espacial R-tree) en lugar de un bucle Python.

    Equivale a [assign_region(lat, lon) for ...] pero ~70x más rápido.
    Devuelve una Series alineada con el índice de df.
    """
    gdf_seas = _load_iho_shapefile()

    pts = gpd.GeoDataFrame(
        df[["latitude", "longitude"]].copy(),
        geometry=[Point(xy) for xy in zip(df["longitude"], df["latitude"])],
        crs="EPSG:4326",
    )

    # sjoin_nearest asigna el polígono más cercano (cubre el caso point-in-polygon,
    # donde la distancia es 0, y el fallback nearest en un solo paso).
    joined = gpd.sjoin_nearest(
        pts, gdf_seas[["geometry", "NAME"]], how="left", distance_col="_dist_sea"
    )
    # Un punto puede empatar con dos polígonos: conservar el primero (dist menor).
    joined = joined[~joined.index.duplicated(keep="first")]

    # Mapear nombre IHO -> región CMEMS; si no está en REGION_MAP, conservar el nombre IHO
    regions = joined["NAME"].map(lambda n: REGION_MAP.get(n, n))
    return regions.reindex(df.index)

# =============================================================================
# 0. MAPA DE REGIONES IHO Y UBICACIÓN DE MAREÓGRAFOS
# =============================================================================

def plot_study_area_map(df_combined: pd.DataFrame) -> None:
    """
    Genera la Figura 1 del artículo: Mapa del área de estudio con regiones IHO 
    y ubicación de los mareógrafos utilizados en la validación.
    """
    print("[PROCESSING] Generando mapa del área de estudio...")
    
    # 1. Cargar y filtrar shapefile
    gdf = gpd.read_file(PATHS["iho_shp"]).to_crs("EPSG:4326")
    gdf_eur = gdf.cx[EXTENT[0]:EXTENT[1], EXTENT[2]:EXTENT[3]].copy()
    gdf_eur['CMEMS_Region'] = gdf_eur['NAME'].map(REGION_MAP)
    gdf_eur = gdf_eur.dropna(subset=['CMEMS_Region'])

    # 2. Configurar figura
    proj = ccrs.Mercator()
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': proj})
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())

    # Características del mapa base
    ax.add_feature(cfeature.LAND, facecolor='#d5d8dc', edgecolor='none', zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, color='#2c3e50', zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=':', color='#566573', zorder=2)
    
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, color='#7f8c8d', alpha=0.6, zorder=4)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 14}
    gl.ylabel_style = {'size': 14}

    # 3. Dibujar regiones
    legend_elements = []
    unique_regions = sorted(gdf_eur['CMEMS_Region'].unique())
    
    for region_name in unique_regions:
        region_geom = gdf_eur[gdf_eur['CMEMS_Region'] == region_name]
        color = SERIOUS_COLORS.get(region_name, '#808080')
        
        ax.add_geometries(region_geom.geometry, crs=ccrs.PlateCarree(),
                          facecolor=color, edgecolor='white', linewidth=0.4, 
                          alpha=0.75, zorder=1)
        legend_elements.append(Patch(color=color, alpha=0.75, label=region_name))

    # 4. Dibujar mareógrafos (usando las estaciones únicas del dataframe analizado)
    tg_unique = df_combined.drop_duplicates(subset=['station'])
    tg_scatter = ax.scatter(tg_unique['longitude'], tg_unique['latitude'], 
                            color='black', marker='o', s=40, 
                            edgecolor='white', linewidth=0.6,
                            transform=ccrs.PlateCarree(), zorder=5, 
                            label='Tide Gauge Stations')
    
    legend_elements.append(tg_scatter)

    # 5. Leyenda y guardado
    ax.legend(handles=legend_elements, loc='upper left', 
              title='Oceanic Regions & TG Network', fontsize=11, title_fontsize=12, 
              framealpha=0.95, edgecolor='#2c3e50')

    fname = PATHS["output"] / "fig01_study_area_map"
    plt.tight_layout()
    fig.savefig(fname.with_suffix(".png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved Study Area Map: {fname}.png")

# =============================================================================
# 1. CARGA DE DATOS
# =============================================================================

def load_metrics(phase: str, loess: bool = True) -> pd.DataFrame:
    """Carga las métricas del CSV de P1 v7 y crea ALIAS de columnas para el par
    de comparación activo (BASELINE/TEST) hacia los nombres internos _my/_miost
    que usa el resto del pipeline. Así el código downstream no necesita cambios.

    También conserva las columnas del COMMON_BASELINE (MY) bajo el alias _cmn
    para poder calcular el PI con baseline común además del baseline propio.
    """
    prefix = "LOESS_" if loess else ""
    path = PATHS["metrics"] / f"overall_metrics_{prefix}PARALLEL_{phase}_v7.csv"
    df = pd.read_csv(path)

    if loess:
        df.columns = [c.replace("_loess", "") for c in df.columns]

    # Filtrado de outliers sobre TODAS las columnas rmsd_* presentes
    rmsd_cols = [c for c in df.columns if c.startswith("rmsd_")]
    mask = (df[rmsd_cols] > RMSD_OUTLIER_CAP).any(axis=1)
    n_removed = mask.sum()
    if n_removed:
        print(f"  [{phase}] Eliminadas {n_removed} estaciones con RMSD > {RMSD_OUTLIER_CAP} cm")
    df = df[~mask].reset_index(drop=True)

    # --- Capa de ALIAS: mapear el par activo a _my (baseline) / _miost (test) ---
    for metric in ["rmsd", "var", "std", "corr"]:
        col_base = f"{metric}_{BASELINE}"
        col_test = f"{metric}_{TEST}"
        col_cmn  = f"{metric}_{COMMON_BASELINE}"
        if col_base in df.columns:
            df[f"{metric}_my"] = df[col_base]       # alias baseline -> _my
        if col_test in df.columns:
            df[f"{metric}_miost"] = df[col_test]    # alias test     -> _miost
        if col_cmn in df.columns:
            df[f"{metric}_cmn"] = df[col_cmn]       # baseline común (MY)

    df["region"] = assign_regions_batch(df)
    return df


def load_combined_metrics(loess: bool = True) -> pd.DataFrame:
    df_calval = load_metrics("CalVal", loess)
    df_science = load_metrics("Science", loess)

    df_calval["phase"] = "CalVal"
    df_science["phase"] = "Science"

    return pd.concat([df_calval, df_science], ignore_index=True)


# =============================================================================
# 2. METRICAS DE PORCENTAJE DE MEJORA (PI)
# =============================================================================

def compute_improvement_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el Porcentaje de Mejora (PI) del producto TEST respecto al BASELINE
    del par activo, y ADEMÁS respecto al baseline común (MY).

    Baseline propio del par:
      PI_RMSD = 100 * (RMSD_baseline - RMSD_test) / RMSD_baseline   >0 ⇒ test mejor
      PI_Corr = 100 * (Corr_test - Corr_baseline) / (1 - Corr_baseline)

    Baseline común (MY), sufijo _vsMY:
      PI_RMSD_vsMY = 100 * (RMSD_MY - RMSD_test) / RMSD_MY
      PI_Corr_vsMY = 100 * (Corr_test - Corr_MY) / (1 - Corr_MY)
    """
    df = df.copy()

    # _my y _miost son ALIAS (baseline/test del par activo); _cmn es MY común.
    rmsd_b = df["rmsd_my"].values        # baseline del par
    rmsd_t = df["rmsd_miost"].values     # test
    corr_b = df["corr_my"].values
    corr_t = df["corr_miost"].values

    with np.errstate(divide='ignore', invalid='ignore'):
        df["PI_RMSD"] = 100.0 * (rmsd_b - rmsd_t) / rmsd_b
    denom = 1.0 - corr_b
    denom = np.where(np.abs(denom) < 1e-6, np.nan, denom)
    df["PI_Corr"] = 100.0 * (corr_t - corr_b) / denom

    df["delta_RMSD"] = rmsd_b - rmsd_t   # positivo = test mejor
    df["delta_Corr"] = corr_t - corr_b   # positivo = test mejor

    # --- Variante con baseline común MY (si difiere del baseline del par) ---
    if "rmsd_cmn" in df.columns and "corr_cmn" in df.columns:
        rmsd_c = df["rmsd_cmn"].values
        corr_c = df["corr_cmn"].values
        with np.errstate(divide='ignore', invalid='ignore'):
            df["PI_RMSD_vsMY"] = 100.0 * (rmsd_c - rmsd_t) / rmsd_c
        denom_c = 1.0 - corr_c
        denom_c = np.where(np.abs(denom_c) < 1e-6, np.nan, denom_c)
        df["PI_Corr_vsMY"] = 100.0 * (corr_t - corr_c) / denom_c
        df["delta_RMSD_vsMY"] = rmsd_c - rmsd_t
        df["delta_Corr_vsMY"] = corr_t - corr_c

    return df


# =============================================================================
# 3. DIAGRAMAS DE TAYLOR
# =============================================================================

def plot_taylor_diagram(df_calval: pd.DataFrame, df_science: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                              subplot_kw=dict(projection='polar'))

    for ax, df, phase in zip(axes, [df_calval, df_science], PHASES):
        ref_std = df["tg_std"].mean()

        std_my = df["std_my"].mean() / ref_std if "std_my" in df.columns else 1.0
        corr_my = df["corr_my"].mean()

        std_mi = df["std_miost"].mean() / ref_std if "std_miost" in df.columns else 1.0
        corr_mi = df["corr_miost"].mean()

        theta_my = np.arccos(np.clip(corr_my, -1, 1))
        theta_mi = np.arccos(np.clip(corr_mi, -1, 1))

        ax.set_thetamin(0)
        ax.set_thetamax(90)
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location('E')

        for corr_val in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
            theta_line = np.arccos(corr_val)
            ax.plot([theta_line, theta_line], [0, 2], 'k--', lw=0.3, alpha=0.5)
            ax.text(theta_line, 2.05, f'{corr_val}', fontsize=7, ha='center')

        ax.scatter([0], [1], s=150, c='black', marker='*', zorder=10, label='Reference (TG)')
        ax.scatter([theta_my], [std_my], s=120, c='#666666', marker='s',
                   edgecolors='k', linewidths=1, zorder=8, label=f'{BASELINE_LABEL} (r={corr_my:.3f})')
        ax.scatter([theta_mi], [std_mi], s=120, c='#2ca02c', marker='o',
                   edgecolors='k', linewidths=1, zorder=9, label=f'{TEST_LABEL} (r={corr_mi:.3f})')

        ax.set_rmax(2.0)
        ax.set_rticks([0.5, 1.0, 1.5, 2.0])
        ax.set_rlabel_position(45)

        ax.set_title(f"{PHASE_LABELS[phase]}", fontsize=FS, fontweight='bold', pad=15)
        ax.legend(loc='upper right', fontsize=FS-2)

    fig.suptitle(f"Taylor Diagram: {BASELINE_LABEL} vs {TEST_LABEL} (reference: Tide Gauges)",
                 fontsize=FS+1, fontweight='bold', y=1.02)

    fname = PATHS["output"] / "fig06_taylor_diagram_combined"
    fig.savefig(fname.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}.png")


def plot_taylor_by_region(df: pd.DataFrame, phase: str) -> None:
    n_regions = len(REGION_ORDER)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10),
                              subplot_kw=dict(projection='polar'))
    axes = axes.flatten()

    for ax, region in zip(axes, REGION_ORDER):
        df_reg = df[df["region"] == region]

        if len(df_reg) < 3:
            ax.set_visible(False)
            continue

        ref_std = df_reg["tg_std"].mean()

        std_my = df_reg["std_my"].mean() / ref_std if "std_my" in df_reg.columns else 1.0
        corr_my = df_reg["corr_my"].mean()
        std_mi = df_reg["std_miost"].mean() / ref_std if "std_miost" in df_reg.columns else 1.0
        corr_mi = df_reg["corr_miost"].mean()

        theta_my = np.arccos(np.clip(corr_my, -1, 1))
        theta_mi = np.arccos(np.clip(corr_mi, -1, 1))

        ax.set_thetamin(0)
        ax.set_thetamax(90)
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location('E')

        ax.scatter([0], [1], s=100, c='black', marker='*', zorder=10)
        ax.scatter([theta_my], [std_my], s=80, c='#666666', marker='s',
                   edgecolors='k', linewidths=0.5, zorder=8)
        ax.scatter([theta_mi], [std_mi], s=80, c=REGION_COLORS[region], marker='o',
                   edgecolors='k', linewidths=0.5, zorder=9)

        ax.set_rmax(2.0)
        ax.set_title(f"{region}\n(N={len(df_reg)})", fontsize=FS-1, pad=10)

    fig.suptitle(f"Taylor Diagrams by Region - {PHASE_LABELS[phase]}",
                 fontsize=FS+1, fontweight='bold', y=1.0)

    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='black',
               markersize=12, label='Reference (TG)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#666666',
               markeredgecolor='k', markersize=10, label=BASELINE_LABEL),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c',
               markeredgecolor='k', markersize=10, label=TEST_LABEL),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               fontsize=FS-1, bbox_to_anchor=(0.5, -0.02))

    fname = PATHS["output"] / f"fig06b_taylor_by_region_{phase}"
    fig.savefig(fname.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}.png")


# =============================================================================
# 4. MAPAS DE PERCENTAGE IMPROVEMENT
# =============================================================================

def _make_merc_ax(fig, pos, title="", extent=EXTENT):
    ax = fig.add_subplot(pos, projection=ccrs.Mercator(central_longitude=13))
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor="#d9d9d9", zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor="#c6e2f5", zorder=0)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="0.4", zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="0.2", zorder=3)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.4, color="0.5", alpha=0.7, linestyle="--", zorder=2)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(range(-15, 45, 15))
    gl.ylocator = mticker.FixedLocator(range(30, 75, 10))
    gl.xlabel_style = {"size": FS+2}
    gl.ylabel_style = {"size": FS+2}

    if title:
        ax.set_title(title, fontsize=FS, fontweight="bold", pad=4)
    return ax

def _make_merc_ax_1(fig, pos, title="", extent=EXTENT):
    ax = fig.add_subplot(pos, projection=ccrs.Mercator(central_longitude=13))
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND,      facecolor="#d9d9d9", zorder=1)
    ax.add_feature(cfeature.OCEAN,     facecolor="#c6e2f5", zorder=0)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.3, edgecolor="0.4", zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="0.2", zorder=3)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.4, color="0.5", alpha=0.7,
                      linestyle="--", zorder=2)
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(range(-15, 45, 15))
    gl.ylocator = mticker.FixedLocator(range(30, 75, 10))
    gl.xlabel_style = {"size": FS + 2}
    gl.ylabel_style = {"size": FS + 2}

    if title:
        ax.set_title(title, fontsize=FS, fontweight="bold", pad=4)

    return ax, gl          # ← devuelve también el gridliner


def _scatter_map(ax, lons, lats, values, cmap, vmin, vmax,
                 s=32, edgecolor="k", linewidth=0.3):
    sc = ax.scatter(lons, lats, c=values, s=s, cmap=cmap,
                    vmin=vmin, vmax=vmax, edgecolors=edgecolor,
                    linewidths=linewidth, transform=ccrs.PlateCarree(), zorder=5)
    return sc


def _add_cbar(fig, sc, ax, label, orientation="vertical", shrink=0.85, pad=0.04):
    cb = fig.colorbar(sc, ax=ax, orientation=orientation,
                      shrink=shrink, pad=pad, extend="both")
    cb.set_label(label, fontsize=FS - 1)
    cb.ax.tick_params(labelsize=FS - 2)
    return cb


def plot_percentage_improvement_maps(df_calval: pd.DataFrame, df_science: pd.DataFrame) -> None:
    """
    Mapas 2×2 de Percentage Improvement:
      Filas: CalVal, Science
      Columnas: PI_RMSD, PI_Corr
    """
    fig = plt.figure(figsize=(13, 11))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.12, wspace=0.08,
                           left=0.03, right=0.97, top=0.93, bottom=0.04)

    datasets = [
        (df_calval, "CalVal"),
        (df_science, "Science"),
    ]

    panel_labels = [["(a)", "(b)"], ["(c)", "(d)"]]

    for row, (df, phase) in enumerate(datasets):
        df = compute_improvement_metrics(df)
        lons = df["longitude"].values
        lats = df["latitude"].values

        RESULTS["improvement_metrics"][phase] = {
            "stations": df["station"].tolist(),
            "PI_RMSD": df["PI_RMSD"].tolist(),
            "PI_Corr": df["PI_Corr"].tolist(),
            "delta_RMSD": df["delta_RMSD"].tolist(),
            "delta_Corr": df["delta_Corr"].tolist(),
        }

        metrics = [
            ("PI_RMSD", "PI$_{RMSD}$ = 100 × (RMSD$_{base}$ - RMSD$_{test}$) / RMSD$_{base}$\n(+) = " + TEST_LABEL + " better [%]"),
            ("PI_Corr", "PI$_{Corr}$ = 100 × (r$_{test}$ - r$_{base}$) / (1 - r$_{base}$)\n(+) = " + TEST_LABEL + " better [%]"),
        ]

        for col, (pi_col, label) in enumerate(metrics):
            vals = df[pi_col].values

            metric_name = "PI RMSD" if "RMSD" in pi_col else "PI Corr"
            title = f"{panel_labels[row][col]} {metric_name} – {phase}"

            ax, gl = _make_merc_ax_1(fig, gs[row, col], title=title)  # ← desempaquetar

            # Suprimir etiquetas X en la fila superior
            if row == 0:
                gl.bottom_labels = False

            # Colormap y límites según métrica  ← Fix del bug de mayúsculas
            if pi_col == "PI_Corr_HP":
                sc = _scatter_map(ax, lons, lats, vals,
                                cmap="PiYG", vmin=-0.05, vmax=0.05, s=30)
            else:
                sc = _scatter_map(ax, lons, lats, vals,
                                cmap="RdYlGn",
                                vmin=PI_CLIM[0], vmax=PI_CLIM[1], s=30)

            n_better = (vals > 0).sum()
            n_total  = np.isfinite(vals).sum()
            pct = 100 * n_better / n_total if n_total > 0 else 0

            ax.text(0.02, 0.97, f"Mean: {np.nanmean(vals):+.2f} | Median: {np.nanmedian(vals):+.2f}",
                    transform=ax.transAxes, fontsize=FS, va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.85),
                    zorder=10)

            ax.text(0.02, 0.03, f"{TEST_LABEL} better: {pct:.0f}% ({n_better}/{n_total})",
                    transform=ax.transAxes, fontsize=FS, va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.85),
                    zorder=10)

            _add_cbar(fig, sc, ax, label=label.split("\n")[0], shrink=0.80, pad=0.02)
    
    fname = PATHS["output"] / "fig07_percentage_improvement_maps"
    fig.savefig(fname.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"  Saved {fname}.png")


def plot_rmsd_absolute_maps(df_calval: pd.DataFrame, df_science: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(16, 7))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.08,
                           left=0.03, right=0.97, top=0.92, bottom=0.06)

    datasets = [
        (df_calval, "CalVal"),
        (df_science, "Science"),
    ]

    panel_labels = ["(a)", "(b)"]
    delta_vmin, delta_vmax = -2, 2

    for col, (df, phase) in enumerate(datasets):
        lons = df["longitude"].values
        lats = df["latitude"].values

        delta_rmsd = df["rmsd_my"].values - df["rmsd_miost"].values

        title = f"{panel_labels[col]} ΔRMSD ({BASELINE_LABEL} − {TEST_LABEL}) – {PHASE_LABELS[phase]}"
        ax = _make_merc_ax(fig, gs[0, col], title=title)

        sc = _scatter_map(ax, lons, lats, delta_rmsd, cmap="seismic",
                          vmin=delta_vmin, vmax=delta_vmax, s=35)

        mean_val = np.nanmean(delta_rmsd)
        median_val = np.nanmedian(delta_rmsd)
        n_better = (delta_rmsd > 0).sum()
        n_valid = np.isfinite(delta_rmsd).sum()
        pct_better = 100 * n_better / n_valid if n_valid > 0 else 0

        RESULTS["rmsd_analysis"][phase] = {
            "delta_rmsd_mean": float(mean_val),
            "delta_rmsd_median": float(median_val),
            "delta_rmsd_std": float(np.nanstd(delta_rmsd)),
            "n_miost_better": int(n_better),
            "n_valid": int(n_valid),
            "pct_miost_better": float(pct_better),
            "stations": df["station"].tolist(),
            "delta_rmsd_per_station": delta_rmsd.tolist(),
        }

        ax.text(0.02, 0.03,
                f"Mean: {mean_val:+.2f} cm | Median: {median_val:+.2f} cm\n{TEST_LABEL} better: {pct_better:.0f}% ({n_better}/{n_valid})",
                transform=ax.transAxes, fontsize=FS, va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.85),
                zorder=10)

        _add_cbar(fig, sc, ax, label="ΔRMSD (cm)\n(+) = " + TEST_LABEL + " better", shrink=0.75, pad=0.02)

    fname = PATHS["output"] / "fig07b_rmsd_difference_maps"
    fig.savefig(fname.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"  Saved {fname}.png")


# =============================================================================
# 4b. SCATTERPLOT RMSD MY vs RMSD MIOST POR REGIÓN
# =============================================================================

def plot_rmsd_scatter_by_region(df_calval: pd.DataFrame, df_science: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 11), constrained_layout=True)

    datasets = [
        (df_calval, "CalVal"),
        (df_science, "Science"),
    ]

    for row, (df, phase) in enumerate(datasets):
        ax = axes[row]

        x = df["rmsd_my"].values
        y = df["rmsd_miost"].values

        for region in REGION_ORDER:
            mask = df["region"] == region
            if not mask.any():
                continue
            ax.scatter(x[mask], y[mask], c=REGION_COLORS[region],
                       s=50, alpha=0.7, edgecolors="white", linewidths=0.5,
                       label=region, zorder=4)

        limit_min = min(np.nanmin(x), np.nanmin(y))
        limit_max = max(np.nanmax(x), np.nanmax(y))
        ax.plot([limit_min, limit_max], [limit_min, limit_max],
               "k--", lw=1.5, zorder=2, alpha=0.6)

        # ax.fill_between([limit_min, limit_max], limit_min, [limit_min, limit_max],
        #                alpha=0.05, color="green", zorder=1, label="" + TEST_LABEL + " better")

        valid = np.isfinite(x) & np.isfinite(y)
        n_valid = valid.sum()
        if n_valid > 0:
            mean_x = np.mean(x[valid])
            mean_y = np.mean(y[valid])
            n_miost_better = (y[valid] < x[valid]).sum()
            pct_better = 100 * n_miost_better / n_valid

            if "rmsd_scatter" not in RESULTS["rmsd_analysis"]:
                RESULTS["rmsd_analysis"]["rmsd_scatter"] = {}
            RESULTS["rmsd_analysis"]["rmsd_scatter"][phase] = {
                "rmsd_my_mean": float(mean_x),
                "rmsd_miost_mean": float(mean_y),
                "n_valid": int(n_valid),
                "n_miost_better": int(n_miost_better),
                "pct_miost_better": float(pct_better),
            }

            stats_text = (f"RMSD$_{{MY}}$ mean: {mean_x:.2f} cm | "
                         f"RMSD$_{{MIOST}}$ mean: {mean_y:.2f} cm\n"
                         f"{TEST_LABEL} better: {pct_better:.0f}% ({n_miost_better}/{n_valid})")

            ax.text(0.02, 0.98, stats_text,
                   transform=ax.transAxes, fontsize=FS-1, va="top", ha="left",
                   bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.5", alpha=0.85),
                   zorder=10)

        ax.set_xlabel("RMSD$_{base}$ (cm)", fontsize=FS)
        ax.set_ylabel("RMSD$_{test}$ (cm)", fontsize=FS)
        ax.set_title(f"RMSD by Maritime Region – {PHASE_LABELS[phase]}",
                    fontsize=FS, fontweight="bold")
        ax.grid(True, linewidth=0.4, alpha=0.5, zorder=0)

        x_range = np.nanmax(x) - np.nanmin(x)
        y_range = np.nanmax(y) - np.nanmin(y)
        if 0.8 < y_range / x_range < 1.2:
            ax.set_aspect('equal', adjustable='box')

        if row == 0:
            ax.legend(fontsize=FS-2, loc="lower right", ncol=1, framealpha=0.95)

    fname = PATHS["output"] / "fig07c_rmsd_scatter_by_region"
    fig.savefig(fname.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}.png")


# =============================================================================
# 5. ANÁLISIS POR VARIABILIDAD LOCAL
# =============================================================================

def plot_metrics_vs_variability(df_calval: pd.DataFrame, df_science: pd.DataFrame) -> None:
    """
    Scatter plots: Percentage Improvement (PI) vs variabilidad local (tg_std).
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)

    datasets = [
        (df_calval, "CalVal"),
        (df_science, "Science"),
    ]

    metrics = ["PI_RMSD", "PI_Corr"]

    for col, (df, phase) in enumerate(datasets):
        df = compute_improvement_metrics(df)

        for row, metric in enumerate(metrics):
            ax = axes[row, col]

            x = df["tg_std"].values
            y = df[metric].values

            for region in REGION_ORDER:
                mask = df["region"] == region
                if not mask.any():
                    continue
                ax.scatter(x[mask], y[mask], c=REGION_COLORS[region],
                           s=35, alpha=0.7, edgecolors="white", linewidths=0.3,
                           label=region, zorder=4)

            ax.axhline(0, color="k", ls="--", lw=1, zorder=2, alpha=0.7)

            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() > 5:
                slope, intercept, r, pval, _ = stats.linregress(x[valid], y[valid])
                xfit = np.linspace(np.nanmin(x), np.nanmax(x), 100)
                ax.plot(xfit, slope * xfit + intercept, "r-", lw=1.5, zorder=3,
                        label=f"r = {r:.3f} (p = {pval:.2e})")

                reg_key = f"{phase}_{metric}_vs_tg_std"
                RESULTS["regression_analysis"][reg_key] = {
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "r": float(r),
                    "r_squared": float(r**2),
                    "p_value": float(pval),
                    "n_points": int(valid.sum()),
                }

            ax.set_xlabel("TG Standard Deviation (cm)", fontsize=FS)
            ax.set_ylabel(f"{metric} [%]", fontsize=FS)
            ax.set_title(f"{'(abcd)'[row*2+col]}) {metric} vs Local Variability – {PHASE_LABELS[phase]}",
                         fontsize=FS, fontweight="bold")
            ax.grid(True, linewidth=0.4, alpha=0.5)

            if row == 0 and col == 0:
                ax.legend(fontsize=FS-2, loc="lower right", ncol=2)

    fname = PATHS["output"] / "fig08_metrics_vs_variability"
    fig.savefig(fname.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}.png")


# =============================================================================
# 6. HISTOGRAMAS DE MEJORA
# =============================================================================

def plot_improvement_histograms_simple(df_calval: pd.DataFrame, df_science: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)

    datasets = [
        (df_calval, "CalVal"),
        (df_science, "Science"),
    ]

    metrics = [
        ("delta_RMSD", "ΔRMSD = RMSD(base) - RMSD(test) [cm]\n(+) = " + TEST_LABEL + " better", "cm"),
        ("delta_Corr", "ΔCorr = r(test) - r(base)\n(+) = " + TEST_LABEL + " better", ""),
        ("PI_RMSD", "PI$_{RMSD}$ = 100 × ΔRMSD / RMSD$_{base}$\n(+) = " + TEST_LABEL + " better", "%"),
        ]

    for row, (df, phase) in enumerate(datasets):
        df = compute_improvement_metrics(df)

        for col, (metric, label, unit) in enumerate(metrics):
            ax = axes[row, col]

            vals = df[metric].dropna().values

            bins = 30
            n, bins_edges, patches = ax.hist(vals, bins=bins, color="steelblue",
                                              edgecolor="white", alpha=0.8, zorder=3)

            for patch, left_edge in zip(patches, bins_edges[:-1]):
                if left_edge >= 0:
                    patch.set_facecolor("#2ca02c")
                else:
                    patch.set_facecolor("#d62728")

            ax.axvline(0, color="k", ls="--", lw=1.5, zorder=4)

            mean_val = np.mean(vals)
            median_val = np.median(vals)
            pct_better = 100 * (vals > 0).sum() / len(vals)

            if "histograms" not in RESULTS["improvement_stats"]:
                RESULTS["improvement_stats"]["histograms"] = {}
            hist_key = f"{phase}_{metric}"
            RESULTS["improvement_stats"]["histograms"][hist_key] = {
                "mean": float(mean_val),
                "median": float(median_val),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "pct_miost_better": float(pct_better),
                "n_stations": int(len(vals)),
                "q25": float(np.percentile(vals, 25)),
                "q75": float(np.percentile(vals, 75)),
            }

            stats_txt = (f"Mean: {mean_val:+.3f}{unit}\n"
                         f"Median: {median_val:+.3f}{unit}\n"
                         f"{TEST_LABEL} better: {pct_better:.0f}%\n"
                         f"N = {len(vals)}")
            ax.text(0.03, 0.97, stats_txt, transform=ax.transAxes,
                    fontsize=FS, va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.9))

            ax.set_xlabel(label.split("\n")[0], fontsize=FS)
            ax.set_ylabel("Frequency", fontsize=FS)
            ax.set_title(f"{'(abcdef)'[row*3+col]}) {label.split('=')[0].strip()} – {PHASE_LABELS[phase]}",
                         fontsize=FS, fontweight="bold")
            ax.grid(True, linewidth=0.4, alpha=0.5, axis="y")

    fname = PATHS["output"] / "fig09_improvement_histograms"
    fig.savefig(fname.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}.png")


def plot_improvement_histograms_by_region(df_calval: pd.DataFrame, df_science: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)

    datasets = [
        (df_calval, "CalVal"),
        (df_science, "Science"),
    ]

    metrics = [
        ("delta_RMSD", "ΔRMSD = RMSD(base) - RMSD(test) [cm]\n(+) = " + TEST_LABEL + " better", "cm"),
        ("delta_Corr", "ΔCorr = r(test) - r(base)\n(+) = " + TEST_LABEL + " better", ""),
        ("PI_RMSD", "PI$_{RMSD}$ = 100 × ΔRMSD / RMSD$_{base}$\n(+) = " + TEST_LABEL + " better", "%"),
        ]

    for row, (df, phase) in enumerate(datasets):
        df = compute_improvement_metrics(df)

        for col, (metric, label, unit) in enumerate(metrics):
            ax = axes[row, col]

            vals = df[metric].dropna()

            bins = 30
            bins_edges = np.histogram_bin_edges(vals.values, bins=bins)

            region_hists = {}
            for region in REGION_ORDER:
                df_region = df[df["region"] == region]
                vals_region = df_region[metric].dropna().values
                if len(vals_region) > 0:
                    counts, _ = np.histogram(vals_region, bins=bins_edges)
                    region_hists[region] = counts

            bins_centers = (bins_edges[:-1] + bins_edges[1:]) / 2
            bins_width = bins_edges[1] - bins_edges[0]

            bottom = np.zeros(len(bins_centers))
            for region in REGION_ORDER:
                if region in region_hists:
                    counts = region_hists[region]
                    ax.bar(bins_centers, counts, width=bins_width * 0.95,
                          bottom=bottom, label=region,
                          color=REGION_COLORS[region], edgecolor="white",
                          linewidth=0.5, alpha=0.85, zorder=3)
                    bottom += counts

            ax.axvline(0, color="k", ls="--", lw=1.5, zorder=4)

            mean_val = vals.mean()
            median_val = vals.median()
            pct_better = 100 * (vals > 0).sum() / len(vals)

            stats_txt = (f"Mean: {mean_val:+.3f}{unit}\n"
                         f"Median: {median_val:+.3f}{unit}\n"
                         f"{TEST_LABEL} better: {pct_better:.0f}%\n"
                         f"N = {len(vals)}")
            
            if row == 1 and col == 2:
                ax.text(0.03, 0.97, stats_txt, transform=ax.transAxes,
                        fontsize=FS-1, va="top", ha="left",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.9))
            else:
                ax.text(0.97, 0.97, stats_txt, transform=ax.transAxes,
                        fontsize=FS-1, va="top", ha="right",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.9))
            

            ax.set_xlabel(label.split("\n")[0], fontsize=FS)
            ax.set_ylabel("Frequency", fontsize=FS)
            ax.set_title(f"{'(abcdef)'[row*3+col]}) {label.split('=')[0].strip()} – {PHASE_LABELS[phase]}",
                         fontsize=FS, fontweight="bold")
            ax.grid(True, linewidth=0.4, alpha=0.5, axis="y")

            if row == 0 and col == 2:
                ax.legend(loc="upper left", fontsize=FS, frameon=True,
                         fancybox=True, shadow=False, ncol=1, framealpha=0.9)

    fname = PATHS["output"] / "fig09b_improvement_histograms_by_region"
    fig.savefig(fname.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}.png")

def plot_calval_swot_improvement_histograms(df_calval: pd.DataFrame, swot_tracks: dict) -> None:
    df_near, df_far = filter_stations_by_swot_proximity(df_calval, swot_tracks)
    df_near = compute_improvement_metrics(df_near)
    df_far = compute_improvement_metrics(df_far)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    metrics = [("PI_RMSD", "PI$_{RMSD}$ [%]"), ("PI_Corr", "PI$_{Corr}$ [%]")]

    for ax, (metric, label) in zip(axes, metrics):
        data_near = df_near[metric].dropna().values
        data_far = df_far[metric].dropna().values

        if len(data_near) == 0 or len(data_far) == 0:
            continue

        all_data = np.concatenate([data_near, data_far])
        bins = np.histogram_bin_edges(all_data, bins=30)

        ax.hist(data_far, bins=bins, color="gray", edgecolor="dimgray", alpha=0.6,
                label=f"Far from SWOT (N={len(data_far)})", zorder=3)
        ax.hist(data_near, bins=bins, color="#2ca02c", edgecolor="darkgreen", alpha=0.85,
                label=f"Near SWOT (N={len(data_near)})", zorder=4)

        med_near = np.median(data_near)
        med_far = np.median(data_far)

        # --- CÁLCULO DE MACRO PI ---
        if metric == "PI_RMSD":
            my_n = df_near["rmsd_my"].median()
            mi_n = df_near["rmsd_miost"].median()
            macro_near = 100 * (my_n - mi_n) / my_n if my_n != 0 else np.nan
            my_f = df_far["rmsd_my"].median()
            mi_f = df_far["rmsd_miost"].median()
            macro_far = 100 * (my_f - mi_f) / my_f if my_f != 0 else np.nan
        else:
            my_n = df_near["corr_my"].median()
            mi_n = df_near["corr_miost"].median()
            macro_near = 100 * (mi_n - my_n) / (1 - my_n) if my_n != 1 else np.nan
            my_f = df_far["corr_my"].median()
            mi_f = df_far["corr_miost"].median()
            macro_far = 100 * (mi_f - my_f) / (1 - my_f) if my_f != 1 else np.nan

        # Líneas de medianas individuales (Punteadas y finas)
        ax.axvline(med_far, color="dimgray", ls=":", lw=2, zorder=10)
        ax.axvline(med_near, color="darkgreen", ls=":", lw=2, zorder=10)
        
        # Líneas de Global PI (Azul y Negro, Discontinuas gruesas para que destaquen sobre los histogramas)
        # ax.axvline(macro_far, color="black", ls="--", lw=3, zorder=15)
        # ax.axvline(macro_near, color="blue", ls="--", lw=3, zorder=15)

        ax.axvline(0, color="black", ls="-", lw=1, zorder=2)

        try:
            stat, pval = stats.mannwhitneyu(data_near, data_far, alternative="two-sided")
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
            pval_text = f"M-W Test: p = {pval:.3e} {sig}"
        except Exception:
            pval_text = ""

        stats_txt = (
            f"NEAR KaRIn -> Median PI: {med_near:+.1f}%\n"
            f"FAR KaRIn  -> Median PI: {med_far:+.1f}%\n"
            f"{pval_text}"
        )

        ax.text(0.02, 0.96, stats_txt, transform=ax.transAxes,
                fontsize=FS-1, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="0.5", alpha=0.9), zorder=20)

        # Leyenda customizada
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([0], [0], color='k', ls=':', lw=2))
        labels.append('Median PI (Per Station)')
        handles.append(Line2D([0], [0], color='blue', ls='--', lw=3))
        labels.append('Global PI (Dashed)')

        ax.set_xlabel(label, fontsize=FS)
        ax.set_ylabel("Frequency", fontsize=FS)
        ax.set_title(f"{label} Distribution – CalVal Phase", fontsize=FS+1, fontweight="bold")
        ax.legend(handles=handles, labels=labels, loc="center right", fontsize=FS-2, framealpha=0.95)
        ax.grid(True, linewidth=0.4, alpha=0.5, axis="y")

    fname = PATHS["output"] / "fig12c_calval_swot_pi_histograms"
    fig.savefig(fname.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}.png")

# =============================================================================
# 7. BARRAS REGIONALES DE PERCENTAGE IMPROVEMENT
# =============================================================================

def plot_regional_improvement_bars(df_calval: pd.DataFrame, df_science: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)

    datasets = [
        (df_calval, "CalVal"),
        (df_science, "Science"),
    ]

    metrics = [("PI_RMSD", "PI$_{RMSD}$ [%]"), ("PI_Corr", "PI$_{Corr}$ [%]")]

    for col, (df_orig, phase) in enumerate(datasets):
        df = compute_improvement_metrics(df_orig)

        for row, (metric, metric_label) in enumerate(metrics):
            ax = axes[row, col]

            region_stats = []
            for region in REGION_ORDER:
                df_reg = df[df["region"] == region]
                if len(df_reg) >= 3:
                    val_mean = df_reg[metric].mean()
                    val_std = df_reg[metric].std()
                    n_sta = len(df_reg)
                    region_stats.append((region, val_mean, val_std, n_sta))

            if "regional_bars" not in RESULTS["regional_stats"]:
                RESULTS["regional_stats"]["regional_bars"] = {}
            bars_key = f"{phase}_{metric}"
            RESULTS["regional_stats"]["regional_bars"][bars_key] = {
                region: {
                    "mean": float(val_mean),
                    "std": float(val_std),
                    "n_stations": int(n_sta)
                }
                for region, val_mean, val_std, n_sta in region_stats
            }

            if not region_stats:
                ax.set_visible(False)
                continue

            regions = [r[0] for r in region_stats]
            means = [r[1] for r in region_stats]
            stds = [r[2] for r in region_stats]
            n_stations = [r[3] for r in region_stats]

            y_pos = np.arange(len(regions))
            colors = [REGION_COLORS[r] for r in regions]

            bars = ax.barh(y_pos, means, xerr=stds, height=0.6,
                           color=colors, edgecolor="0.2", linewidth=0.7,
                           error_kw=dict(ecolor="0.3", lw=1, capsize=3))

            ax.axvline(0, color="k", ls="-", lw=1.2, zorder=3)

            ax.axvspan(-20, 0, alpha=0.08, color="red", zorder=0)
            ax.axvspan(0, 20, alpha=0.08, color="green", zorder=0)

            for i, (mean, n) in enumerate(zip(means, n_stations)):
                x_pos = mean + stds[i] + 0.5 if mean >= 0 else mean - stds[i] - 0.5
                ha = "left" if mean >= 0 else "right"
                ax.text(x_pos, y_pos[i], f"N={n}", va="center", ha=ha,
                        fontsize=FS-2, color="0.3")

            ax.set_yticks(y_pos)
            ax.set_yticklabels(regions, fontsize=FS-1)
            ax.set_xlabel(metric_label, fontsize=FS)
            ax.set_xlim(-20, 20)  # Ampliado para porcentajes
            ax.set_title(f"{'(abcd)'[row*2+col]}) {metric_label} by Region – {PHASE_LABELS[phase]}",
                         fontsize=FS, fontweight="bold")
            ax.grid(True, linewidth=0.4, alpha=0.5, axis="x")
            ax.spines[["top", "right"]].set_visible(False)

    fname = PATHS["output"] / "fig10_regional_improvement_bars"
    fig.savefig(fname.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}.png")


# =============================================================================
# 8. TESTS DE SIGNIFICANCIA ESTADÍSTICA
# =============================================================================

def paired_significance_tests(df: pd.DataFrame, phase: str) -> pd.DataFrame:
    results = []

    for metric_base in ["rmsd", "corr"]:
        col_my = f"{metric_base}_my"
        col_mi = f"{metric_base}_miost"

        valid = df[[col_my, col_mi]].dropna()
        x = valid[col_my].values
        y = valid[col_mi].values

        if metric_base == "rmsd":
            diff = x - y  
        else:
            diff = y - x  

        t_stat, t_pval = stats.ttest_rel(y, x) if metric_base == "corr" else stats.ttest_rel(x, y)

        try:
            w_stat, w_pval = stats.wilcoxon(diff)
        except ValueError:
            w_stat, w_pval = np.nan, np.nan

        n_bootstrap = 10000
        rng = np.random.default_rng(42)
        # Generar todas las muestras de golpe en una matriz 2D y calcular las medias por fila
        samples = rng.choice(diff, size=(n_bootstrap, len(diff)), replace=True)
        boot_means = np.mean(samples, axis=1)
        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

        results.append({
            "phase": phase,
            "metric": metric_base.upper(),
            "n_stations": len(diff),
            "mean_diff": np.mean(diff),
            "std_diff": np.std(diff),
            "median_diff": np.median(diff),
            "pct_miost_better": 100 * (diff > 0).sum() / len(diff),
            "t_statistic": t_stat,
            "t_pvalue": t_pval,
            "wilcoxon_stat": w_stat,
            "wilcoxon_pval": w_pval,
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
        })

    return pd.DataFrame(results)


def plot_significance_summary(results_calval: pd.DataFrame,
                               results_science: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    all_results = pd.concat([results_calval, results_science], ignore_index=True)

    for ax, metric in zip(axes, ["RMSD", "Corr"]):
        df_metric = all_results[all_results["metric"] == metric]

        phases = df_metric["phase"].values
        means = df_metric["mean_diff"].values
        ci_lows = df_metric["bootstrap_ci_low"].values
        ci_highs = df_metric["bootstrap_ci_high"].values
        pvals = df_metric["t_pvalue"].values

        y_pos = np.arange(len(phases))
        colors = ["#4E79A7", "#F28E2B"]

        for i, (phase, mean, ci_l, ci_h, pval, color) in enumerate(
                zip(phases, means, ci_lows, ci_highs, pvals, colors)):

            ax.errorbar(mean, y_pos[i], xerr=[[mean - ci_l], [ci_h - mean]],
                        fmt='o', markersize=12, color=color, capsize=8, capthick=2,
                        elinewidth=2, markeredgecolor='k', markeredgewidth=1)

            sig_marker = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
            ax.text(ci_h + 0.01, y_pos[i], f"p={pval:.2e} {sig_marker}",
                    va="center", ha="left", fontsize=FS-1)

        ax.axvline(0, color="k", ls="--", lw=1.2)
        ax.axvspan(0 if metric == "Corr" else -1, 1, alpha=0.08, color="green", zorder=0)
        ax.axvspan(-1 if metric == "Corr" else 0, 0 if metric == "Corr" else 1,
                   alpha=0.08, color="red", zorder=0)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(phases, fontsize=FS)
        ax.set_xlabel(f"Mean Δ{metric} ({TEST_LABEL} better →)" if metric == "Corr"
                      else f"Mean Δ{metric} [cm] (← {TEST_LABEL} better)", fontsize=FS)
        ax.set_title(f"{metric}: Bootstrap 95% CI", fontsize=FS, fontweight="bold")
        ax.grid(True, linewidth=0.4, alpha=0.5, axis="x")

    fname = PATHS["output"] / "fig13_significance_ci"
    fig.savefig(fname.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}.png")


def save_summary_table(results_calval: pd.DataFrame,
                        results_science: pd.DataFrame) -> None:
    all_results = pd.concat([results_calval, results_science], ignore_index=True)
    # Etiquetar con la comparación activa para trazabilidad entre carpetas
    all_results.insert(0, "comparison", COMPARISON_KEY)
    all_results.insert(1, "baseline", BASELINE)
    all_results.insert(2, "test", TEST)
    csv_path = PATHS["output"] / "statistical_tests_summary.csv"
    all_results.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"  Saved {csv_path}")

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.axis("off")

    table_data = []
    for _, row in all_results.iterrows():
        sig = "***" if row["t_pvalue"] < 0.001 else "**" if row["t_pvalue"] < 0.01 else "*" if row["t_pvalue"] < 0.05 else ""
        table_data.append([
            row["phase"],
            row["metric"],
            f"{row['n_stations']:.0f}",
            f"{row['mean_diff']:+.3f}",
            f"{row['pct_miost_better']:.1f}%",
            f"{row['t_pvalue']:.2e}{sig}",
            f"{row['wilcoxon_pval']:.2e}",
            f"[{row['bootstrap_ci_low']:+.3f}, {row['bootstrap_ci_high']:+.3f}]",
        ])

    columns = ["Phase", "Metric", "N", "Mean Δ", "" + TEST_LABEL + " better",
               "t-test p", "Wilcoxon p", "95% CI"]

    table = ax.table(cellText=table_data, colLabels=columns, loc="center",
                      cellLoc="center", colWidths=[0.1]*8)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    for i in range(len(columns)):
        table[(0, i)].set_facecolor("#4472C4")
        table[(0, i)].set_text_props(color="white", fontweight="bold")

    ax.set_title(f"Statistical Significance Tests: {TEST_LABEL} vs {BASELINE_LABEL}",
                 fontsize=FS+1, fontweight="bold", pad=20)

    fname = PATHS["output"] / "fig14_summary_table"
    fig.savefig(fname.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}.png")


# =============================================================================
# 8b. ANÁLISIS DE SIGNIFICANCIA POR REGIONES
# =============================================================================

def regional_significance_tests(df: pd.DataFrame, phase: str) -> pd.DataFrame:
    results = []
    for region in REGION_ORDER:
        df_reg = df[df["region"] == region]
        if len(df_reg) < 5: continue

        for metric_base in ["rmsd", "corr"]:
            col_my = f"{metric_base}_my"
            col_mi = f"{metric_base}_miost"

            valid = df_reg[[col_my, col_mi]].dropna()
            if len(valid) < 5: continue

            x = valid[col_my].values
            y = valid[col_mi].values

            if metric_base == "rmsd":
                diff = x - y  
            else:
                diff = y - x  

            t_stat, t_pval = stats.ttest_rel(y, x) if metric_base == "corr" else stats.ttest_rel(x, y)

            try:
                w_stat, w_pval = stats.wilcoxon(diff)
            except ValueError:
                w_stat, w_pval = np.nan, np.nan

            n_bootstrap = 10000
            rng = np.random.default_rng(42)
            samples = rng.choice(diff, size=(n_bootstrap, len(diff)), replace=True)
            boot_means = np.mean(samples, axis=1)
            ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

            results.append({
                "phase": phase,
                "region": region,
                "metric": metric_base.upper(),
                "n_stations": len(diff),
                "mean_diff": np.mean(diff),
                "std_diff": np.std(diff),
                "median_diff": np.median(diff),
                "pct_miost_better": 100 * (diff > 0).sum() / len(diff),
                "t_statistic": t_stat,
                "t_pvalue": t_pval,
                "wilcoxon_stat": w_stat,
                "wilcoxon_pval": w_pval,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
            })

    return pd.DataFrame(results)


def plot_regional_significance(results_calval: pd.DataFrame,
                                results_science: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), constrained_layout=True)
    datasets = [(results_calval, "CalVal", 0), (results_science, "Science", 1)]
    metrics = [("RMSD", 0), ("CORR", 1)]

    for df_results, phase, col in datasets:
        for metric, row in metrics:
            ax = axes[row, col]
            df_metric = df_results[df_results["metric"] == metric]
            if len(df_metric) == 0:
                ax.set_visible(False)
                continue

            regions = df_metric["region"].values
            means = df_metric["mean_diff"].values
            ci_lows = df_metric["bootstrap_ci_low"].values
            ci_highs = df_metric["bootstrap_ci_high"].values
            pvals = df_metric["t_pvalue"].values
            n_stations = df_metric["n_stations"].values

            y_pos = np.arange(len(regions))

            for i, (region, mean, ci_l, ci_h, pval, n) in enumerate(
                    zip(regions, means, ci_lows, ci_highs, pvals, n_stations)):

                color = REGION_COLORS.get(region, "steelblue")
                ax.errorbar(mean, y_pos[i], xerr=[[mean - ci_l], [ci_h - mean]],
                            fmt='o', markersize=10, color=color, capsize=6, capthick=2,
                            elinewidth=2, markeredgecolor='k', markeredgewidth=0.8)

                sig_marker = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                x_text = max(ci_h, mean) + 0.02
                ax.text(x_text, y_pos[i], f"p={pval:.3f}{sig_marker} (N={n:.0f})",
                        va="center", ha="left", fontsize=FS-2, color="0.3")

            ax.axvline(0, color="k", ls="--", lw=1.2, zorder=1)

            xlim = ax.get_xlim()
            ax.axvspan(0, xlim[1], alpha=0.08, color="green", zorder=0)
            ax.axvspan(xlim[0], 0, alpha=0.08, color="red", zorder=0)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(regions, fontsize=FS-1)
            ax.set_xlabel(f"Mean Δ{metric} ({TEST_LABEL} better →)" if metric == "CORR"
                          else f"Mean Δ{metric} [cm] ({TEST_LABEL} better →)", fontsize=FS)
            ax.set_title(f"{'(abcd)'[row*2+col]}) {metric} – {PHASE_LABELS[phase]}",
                         fontsize=FS, fontweight="bold")
            ax.grid(True, linewidth=0.4, alpha=0.5, axis="x")
            ax.spines[["top", "right"]].set_visible(False)

    legend_text = "Significance: * p<0.05, ** p<0.01, *** p<0.001"
    fig.text(0.5, -0.01, legend_text, ha="center", fontsize=FS-1, style="italic")

    fname = PATHS["output"] / "fig16_regional_significance"
    fig.savefig(fname.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}.png")


def save_regional_significance_table(results_calval: pd.DataFrame,
                                      results_science: pd.DataFrame) -> None:
    all_results = pd.concat([results_calval, results_science], ignore_index=True)
    if not all_results.empty:
        all_results.insert(0, "comparison", COMPARISON_KEY)
        all_results.insert(1, "baseline", BASELINE)
        all_results.insert(2, "test", TEST)
    csv_path = PATHS["output"] / "regional_significance_tests.csv"
    all_results.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"  Saved {csv_path}")

    for phase in PHASES:
        df_phase = all_results[all_results["phase"] == phase]
        if len(df_phase) == 0: continue

        fig, ax = plt.subplots(figsize=(18, 8))
        ax.axis("off")

        table_data = []
        for _, row in df_phase.iterrows():
            sig = "***" if row["t_pvalue"] < 0.001 else "**" if row["t_pvalue"] < 0.01 else "*" if row["t_pvalue"] < 0.05 else ""
            table_data.append([
                row["region"], row["metric"], f"{row['n_stations']:.0f}",
                f"{row['mean_diff']:+.4f}", f"{row['pct_miost_better']:.1f}%",
                f"{row['t_pvalue']:.4f}{sig}", f"{row['wilcoxon_pval']:.4f}",
                f"[{row['bootstrap_ci_low']:+.4f}, {row['bootstrap_ci_high']:+.4f}]",
            ])

        columns = ["Region", "Metric", "N", "Mean Δ", "" + TEST_LABEL + " better", "t-test p", "Wilcoxon p", "95% CI"]
        table = ax.table(cellText=table_data, colLabels=columns, loc="center",
                          cellLoc="center", colWidths=[0.15, 0.08, 0.06, 0.1, 0.1, 0.12, 0.12, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.4)

        for i in range(len(columns)):
            table[(0, i)].set_facecolor("#4472C4")
            table[(0, i)].set_text_props(color="white", fontweight="bold")

        for i, row in enumerate(table_data):
            region = row[0]
            color = REGION_COLORS.get(region, "#FFFFFF")
            table[(i+1, 0)].set_facecolor(color)
            table[(i+1, 0)].set_alpha(0.3)

        ax.set_title(f"Regional Statistical Significance – {PHASE_LABELS[phase]}",
                     fontsize=FS+1, fontweight="bold", pad=20)

        fname = PATHS["output"] / f"fig17_regional_table_{phase}"
        fig.savefig(fname.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}.png")


def plot_regional_heatmap(results_calval: pd.DataFrame,
                           results_science: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), constrained_layout=True)
    all_results = pd.concat([results_calval, results_science], ignore_index=True)

    for ax, metric in zip(axes, ["RMSD", "CORR"]):
        df_metric = all_results[all_results["metric"] == metric]
        regions_present = [r for r in REGION_ORDER if r in df_metric["region"].values]

        matrix = np.full((len(regions_present), len(PHASES)), np.nan)
        annotations = [['' for _ in PHASES] for _ in regions_present]

        for i, region in enumerate(regions_present):
            for j, phase in enumerate(PHASES):
                row = df_metric[(df_metric["region"] == region) & (df_metric["phase"] == phase)]
                if len(row) > 0:
                    matrix[i, j] = row["mean_diff"].values[0]
                    pval = row["t_pvalue"].values[0]
                    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                    n = row["n_stations"].values[0]
                    annotations[i][j] = f"{matrix[i,j]:+.3f}{sig}\n(N={n:.0f})"

        vmax = np.nanmax(np.abs(matrix)) if not np.all(np.isnan(matrix)) else 1
        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)

        for i in range(len(regions_present)):
            for j in range(len(PHASES)):
                text_color = "white" if abs(matrix[i, j]) > vmax * 0.7 else "black"
                ax.text(j, i, annotations[i][j], ha="center", va="center",
                        fontsize=FS-2, color=text_color)

        ax.set_xticks(range(len(PHASES)))
        ax.set_xticklabels([PHASE_LABELS[p] for p in PHASES], fontsize=FS-1)
        ax.set_yticks(range(len(regions_present)))
        ax.set_yticklabels(regions_present, fontsize=FS-1)

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        unit = " [cm]" if metric == "RMSD" else ""
        cbar.set_label(f"Mean Δ{metric}{unit}", fontsize=FS-1)

        ax.set_title(f"Δ{metric} by Region and Phase\n(+) = {TEST_LABEL} better",
                     fontsize=FS, fontweight="bold")

    fname = PATHS["output"] / "fig18_regional_heatmap"
    fig.savefig(fname.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}.png")



# =============================================================================
# 9. FILTRO CALVAL POR PROXIMIDAD A TRACKS SWOT
# =============================================================================

def load_swot_tracks(swot_dir: Path) -> dict:
    swot_dir = Path(swot_dir)  # coerción defensiva: acepta str o Path
    tracks = {}
    for pass_num in SWOT_CALVAL_PASSES:
        pattern = f"SWOT_L3_LR_SSH_Expert_*_{pass_num:03d}_*.nc"
        files = sorted(swot_dir.glob(pattern))

        if not files:
            continue

        all_lats, all_lons = [], []
        for f in files[:10]: 
            try:
                ds = xr.open_dataset(f)
                if "latitude" in ds.variables and "longitude" in ds.variables:
                    lats = ds["latitude"].values.flatten()
                    lons = ds["longitude"].values.flatten()
                    lons = np.where(lons > 180, lons - 360, lons)
                    valid = (np.isfinite(lats) & np.isfinite(lons) &
                             (lats > EXTENT[2]) & (lats < EXTENT[3]) &
                             (lons > EXTENT[0]) & (lons < EXTENT[1]))
                    all_lats.extend(lats[valid])
                    all_lons.extend(lons[valid])
                ds.close()
            except Exception:
                continue

        if all_lats:
            tracks[pass_num] = (np.array(all_lats), np.array(all_lons))
            print(f"  Loaded track pass {pass_num}: {len(all_lats)} points")
    return tracks


def filter_stations_by_swot_proximity(df: pd.DataFrame, swot_tracks: dict, max_dist_km: float = SWOT_TRACK_DISTANCE_KM) -> tuple[pd.DataFrame, pd.DataFrame]:
    near_mask, min_distances = [], []
    for _, row in df.iterrows():
        lat_tg, lon_tg = row["latitude"], row["longitude"]
        min_dist = np.inf
        for pass_num, (track_lats, track_lons) in swot_tracks.items():
            dists = haversine(track_lons, track_lats, lon_tg, lat_tg)
            min_dist = min(min_dist, np.min(dists))
        near_mask.append(min_dist <= max_dist_km)
        min_distances.append(min_dist)

    df = df.copy()
    df["swot_track_distance_km"] = min_distances
    return df[near_mask].copy(), df[~np.array(near_mask)].copy()


def plot_calval_swot_tracks(df_calval: pd.DataFrame, swot_tracks: dict) -> None:
    df_near, df_far = filter_stations_by_swot_proximity(df_calval, swot_tracks)
    fig = plt.figure(figsize=(14, 10))
    ax = _make_merc_ax(fig, 111, title="CalVal Phase: Stations and SWOT Tracks")

    for pass_num, (lats, lons) in swot_tracks.items():
        idx = np.argsort(lats)
        ax.plot(lons[idx], lats[idx], "b-", lw=1.5, alpha=0.5, transform=ccrs.PlateCarree(), zorder=4)

    if len(df_far) > 0:
        ax.scatter(df_far["longitude"], df_far["latitude"], c="#d62728", s=40, edgecolors="k", linewidths=0.5,
                   transform=ccrs.PlateCarree(), zorder=6, label=f"Far from SWOT (>{SWOT_TRACK_DISTANCE_KM:.0f} km, N={len(df_far)})")

    if len(df_near) > 0:
        ax.scatter(df_near["longitude"], df_near["latitude"], c="#2ca02c", s=50, edgecolors="k", linewidths=0.5,
                   transform=ccrs.PlateCarree(), zorder=7, label=f"Near SWOT (<{SWOT_TRACK_DISTANCE_KM:.0f} km, N={len(df_near)})")

    ax.legend(loc="lower left", fontsize=FS-1)

    fname = PATHS["output"] / "fig11_calval_swot_tracks"
    fig.savefig(fname.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}.png")


def plot_calval_swot_impact(df_calval: pd.DataFrame, swot_tracks: dict) -> None:
    # "
    # Boxplots PI Near vs Far SWOT + persistencia de estadísticas.
    # Versión extendida (patch P3_patch_swot_analysis.py)."

    from P3_patch_swot_analysis import plot_calval_swot_impact_extended
    plot_calval_swot_impact_extended(
        df_calval=df_calval,
        swot_tracks=swot_tracks,
        RESULTS=RESULTS,
        PATHS=PATHS,
        compute_improvement_metrics=compute_improvement_metrics,
        filter_stations_by_swot_proximity=filter_stations_by_swot_proximity,
        SWOT_TRACK_DISTANCE_KM=SWOT_TRACK_DISTANCE_KM,
        REGION_ORDER=REGION_ORDER,
        FS=FS,
    )

# def plot_calval_swot_impact(df_calval: pd.DataFrame, swot_tracks: dict) -> None:
#     df_near, df_far = filter_stations_by_swot_proximity(df_calval, swot_tracks)
#     df_near = compute_improvement_metrics(df_near)
#     df_far = compute_improvement_metrics(df_far)

#     fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
#     metrics = [("PI_RMSD", "PI$_{RMSD}$ [%]"), ("PI_Corr", "PI$_{Corr}$ [%]")]

#     for ax, (metric, label) in zip(axes, metrics):
#         data_near = df_near[metric].dropna().values
#         data_far = df_far[metric].dropna().values

#         bp = ax.boxplot([data_near, data_far],
#                         tick_labels=[f"Near SWOT\n(<{SWOT_TRACK_DISTANCE_KM:.0f} km)\nN={len(data_near)}",
#                                      f"Far from SWOT\n(>{SWOT_TRACK_DISTANCE_KM:.0f} km)\nN={len(data_far)}"],
#                         patch_artist=True, widths=0.6, zorder=5)

#         bp["boxes"][0].set_facecolor("#2ca02c")
#         bp["boxes"][1].set_facecolor("#d62728")

#         # --- CÁLCULO DE OVERALL PI (anteriormente Macro PI) ---
#         if metric == "PI_RMSD":
#             my_n = df_near["rmsd_my"].median()
#             mi_n = df_near["rmsd_miost"].median()
#             overall_near = 100 * (my_n - mi_n) / my_n if my_n != 0 else np.nan
#             my_f = df_far["rmsd_my"].median()
#             mi_f = df_far["rmsd_miost"].median()
#             overall_far = 100 * (my_f - mi_f) / my_f if my_f != 0 else np.nan
#         else:
#             my_n = df_near["corr_my"].median()
#             mi_n = df_near["corr_miost"].median()
#             overall_near = 100 * (mi_n - my_n) / (1 - my_n) if my_n != 1 else np.nan
#             my_f = df_far["corr_my"].median()
#             mi_f = df_far["corr_miost"].median()
#             overall_far = 100 * (mi_f - my_f) / (1 - my_f) if my_f != 1 else np.nan

#         # Línea discontinua azul eliminada según solicitud del usuario

#         ax.axhline(0, color="k", ls="--", lw=1.2, zorder=2)
#         ax.axhspan(0, ax.get_ylim()[1], alpha=0.08, color="green", zorder=0)
#         ax.axhspan(ax.get_ylim()[0], 0, alpha=0.08, color="red", zorder=0)

#         if len(data_near) > 0 and len(data_far) > 0:
#             stat, pval = stats.mannwhitneyu(data_near, data_far, alternative="two-sided")
#             sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
            
#             # Texto con estadísticas y Overall PI
#             mean_near = np.mean(data_near)
#             median_near = np.median(data_near)
#             mean_far = np.mean(data_far)
#             median_far = np.median(data_far)
            
#             stats_text = (f"M-W Test: p = {pval:.3e} {sig}\n"
#                         #  f"Overall PI Near: {overall_near:+.1f}%\n"
#                         #  f"Overall PI Far: {overall_far:+.1f}%\n"
#                          f"Mean Near: {mean_near:+.1f}% | Median: {median_near:+.1f}%\n"
#                          f"Mean Far: {mean_far:+.1f}% | Median: {median_far:+.1f}%")
         
#             ax.text(0.5, 0.03, stats_text,
#                     transform=ax.transAxes, ha="center", va="bottom", fontsize=FS-2,
#                     bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.5", alpha=0.9), zorder=25)

#         ax.set_ylabel(label, fontsize=FS)
#         ax.set_title(f"{label} – CalVal Phase", fontsize=FS, fontweight="bold")
#         ax.grid(True, linewidth=0.4, alpha=0.5, axis="y")

#     fname = PATHS["output"] / "fig12_calval_swot_impact"
#     fig.savefig(fname.with_suffix(".png"), dpi=300, bbox_inches="tight")
#     plt.close(fig)
#     print(f"  Saved {fname}.png")



def plot_calval_swot_impact_by_region(df_calval: pd.DataFrame, swot_tracks: dict) -> None:
    df_near, df_far = filter_stations_by_swot_proximity(df_calval, swot_tracks)
    df_near = compute_improvement_metrics(df_near)
    df_far = compute_improvement_metrics(df_far)

    regions_available = [r for r in REGION_ORDER if len(df_near[df_near["region"] == r]) >= 3 and len(df_far[df_far["region"] == r]) >= 3]

    if not regions_available: return

    n_regions = len(regions_available)
    fig, axes = plt.subplots(n_regions, 2, figsize=(14, 4 * n_regions), constrained_layout=True)
    if n_regions == 1: axes = axes.reshape(1, -1)

    metrics = [("PI_RMSD", "PI$_{RMSD}$ [%]"), ("PI_Corr", "PI$_{Corr}$ [%]")]

    for row, region in enumerate(regions_available):
        df_near_reg = df_near[df_near["region"] == region]
        df_far_reg = df_far[df_far["region"] == region]

        for col, (metric, label) in enumerate(metrics):
            ax = axes[row, col]
            data_near = df_near_reg[metric].dropna().values
            data_far = df_far_reg[metric].dropna().values

            bp = ax.boxplot([data_near, data_far], tick_labels=[f"Near SWOT\nN={len(data_near)}", f"Far SWOT\nN={len(data_far)}"], patch_artist=True, widths=0.6, zorder=5)
            bp["boxes"][0].set_facecolor(REGION_COLORS[region])
            bp["boxes"][0].set_alpha(0.8)
            bp["boxes"][1].set_facecolor(REGION_COLORS[region])
            bp["boxes"][1].set_alpha(0.4)

            # --- CÁLCULO DE OVERALL PI (anteriormente Macro PI) ---
            if metric == "PI_RMSD":
                my_n = df_near_reg["rmsd_my"].median()
                mi_n = df_near_reg["rmsd_miost"].median()
                overall_near = 100 * (my_n - mi_n) / my_n if my_n != 0 else np.nan
                my_f = df_far_reg["rmsd_my"].median()
                mi_f = df_far_reg["rmsd_miost"].median()
                overall_far = 100 * (my_f - mi_f) / my_f if my_f != 0 else np.nan
            else:
                my_n = df_near_reg["corr_my"].median()
                mi_n = df_near_reg["corr_miost"].median()
                overall_near = 100 * (mi_n - my_n) / (1 - my_n) if my_n != 1 else np.nan
                my_f = df_far_reg["corr_my"].median()
                mi_f = df_far_reg["corr_miost"].median()
                overall_far = 100 * (mi_f - my_f) / (1 - my_f) if my_f != 1 else np.nan

            # --- CÁLCULO DE MICRO PI (mediana de las mejoras individuales) ---
            if metric == "PI_RMSD":
                micro_pi_near = np.median(100 * (df_near_reg["rmsd_my"] - df_near_reg["rmsd_miost"]) / df_near_reg["rmsd_my"])
                micro_pi_far = np.median(100 * (df_far_reg["rmsd_my"] - df_far_reg["rmsd_miost"]) / df_far_reg["rmsd_my"])
            else:
                micro_pi_near = np.median(100 * (df_near_reg["corr_miost"] - df_near_reg["corr_my"]) / (1 - df_near_reg["corr_my"]))
                micro_pi_far = np.median(100 * (df_far_reg["corr_miost"] - df_far_reg["corr_my"]) / (1 - df_far_reg["corr_my"]))

            # Línea discontinua azul eliminada según solicitud del usuario

            ax.axhline(0, color="k", ls="--", lw=1.2, zorder=2)
            ylim = ax.get_ylim()
            ax.axhspan(0, ylim[1], alpha=0.05, color="green", zorder=0)
            ax.axhspan(ylim[0], 0, alpha=0.05, color="red", zorder=0)

            if len(data_near) > 0 and len(data_far) > 0:
                try:
                    stat, pval = stats.mannwhitneyu(data_near, data_far, alternative="two-sided")
                    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
                    stats_text = (f"M-W Test p = {pval:.3f} {sig}\n"
                                #   f"Overall PI Near: {overall_near:+.1f}%\n"
                                #   f"Overall PI Far: {overall_far:+.1f}%\n"
                                  f"Micro PI Near: {micro_pi_near:+.1f}%\n"
                                  f"Micro PI Far: {micro_pi_far:+.1f}%")
                    ax.text(0.5, 0.98, stats_text, transform=ax.transAxes, ha="center", va="top", fontsize=FS-2, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.9), zorder=25)
                except ValueError:
                    pass

            ax.set_ylabel(label, fontsize=FS)
            if row == 0: ax.set_title(label, fontsize=FS, fontweight="bold")
            if col == 0: ax.text(-0.15, 0.5, region, transform=ax.transAxes, fontsize=FS, fontweight="bold", rotation=90, va="center", ha="right")
            ax.grid(True, linewidth=0.4, alpha=0.5, axis="y")

    fig.suptitle("SWOT Impact by Maritime Region – CalVal Phase", fontsize=FS+2, fontweight="bold", y=0.995)
    fname = PATHS["output"] / "fig12b_calval_swot_impact_by_region"
    fig.savefig(fname.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}.png")

# =============================================================================
# 10. ANÁLISIS COMBINADO (CalVal + Science)
# =============================================================================

def plot_combined_analysis(df_combined: pd.DataFrame) -> None:
    df = compute_improvement_metrics(df_combined)
    fig = plt.figure(figsize=(12, 9))
    ax = _make_merc_ax(fig, 111, title="Combined (All phases): PI$_{RMSD}$ [%]")

    lons, lats, vals = df["longitude"].values, df["latitude"].values, df["PI_RMSD"].values
    sc = _scatter_map(ax, lons, lats, vals, cmap="RdYlGn", vmin=PI_CLIM[0], vmax=PI_CLIM[1], s=30)

    n_better = (vals > 0).sum()
    n_total = np.isfinite(vals).sum()
    pct = 100 * n_better / n_total

    ax.text(0.02, 0.97, f"Mean: {np.nanmean(vals):+.2f} | Median: {np.nanmedian(vals):+.2f}",
            transform=ax.transAxes, fontsize=FS, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.85),
            zorder=10)

    ax.text(0.02, 0.03, f"{TEST_LABEL} better: {pct:.0f}% ({n_better}/{n_total})",
            transform=ax.transAxes, fontsize=FS-1, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.85), zorder=10)

    _add_cbar(fig, sc, ax, label="PI$_{RMSD}$ [%]", shrink=0.75, pad=0.02)

    fname = PATHS["output"] / "fig15_combined_skill_map"
    fig.savefig(fname.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}.png")


# =============================================================================
# 11. ANÁLISIS DE VARIABILIDAD ESPACIAL Y MÁRGENES DE ERROR
# =============================================================================

def load_spatial_stats(phase: str) -> pd.DataFrame | None:
    # --- BUG FIX (2026-08-10): igual que el fix de 2026-07-03 para swot_l3/coverage_csv,
    # esta ruta debe resolverse contra DATA_BASE_DIR (el --base-dir de P1, donde vive
    # results/spatial_stats/), NO contra BASE_DIR (carpeta del propio script P2, que
    # puede ser un subdirectorio como PIPELINE_FINAL_LIMPIO/ sin esa carpeta).
    spatial_stats_path = DATA_BASE_DIR / "results/spatial_stats" / f"spatial_stats_comparison_{phase}.csv"
    if not spatial_stats_path.exists(): return None
    return pd.read_csv(spatial_stats_path)


def merge_with_spatial_stats(df: pd.DataFrame, phase: str) -> pd.DataFrame:
    df_spatial = load_spatial_stats(phase)
    if df_spatial is None: return df
    return df.merge(
        df_spatial[["station", "mean_spatial_std_my", "mean_spatial_std_miost", "delta_spatial_std", "mean_n_points_my", "mean_n_points_miost", "mean_iqr_my", "mean_iqr_miost"]],
        on="station", how="left"
    )


def plot_variability_analysis(df_calval: pd.DataFrame, df_science: pd.DataFrame) -> None:
    df_calval_sp = merge_with_spatial_stats(df_calval.copy(), "CalVal")
    df_science_sp = merge_with_spatial_stats(df_science.copy(), "Science")

    if "mean_spatial_std_my" not in df_calval_sp.columns: return

    # Scatter RMSD vs Spatial STD
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), constrained_layout=True)
    for col, (df, phase) in enumerate([(df_calval_sp, "CalVal"), (df_science_sp, "Science")]):
        for row, (product_key, product) in enumerate([("my", BASELINE_LABEL), ("miost", TEST_LABEL)]):
            ax = axes[row, col]
            x, y = df[f"rmsd_{product_key}"].values, df[f"mean_spatial_std_{product_key}"].values

            for region in REGION_ORDER:
                mask = (df["region"] == region) & np.isfinite(x) & np.isfinite(y)
                if mask.any(): ax.scatter(x[mask], y[mask], c=REGION_COLORS[region], s=40, alpha=0.7, edgecolors="white", linewidths=0.3, label=region, zorder=4)

            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() > 5:
                slope, intercept, r, pval, _ = stats.linregress(x[valid], y[valid])
                xfit = np.linspace(np.nanmin(x[valid]), np.nanmax(x[valid]), 100)
                ax.plot(xfit, slope * xfit + intercept, "r-", lw=2, zorder=5, label=f"r = {r:.3f} (p = {pval:.2e})")

            ax.set_xlabel(f"RMSD {product} (cm)", fontsize=FS)
            ax.set_ylabel(f"Spatial STD {product} (cm)", fontsize=FS)
            ax.set_title(f"{'(abcd)'[row*2+col]}) {product} – {PHASE_LABELS[phase]}", fontsize=FS, fontweight="bold")
            ax.grid(True, linewidth=0.4, alpha=0.5)
            if row == 0 and col == 0: ax.legend(fontsize=FS-2, loc="upper left", ncol=2)

    fig.suptitle(f"Relationship: RMSD vs Spatial Variability ({FIXED_RADIUS_KM} km radius)", fontsize=FS+1, fontweight="bold", y=1.01)
    fname = PATHS["output"] / "fig19_rmsd_vs_spatial_std"
    fig.savefig(fname.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}.png")

    # Scatter RMSD MY vs MIOST with Error Bars
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    for ax, (df, phase) in zip(axes, [(df_calval_sp, "CalVal"), (df_science_sp, "Science")]):
        x, y = df["rmsd_my"].values, df["rmsd_miost"].values
        xerr, yerr = df["mean_spatial_std_my"].values, df["mean_spatial_std_miost"].values

        for region in REGION_ORDER:
            mask = (df["region"] == region) & np.isfinite(x) & np.isfinite(y)
            if mask.any(): ax.errorbar(x[mask], y[mask], xerr=xerr[mask], yerr=yerr[mask], fmt='o', markersize=6, color=REGION_COLORS[region], ecolor=REGION_COLORS[region], alpha=0.5, capsize=0, elinewidth=0.8, markeredgewidth=0, label=region, zorder=4)

        limit_min = max(0, min(np.nanmin(x), np.nanmin(y)) - 0.5)
        limit_max = max(np.nanmax(x), np.nanmax(y)) + 0.5
        ax.plot([limit_min, limit_max], [limit_min, limit_max], "k--", lw=1.5, zorder=2, alpha=0.6)
        ax.fill_between([limit_min, limit_max], limit_min, [limit_min, limit_max], alpha=0.05, color="green", zorder=1)

        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() > 0:
            pct_better = 100 * (y[valid] < x[valid]).sum() / valid.sum()
            stats_text = f"Mean error bar MY: {np.nanmean(xerr[valid]):.2f} cm\nMean error bar MIOST: {np.nanmean(yerr[valid]):.2f} cm\n{TEST_LABEL} better: {pct_better:.0f}% ({(y[valid] < x[valid]).sum()}/{valid.sum()})"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=FS-1, va="top", ha="left", bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.5", alpha=0.85), zorder=10)

        ax.set_xlabel("RMSD$_{base}$ (cm)", fontsize=FS)
        ax.set_ylabel("RMSD$_{test}$ (cm)", fontsize=FS)
        ax.set_title(f"RMSD Comparison with Spatial Uncertainty – {PHASE_LABELS[phase]}", fontsize=FS, fontweight="bold")
        ax.grid(True, linewidth=0.4, alpha=0.5, zorder=0)
        ax.legend(fontsize=FS-2, loc="lower right", ncol=2)

    fname = PATHS["output"] / "fig20_rmsd_scatter_with_error_bars"
    fig.savefig(fname.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}.png")

    # Map of delta_spatial_std (MIOST - MY)
    fig = plt.figure(figsize=(16, 7))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.08, left=0.03, right=0.97, top=0.92, bottom=0.06)
    for col, (df, phase) in enumerate([(df_calval_sp, "CalVal"), (df_science_sp, "Science")]):
        delta_std = df["delta_spatial_std"].values
        ax = _make_merc_ax(fig, gs[0, col], title=f"({['a', 'b'][col]}) Δ Spatial STD (MIOST − MY) – {PHASE_LABELS[phase]}")
        sc = _scatter_map(ax, df["longitude"].values, df["latitude"].values, delta_std, cmap="PuOr_r", vmin=-0.1, vmax=0.1, s=35)
        
        valid = np.isfinite(delta_std).sum()
        if valid > 0: ax.text(0.02, 0.03, f"Mean: {np.nanmean(delta_std):+.3f} cm\n{TEST_LABEL} higher variability: {100 * (delta_std > 0).sum() / valid:.0f}%", transform=ax.transAxes, fontsize=FS, va="bottom", ha="left", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.85), zorder=10)
        _add_cbar(fig, sc, ax, label="Δ Spatial STD (cm)\n(+) = " + TEST_LABEL + " higher", shrink=0.75, pad=0.02)

    fname = PATHS["output"] / "fig21_delta_spatial_std_maps"
    fig.savefig(fname.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"  Saved {fname}.png")

    # Regional bars with error bars
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    for ax, (df, phase) in zip(axes, [(df_calval_sp, "CalVal"), (df_science_sp, "Science")]):
        df_ss = compute_improvement_metrics(df)
        region_stats = []
        for region in REGION_ORDER:
            df_reg = df_ss[df_ss["region"] == region]
            if len(df_reg) >= 3:
                spatial_err = np.sqrt(df_reg["mean_spatial_std_my"].mean()**2 + df_reg["mean_spatial_std_miost"].mean()**2) / np.sqrt(len(df_reg))
                region_stats.append((region, df_reg["delta_RMSD"].mean(), spatial_err, len(df_reg)))

        if not region_stats: continue
        regions, means, errors, n_stations = zip(*region_stats)
        y_pos = np.arange(len(regions))
        
        ax.barh(y_pos, means, xerr=errors, height=0.6, color=[REGION_COLORS[r] for r in regions], edgecolor="0.2", linewidth=0.7, error_kw=dict(ecolor="0.3", lw=1.5, capsize=4, capthick=1.5))
        ax.axvline(0, color="k", ls="-", lw=1.2, zorder=3)
        ax.axvspan(-2, 0, alpha=0.08, color="red", zorder=0)
        ax.axvspan(0, 2, alpha=0.08, color="green", zorder=0)

        for i, (mean, err, n) in enumerate(zip(means, errors, n_stations)):
            x_pos = mean + err + 0.05 if mean >= 0 else mean - err - 0.05
            ax.text(x_pos, y_pos[i], f"N={n}", va="center", ha="left" if mean >= 0 else "right", fontsize=FS-2, color="0.3")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(regions, fontsize=FS-1)
        ax.set_xlabel("ΔRMSD = RMSD$_{base}$ − RMSD$_{test}$ (cm)", fontsize=FS)
        ax.set_title(f"Regional ΔRMSD with Spatial Uncertainty – {PHASE_LABELS[phase]}", fontsize=FS, fontweight="bold")
        ax.grid(True, linewidth=0.4, alpha=0.5, axis="x")
        ax.spines[["top", "right"]].set_visible(False)

    fig.text(0.5, -0.02, f"Error bars: propagated spatial std within {FIXED_RADIUS_KM} km radius", ha="center", fontsize=FS-1, style="italic")
    fname = PATHS["output"] / "fig22_regional_bars_with_spatial_error"
    fig.savefig(fname.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}.png")


def plot_variability_difference_histograms(df_calval: pd.DataFrame, df_science: pd.DataFrame) -> None:
    df_calval_sp = merge_with_spatial_stats(df_calval.copy(), "CalVal")
    df_science_sp = merge_with_spatial_stats(df_science.copy(), "Science")
    if "delta_spatial_std" not in df_calval_sp.columns: return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, (df, phase) in zip(axes, [(df_calval_sp, "CalVal"), (df_science_sp, "Science")]):
        vals = df["delta_spatial_std"].dropna().values
        if len(vals) == 0: continue

        n, bins_edges, patches = ax.hist(vals, bins=30, color="steelblue", edgecolor="white", alpha=0.8, zorder=3)
        for patch, left_edge in zip(patches, bins_edges[:-1]):
            patch.set_facecolor("#9467bd" if left_edge >= 0 else "#17becf")

        ax.axvline(0, color="k", ls="--", lw=1.5, zorder=4)
        ax.text(0.97, 0.97, f"Mean: {np.mean(vals):+.3f} cm\nMedian: {np.median(vals):+.3f} cm\n{TEST_LABEL} higher: {100 * (vals > 0).sum() / len(vals):.0f}%\nN = {len(vals)}", transform=ax.transAxes, fontsize=FS-1, va="top", ha="right", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.9))
        ax.set_xlabel("Δ Spatial STD = STD$_{test}$ − STD$_{base}$ (cm)", fontsize=FS)
        ax.set_ylabel("Frequency", fontsize=FS)
        ax.set_title(f"Spatial Variability Difference – {PHASE_LABELS[phase]}", fontsize=FS, fontweight="bold")
        ax.grid(True, linewidth=0.4, alpha=0.5, axis="y")

    fname = PATHS["output"] / "fig23_variability_difference_histograms"
    fig.savefig(fname.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}.png")



def plot_difference_maps_combined(df_calval: pd.DataFrame, df_science: pd.DataFrame) -> None:
    df_calval_sp = merge_with_spatial_stats(df_calval.copy(), "CalVal")
    df_science_sp = merge_with_spatial_stats(df_science.copy(), "Science")

    df_calval_ss = compute_improvement_metrics(df_calval_sp)
    df_science_ss = compute_improvement_metrics(df_science_sp)

    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.15, wspace=0.08, left=0.03, right=0.97, top=0.93, bottom=0.04)
    datasets = [(df_calval_ss, "CalVal"), (df_science_ss, "Science")]

    metrics_config = [
        {"col": "delta_RMSD", "title": "ΔRMSD (base − test)", "label": "ΔRMSD (cm)\n(+) = " + TEST_LABEL + " better", "vmin": -2, "vmax": 2, "pos_meaning": "" + TEST_LABEL + " better"},
        {"col": "delta_std", "title": f"ΔSTD ({BASELINE_LABEL} − {TEST_LABEL})", "label": f"ΔSTD (cm)\n(+) = {BASELINE_LABEL} less variable", "vmin": -2, "vmax": 2, "pos_meaning": f"{BASELINE_LABEL} less variable"},
        {"col": "delta_Corr", "title": "ΔCorr (test − base)", "label": "ΔCorr\n(+) = " + TEST_LABEL + " better", "vmin": -0.15, "vmax": 0.15, "pos_meaning": "" + TEST_LABEL + " better"},
    ]
    panel_labels = [["(a)", "(b)", "(c)"], ["(d)", "(e)", "(f)"]]

    for row, (df, phase) in enumerate(datasets):
        lons, lats = df["longitude"].values, df["latitude"].values
        df["delta_std"] = df["std_my"].values - df["std_miost"].values if "std_my" in df.columns else np.nan

        for col_idx, metric in enumerate(metrics_config):
            if metric["col"] not in df.columns: continue
            vals = df[metric["col"]].values
            ax = _make_merc_ax(fig, gs[row, col_idx], title=f"{panel_labels[row][col_idx]} {metric['title']} – {PHASE_LABELS[phase]}")
            sc = _scatter_map(ax, lons, lats, vals, cmap="seismic", vmin=metric["vmin"], vmax=metric["vmax"], s=30)

            valid_vals = vals[np.isfinite(vals)]
            if len(valid_vals) > 0:
                ax.text(0.02, 0.03, f"Mean: {np.mean(valid_vals):+.3f} | Median: {np.median(valid_vals):+.3f}\n{metric['pos_meaning']}: {100 * (valid_vals > 0).sum() / len(valid_vals):.0f}% ({(valid_vals > 0).sum()}/{len(valid_vals)})",
                        transform=ax.transAxes, fontsize=FS-2, va="bottom", ha="left", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.85), zorder=10)
            _add_cbar(fig, sc, ax, label=metric["label"], shrink=0.75, pad=0.02)

    fig.suptitle(f"Difference Maps: {BASELINE_LABEL} vs {TEST_LABEL} (ΔRMSD, ΔSTD, ΔCorr)", fontsize=FS+2, fontweight="bold", y=0.98)
    fname = PATHS["output"] / "fig24_difference_maps_combined"
    fig.savefig(fname.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}.png")


# =============================================================================
# FUNCIÓN PARA IMPRIMIR LAS MÉTRICAS EXACTAS PARA EL TEXTO DEL ARTÍCULO
# =============================================================================

# =============================================================================
# FUNCIÓN PARA IMPRIMIR LAS MÉTRICAS EXACTAS PARA EL TEXTO DEL ARTÍCULO
# =============================================================================

# =============================================================================
# FUNCIÓN PARA IMPRIMIR LAS MÉTRICAS EXACTAS PARA EL TEXTO DEL ARTÍCULO
# =============================================================================

def print_article_metrics(df_calval: pd.DataFrame, swot_tracks: dict) -> None:
    print("\n" + "="*75)
    print(" 📝 MÉTRICAS PARA EL PAPER: IMPACTO DE KaRIn (NEAR-SWOT vs FAR-SWOT)")
    print("="*75)

    df_near, df_far = filter_stations_by_swot_proximity(df_calval, swot_tracks)
    df_near = compute_improvement_metrics(df_near)
    df_far = compute_improvement_metrics(df_far)

    if len(df_near) == 0 or len(df_far) == 0:
        print("⚠️ Faltan estaciones Near o Far para hacer la comparativa completa.")
        return

    rmsd_miost_near = df_near["rmsd_miost"].median()
    rmsd_my_near = df_near["rmsd_my"].median()
    rmsd_miost_far = df_far["rmsd_miost"].median()
    rmsd_my_far = df_far["rmsd_my"].median()

    pi_rmsd_near = df_near["PI_RMSD"].median()
    pi_rmsd_far = df_far["PI_RMSD"].median()
    
    corr_miost_near = df_near["corr_miost"].median()
    corr_my_near = df_near["corr_my"].median()
    corr_miost_far = df_far["corr_miost"].median()
    corr_my_far = df_far["corr_my"].median()

    valid_corr_near = df_near["delta_Corr"].dropna()
    valid_corr_far = df_far["delta_Corr"].dropna()
    
    pct_gain_near = 100 * (valid_corr_near > 0).sum() / len(valid_corr_near) if len(valid_corr_near) > 0 else np.nan
    pct_gain_far = 100 * (valid_corr_far > 0).sum() / len(valid_corr_far) if len(valid_corr_far) > 0 else np.nan

    print("\n--- DESGLOSE REGIONAL RMSD (NEAR vs FAR) ---")
    
    def print_regional_stats(df_near, df_far, region_name):
        df_near_reg = df_near[df_near["region"] == region_name]
        df_far_reg = df_far[df_far["region"] == region_name]
        
        my_near = df_near_reg["rmsd_my"].median() if len(df_near_reg) > 0 else np.nan
        miost_near = df_near_reg["rmsd_miost"].median() if len(df_near_reg) > 0 else np.nan
        pi_near = 100 * (my_near - miost_near) / my_near if len(df_near_reg) > 0 else np.nan
        
        my_far = df_far_reg["rmsd_my"].median() if len(df_far_reg) > 0 else np.nan
        miost_far = df_far_reg["rmsd_miost"].median() if len(df_far_reg) > 0 else np.nan
        pi_far = 100 * (my_far - miost_far) / my_far if len(df_far_reg) > 0 else np.nan
        
        print(f"[{region_name}]")
        print(f"  NEAR-SWOT (N={len(df_near_reg)}): MY = {my_near:.2f} cm | MIOST = {miost_near:.2f} cm -> Mejora: {pi_near:+.1f}%")
        print(f"  FAR-SWOT  (N={len(df_far_reg)}): MY = {my_far:.2f} cm | MIOST = {miost_far:.2f} cm -> Mejora: {pi_far:+.1f}%")
        # SWOT proximity CSV (generado por plot_calval_swot_impact extendida)
        
        swot_csv = PATHS["output"] / "swot_proximity_tests.csv"
        if swot_csv.exists():
            print(f"  (swot_proximity_tests.csv ya guardado por plot_calval_swot_impact)")
        else:
            print(f"  Warning: swot_proximity_tests.csv no encontrado. "
                f"¿Se ejecutó plot_calval_swot_impact con el parche?")
        return my_near, miost_near, pi_near, my_far, miost_far, pi_far

    baltic_my_n, baltic_mi_n, baltic_pi_n, baltic_my_f, baltic_mi_f, baltic_pi_f = print_regional_stats(df_near, df_far, "Baltic Sea")
    nws_my_n, nws_mi_n, nws_pi_n, nws_my_f, nws_mi_f, nws_pi_f = print_regional_stats(df_near, df_far, "North West Shelf")
    med_my_n, med_mi_n, med_pi_n, med_my_f, med_mi_f, med_pi_f = print_regional_stats(df_near, df_far, "Mediterranean Sea")

    print(f"\n[DATOS BASE DEL MUESTREO]")
    print(f"Estaciones evaluadas: {len(df_near)} (Near-SWOT) vs {len(df_far)} (Far-SWOT)")
    
    print("\n--- PROPUESTA DE TEXTO (Párrafo 1: RMSD) ---")
    print(f"Integration of SWOT's KaRIn observations into DUACS L4 demonstrates statistically significant coastal improvement. "
          f"During the CalVal phase (March-July 2023), analysis of stations within the SWOT orbital coverage (Near-SWOT, N={len(df_near)}) "
          f"reveals a significant error reduction compared to those outside the coverage (Far-SWOT, N={len(df_far)}). "
          f"For swath-proximal stations, the experimental L4 product achieves a median relative RMSD improvement of +{pi_rmsd_near:.1f}% "
          f"(median RMSD: {rmsd_miost_near:.2f} cm vs MY: {rmsd_my_near:.2f} cm). In contrast, stations outside the KaRIn swath "
          f"only show a median improvement of +{pi_rmsd_far:.1f}% (median RMSD: {rmsd_miost_far:.2f} cm vs MY: {rmsd_my_far:.2f} cm), "
          f"confirming the direct added value of the wide-swath sensor.")

    print("\n--- PROPUESTA DE TEXTO (Párrafo 2: Correlación y Regiones) ---")
    print(f"Correlation metrics similarly improve: within the KaRIn footprint, the experimental L4 product achieves a median correlation of "
          f"{corr_miost_near:.3f} versus {corr_my_near:.3f} for MY, with {pct_gain_near:.1f}% of swath-proximal stations exhibiting positive skill gains "
          f"(compared to {pct_gain_far:.1f}% for Far-SWOT stations).")
    print(f"\nRegional heterogeneity within the SWOT-sampled zone becomes evident when evaluating the macroscopic relative error reduction "
          f"(calculated as the reduction of median MIOST RMSD relative to median MY RMSD). For instance, in the North West Shelf, "
          f"near-swath stations exhibit a drastic error reduction of {nws_pi_n:+.1f}% (median MIOST: {nws_mi_n:.2f} cm vs MY: {nws_my_n:.2f} cm), "
          f"whereas far-swath improvements are minimal ({nws_pi_f:+.1f}%). Conversely, regions like the Mediterranean show a degradation in near-swath "
          f"performance of {med_pi_n:+.1f}% (median MIOST: {med_mi_n:.2f} cm vs MY: {med_my_n:.2f} cm), and the Baltic Sea remains largely insensitive "
          f"to swath proximity ({baltic_pi_n:+.1f}% in near-SWOT vs {baltic_pi_f:+.1f}% in far-SWOT).")
    print("="*75 + "\n")



# =============================================================================
# FUNCIÓN PARA GUARDAR RESULTADOS NUMÉRICOS
# =============================================================================

def save_all_numerical_results() -> None:
    json_path = PATHS["output"] / "numerical_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved {json_path}")

    skill_data = []
    for phase, data in RESULTS.get("improvement_metrics", {}).items():
        if "stations" in data:
            for i, station in enumerate(data["stations"]):
                skill_data.append({
                    "phase": phase,
                    "station": station,
                    "PI_RMSD": data["PI_RMSD"][i] if i < len(data["PI_RMSD"]) else None,
                    "PI_Corr": data["PI_Corr"][i] if i < len(data["PI_Corr"]) else None,
                    "delta_RMSD": data["delta_RMSD"][i] if i < len(data["delta_RMSD"]) else None,
                    "delta_Corr": data["delta_Corr"][i] if i < len(data["delta_Corr"]) else None,
                })

    if skill_data:
        df_skill = pd.DataFrame(skill_data)
        csv_path = PATHS["output"] / "percentage_improvement_by_station.csv"
        df_skill.to_csv(csv_path, index=False, float_format="%.4f")
        print(f"  Saved {csv_path}")

    improvement_data = []
    for key, data in RESULTS.get("improvement_stats", {}).get("histograms", {}).items():
        phase, metric = key.rsplit("_", 1)
        improvement_data.append({"phase": phase, "metric": metric, **data})

    if improvement_data:
        df_improvement = pd.DataFrame(improvement_data)
        csv_path = PATHS["output"] / "improvement_stats_summary.csv"
        df_improvement.to_csv(csv_path, index=False, float_format="%.4f")
        print(f"  Saved {csv_path}")


# =============================================================================
# ANÁLISIS DE ALTA FRECUENCIA (HIGH-PASS)
# =============================================================================

def load_highpass_metrics(phase: str) -> pd.DataFrame:
    """
    Carga métricas HIGH-PASS del CSV si existen.
    Busca archivos con patrón: overall_metrics_HIGHPASS_PARALLEL_{phase}*.csv
    """
    fpattern = f"overall_metrics_HIGHPASS_PARALLEL_{phase}*.csv"
    files = list(PATHS["metrics"].glob(fpattern))
    
    if not files:
        print(f"    Warning: No high-pass metrics found for {phase}")
        return None
    
    path = files[0]
    df = pd.read_csv(path)
    print(f"    Loaded high-pass metrics from: {path.name}")

    # Capa de ALIAS para el par activo (baseline/test) en columnas _highpass y base.
    for metric in ["rmsd", "var", "std", "corr"]:
        for suf in ["_highpass", ""]:
            col_base = f"{metric}_{BASELINE}{suf}"
            col_test = f"{metric}_{TEST}{suf}"
            if col_base in df.columns:
                df[f"{metric}_my{suf}"] = df[col_base]
            if col_test in df.columns:
                df[f"{metric}_miost{suf}"] = df[col_test]
    return df


def compute_highpass_improvement_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula Percentage Improvement (PI) para el high-pass.
    Usa columnas con sufijo _highpass si existen, sino usa las estándar.
    """
    df = df.copy()

    # Buscar columnas con sufijo _highpass
    rmsd_my_col = 'rmsd_my_highpass' if 'rmsd_my_highpass' in df.columns else 'rmsd_my'
    rmsd_miost_col = 'rmsd_miost_highpass' if 'rmsd_miost_highpass' in df.columns else 'rmsd_miost'
    corr_my_col = 'corr_my_highpass' if 'corr_my_highpass' in df.columns else 'corr_my'
    corr_miost_col = 'corr_miost_highpass' if 'corr_miost_highpass' in df.columns else 'corr_miost'

    # PI_RMSD: mejora en RMSD (mayor es mejor para MIOST)
    df['PI_RMSD_HP'] = np.where(
        df[rmsd_my_col] > 0,
        ((df[rmsd_my_col] - df[rmsd_miost_col]) / df[rmsd_my_col]) * 100,
        np.nan
    )

    # PI_Corr: mejora en correlación (diferencia absoluta, no porcentual)
    df['PI_Corr_HP'] = df[corr_miost_col] - df[corr_my_col]

    return df


def plot_highpass_percentage_improvement_maps(df_calval: pd.DataFrame, df_science: pd.DataFrame) -> None:
    """
    Mapas 2×2 de Percentage Improvement HIGH-PASS:
      Filas: CalVal, Science
      Columnas: PI_RMSD_HP, PI_Corr_HP
    """
    fig = plt.figure(figsize=(13, 11))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.12, wspace=0.08,
                           left=0.03, right=0.97, top=0.93, bottom=0.04)

    datasets = [
        (df_calval, "CalVal"),
        (df_science, "Science"),
    ]

    panel_labels = [["(a)", "(b)"], ["(c)", "(d)"]]

    for row, (df, phase) in enumerate(datasets):
        if df is None or df.empty:
            continue
            
        df = compute_highpass_improvement_metrics(df)
        lons = df["longitude"].values
        lats = df["latitude"].values

        metrics = [
            ("PI_RMSD_HP", "PI$_{RMSD}$ [%] (High-Pass)\n(+) = " + TEST_LABEL + " better"),
            ("PI_Corr_HP", "PI$_{Corr}$ [adim.] (High-Pass)\n(+) = " + TEST_LABEL + " better"),
        ]

        for col, (pi_col, label) in enumerate(metrics):
                    vals = df[pi_col].values

                    metric_name = "PI RMSD HP" if "RMSD" in pi_col else "PI Corr HP"
                    title = f"{panel_labels[row][col]} {metric_name} – {phase}"

                    ax, gl = _make_merc_ax_1(fig, gs[row, col], title=title)  # ← desempaquetar

                    # Suprimir etiquetas X en la fila superior
                    if row == 0:
                        gl.bottom_labels = False

                    # Colormap y límites según métrica  ← Fix del bug de mayúsculas
                    if pi_col == "PI_Corr_HP":
                        sc = _scatter_map(ax, lons, lats, vals,
                                        cmap="PiYG", vmin=-0.05, vmax=0.05, s=30)
                    else:
                        sc = _scatter_map(ax, lons, lats, vals,
                                        cmap="RdYlGn",
                                        vmin=PI_CLIM[0], vmax=PI_CLIM[1], s=30)

                    n_better = (vals > 0).sum()
                    n_total  = np.isfinite(vals).sum()
                    pct = 100 * n_better / n_total if n_total > 0 else 0

                    ax.text(0.02, 0.97, f"Mean: {np.nanmean(vals):+.2f} | Median: {np.nanmedian(vals):+.2f}",
                            transform=ax.transAxes, fontsize=FS, va="top", ha="left",
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.85),
                            zorder=10)

                    ax.text(0.02, 0.03, f"{TEST_LABEL} better: {pct:.0f}% ({n_better}/{n_total})",
                            transform=ax.transAxes, fontsize=FS, va="bottom", ha="left",
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.85),
                            zorder=10)

                    _add_cbar(fig, sc, ax, label=label.split("\n")[0], shrink=0.80, pad=0.02)
    
    fname = PATHS["output"] / "fig07_percentage_improvement_maps_HIGHPASS"
    fig.savefig(fname.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"  Saved {fname.name}.png")


def plot_highpass_rmsd_maps(df_calval: pd.DataFrame, df_science: pd.DataFrame) -> None:
    """Mapas de RMSD absolutos (MY vs MIOST) para high-pass."""
    fig = plt.figure(figsize=(13, 11))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.12, wspace=0.08,
                           left=0.03, right=0.97, top=0.93, bottom=0.04)

    datasets = [
        (df_calval, "CalVal"),
        (df_science, "Science"),
    ]

    panel_labels = [["(a)", "(b)"], ["(c)", "(d)"]]

    for row, (df, phase) in enumerate(datasets):
        if df is None or df.empty:
            continue
            
        lons = df["longitude"].values
        lats = df["latitude"].values

        # MY
        rmsd_my = df['rmsd_my_highpass'].values if 'rmsd_my_highpass' in df.columns else df['rmsd_my'].values
        title = f"{panel_labels[row][0]} RMSD DUACS MY – {phase} (HIGH-PASS)"
        ax = _make_merc_ax(fig, gs[row, 0], title=title)
        sc = _scatter_map(ax, lons, lats, rmsd_my, cmap="viridis",
                          vmin=0, vmax=20, s=30)  # Ajustado límite para high-pass
        if len(rmsd_my) > 0:
            text_my = f"Median: {np.median(rmsd_my):.2f} cm"
            ax.text(0.98, 0.03, text_my, transform=ax.transAxes, fontsize=FS-2, va="bottom", ha="right", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.85), zorder=10)
        _add_cbar(fig, sc, ax, label="RMSD (cm)", shrink=0.80, pad=0.02)
        



        # MIOST
        rmsd_miost = df['rmsd_miost_highpass'].values if 'rmsd_miost_highpass' in df.columns else df['rmsd_miost'].values
        title = f"{panel_labels[row][1]} RMSD MIOST – {phase} (HIGH-PASS)"
        ax = _make_merc_ax(fig, gs[row, 1], title=title)
        sc = _scatter_map(ax, lons, lats, rmsd_miost, cmap="viridis",
                          vmin=0, vmax=20, s=30)
        text_miost = f"Median: {np.median(rmsd_miost):.2f} cm"

        if len(rmsd_miost) > 0:
            text_miost = f"Median: {np.median(rmsd_miost):.2f} cm"
            ax.text(0.98, 0.03, text_miost, transform=ax.transAxes, fontsize=FS, va="bottom", ha="right", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.85), zorder=10)
        _add_cbar(fig, sc, ax, label="RMSD (cm)", shrink=0.80, pad=0.02)

    fname = PATHS["output"] / "fig07b_rmsd_maps_HIGHPASS"
    fig.savefig(fname.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"  Saved {fname.name}.png")


def plot_highpass_improvement_histograms(df_calval: pd.DataFrame, df_science: pd.DataFrame) -> None:
    """
    Histogramas de mejora para high-pass (2×3 grid):
      Columnas: CalVal, Science, Combined
      Filas: PI_RMSD_HP, PI_Corr_HP
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    datasets = {
        "CalVal": df_calval,
        "Science": df_science,
        "Combined": pd.concat([df_calval, df_science], ignore_index=True) if df_calval is not None and df_science is not None else None
    }

    metrics = [
        ("PI_RMSD_HP", "PI$_{RMSD}$ [%] (High-Pass)"),
        ("PI_Corr_HP", "PI$_{Corr}$ [adim.] (High-Pass)")
    ]

    for row, (metric, label) in enumerate(metrics):
        for col, (phase, df) in enumerate(datasets.items()):
            ax = axes[row, col]
            
            if df is None or df.empty:
                ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{label} – {phase}")
                continue
            
            df = compute_highpass_improvement_metrics(df)
            vals = df[metric].dropna().values

            if len(vals) == 0:
                ax.text(0.5, 0.5, "No valid data", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{label} – {phase}")
                continue

            # Histograma con colores verde (mejora) y rojo (empeora)
            vals_better = vals[vals > 0]
            vals_worse = vals[vals <= 0]

            ax.hist(vals_better, bins=20, color="green", alpha=0.6, edgecolor="k", linewidth=0.5, label=f"{TEST_LABEL} better (N={len(vals_better)})")
            ax.hist(vals_worse, bins=20, color="red", alpha=0.6, edgecolor="k", linewidth=0.5, label=f"{TEST_LABEL} worse (N={len(vals_worse)})")

            ax.axvline(0, color="k", ls="--", lw=1.5, zorder=5)
            ax.axvline(np.mean(vals), color="blue", ls=":", lw=2, zorder=10, label=f"Mean: {np.mean(vals):+.1f}%")
            ax.axvline(np.median(vals), color="darkblue", ls="-.", lw=2, zorder=10, label=f"Median: {np.median(vals):+.1f}%")

            ax.set_xlabel(label, fontsize=FS)
            ax.set_ylabel("Frequency", fontsize=FS)
            ax.set_title(f"{label} – {phase}", fontsize=FS+1, fontweight="bold")
            ax.legend(fontsize=FS-2, loc="best", framealpha=0.9)
            ax.grid(True, linewidth=0.4, alpha=0.5, axis="y")

    fname = PATHS["output"] / "fig09_improvement_histograms_HIGHPASS"
    fig.savefig(fname.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname.name}.png")


def run_highpass_analysis():
    """
    Ejecuta el análisis completo de alta frecuencia (high-pass).
    """
    print("\n" + "="*70)
    print("  HIGH-PASS ANALYSIS (High-Frequency Variability)")
    print("="*70)
    
    print("\n[HP-1] Loading high-pass metrics...")
    df_calval_hp = load_highpass_metrics("CalVal")
    df_science_hp = load_highpass_metrics("Science")
    
    if df_calval_hp is None and df_science_hp is None:
        print("  Warning: No high-pass metrics found. Skipping high-pass analysis.")
        print("  Run P1_comparision_TG_NRT_vs_DUACS_MY_MIOST_v7_HIGHPASS.py first.")
        return
    
    print("\n[HP-2] Generating high-pass percentage improvement maps...")
    if df_calval_hp is not None or df_science_hp is not None:
        plot_highpass_percentage_improvement_maps(df_calval_hp, df_science_hp)
    
    print("\n[HP-3] Generating high-pass RMSD maps...")
    if df_calval_hp is not None or df_science_hp is not None:
        plot_highpass_rmsd_maps(df_calval_hp, df_science_hp)
    
    print("\n[HP-4] Generating high-pass improvement histograms...")
    if df_calval_hp is not None or df_science_hp is not None:
        plot_highpass_improvement_histograms(df_calval_hp, df_science_hp)
    
    # Guardar resultados numéricos high-pass
    if "highpass_analysis" not in RESULTS:
        RESULTS["highpass_analysis"] = {}
    
    for phase, df in [("CalVal", df_calval_hp), ("Science", df_science_hp)]:
        if df is not None and not df.empty:
            df_hp = compute_highpass_improvement_metrics(df)
            RESULTS["highpass_analysis"][phase] = {
                "n_stations": len(df_hp),
                "PI_RMSD_HP_mean": float(df_hp["PI_RMSD_HP"].mean()),
                "PI_RMSD_HP_median": float(df_hp["PI_RMSD_HP"].median()),
                "PI_RMSD_HP_std": float(df_hp["PI_RMSD_HP"].std()),
                "PI_Corr_HP_mean": float(df_hp["PI_Corr_HP"].mean()),
                "PI_Corr_HP_median": float(df_hp["PI_Corr_HP"].median()),
                "PI_Corr_HP_std": float(df_hp["PI_Corr_HP"].std()),
                "pct_RMSD_improved": float(100 * (df_hp["PI_RMSD_HP"] > 0).sum() / len(df_hp)),
                "pct_Corr_improved": float(100 * (df_hp["PI_Corr_HP"] > 0).sum() / len(df_hp)),
            }
    
    print("\n  High-pass analysis completed successfully.")
    print("="*70)


# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# FIGURA DE COBERTURA TEMPORAL DE LOS PRODUCTOS COMPARADOS
# =============================================================================

def _derive_coverage_from_timeseries(products):
    """Deriva el periodo temporal REALMENTE comparado de cada producto a partir
    de las series temporales exportadas por P1 (TS_data_*.csv). Devuelve dict
    {product: {phase: (start, end, n_days)}}. Si no hay series, devuelve {}.
    """
    coverage = {p: {} for p in products}
    ts_root = PATHS["ts_data"]
    if not ts_root.exists():
        return {}

    col_for = {p: f"sla_{p}" for p in products}
    for phase in PHASES:
        pdir = ts_root / phase
        if not pdir.exists():
            continue
        files = list(pdir.glob("TS_data_*.csv"))
        if not files:
            continue
        # Acumular rango temporal por producto sobre todas las estaciones
        agg = {p: {"min": None, "max": None, "days": set()} for p in products}
        for fpath in files:
            try:
                df = pd.read_csv(fpath, usecols=lambda c: c in
                                 (["time_d"] + list(col_for.values())))
            except Exception:
                try:
                    df = pd.read_csv(fpath)
                except Exception:
                    continue
            if "time_d" not in df.columns:
                continue
            t = pd.to_datetime(df["time_d"], errors="coerce")
            for p in products:
                col = col_for[p]
                if col not in df.columns:
                    continue
                valid = t[df[col].notna().values]
                if valid.empty:
                    continue
                tmin, tmax = valid.min(), valid.max()
                a = agg[p]
                a["min"] = tmin if a["min"] is None else min(a["min"], tmin)
                a["max"] = tmax if a["max"] is None else max(a["max"], tmax)
                a["days"].update(valid.dt.normalize().tolist())
        for p in products:
            a = agg[p]
            if a["min"] is not None:
                coverage[p][phase] = (a["min"], a["max"], len(a["days"]))
    # Limpiar productos sin datos
    return {p: v for p, v in coverage.items() if v}


def _coverage_from_global_csv(products):
    """Fallback: lee la cobertura GLOBAL de cada producto del CSV de P1
    (product_temporal_coverage_v7.csv). Devuelve dict {product: (start, end)}.
    """
    path = PATHS["coverage_csv"]
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    out = {}
    for _, r in df.iterrows():
        prod = str(r.get("product", ""))
        if prod in products:
            out[prod] = (pd.to_datetime(r["start"]), pd.to_datetime(r["end"]))
    return out


def plot_temporal_coverage_comparison(products=None):
    """Figura tipo Gantt con el periodo temporal de cada producto comparado.
    Resalta el SOLAPAMIENTO efectivo (la intersección de fechas usada de verdad
    en las métricas) y marca las fases CalVal / Science.

    Prioriza derivar la cobertura de las series temporales reales (P1 TS_data);
    si no están disponibles, usa el CSV de cobertura global de P1.
    """
    if products is None:
        # Productos involucrados en la comparación activa + baseline común
        products = []
        for p in [BASELINE, TEST, COMMON_BASELINE]:
            if p not in products:
                products.append(p)

    cov_ts = _derive_coverage_from_timeseries(products)
    use_phase_detail = bool(cov_ts)
    cov_glob = {} if use_phase_detail else _coverage_from_global_csv(products)

    if not cov_ts and not cov_glob:
        print("    [coverage] No hay datos de cobertura temporal disponibles "
              "(ni TS_data ni product_temporal_coverage_v7.csv). Se omite la figura.")
        return

    # Colores por producto (consistentes con el resto del análisis)
    prod_color = {
        "my": "#E69F00",
        "my_miost": "#0072B2",
        "miost": "#009E73",
    }

    phase_span = {
        "CalVal":  (pd.Timestamp("2023-03-28"), pd.Timestamp("2023-07-05")),
        "Science": (pd.Timestamp("2023-07-06"), None),
    }
    phase_color = {"CalVal": "#fde0dd", "Science": "#e0f3f8"}

    fig, ax = plt.subplots(figsize=(13, 2.2 + 0.9 * len(products)))

    y_labels = []
    rows = []  # (y, start, end, color, label, n_days)

    if use_phase_detail:
        # Una barra por (producto, fase), apiladas con etiqueta de nº de días
        y = 0
        for p in products:
            if p not in cov_ts:
                continue
            for phase in PHASES:
                if phase not in cov_ts[p]:
                    continue
                start, end, ndays = cov_ts[p][phase]
                rows.append((y, start, end, prod_color.get(p, "#777777"),
                             f"{product_label(p)} · {phase}", ndays))
                y_labels.append(f"{product_label(p)} · {phase}")
                y += 1
    else:
        # Una barra por producto (cobertura global)
        for y, p in enumerate(products):
            if p not in cov_glob:
                continue
            start, end = cov_glob[p]
            rows.append((y, start, end, prod_color.get(p, "#777777"),
                         product_label(p), None))
            y_labels.append(product_label(p))

    # Determinar fin global para fases abiertas
    all_ends = [r[2] for r in rows if r[2] is not None]
    global_end = max(all_ends) if all_ends else pd.Timestamp("2024-12-31")

    # Sombrear fases
    for phase, (ps, pe) in phase_span.items():
        pe_eff = global_end if pe is None else pe
        ax.axvspan(plt.matplotlib.dates.date2num(ps),
                   plt.matplotlib.dates.date2num(pe_eff),
                   color=phase_color.get(phase, "#eeeeee"), alpha=0.5, zorder=0)
        ax.text(plt.matplotlib.dates.date2num(ps + (pe_eff - ps) / 2),
                len(rows) - 0.3, phase, ha="center", va="bottom",
                fontsize=FS - 1, style="italic", color="#444444")

    # Calcular y resaltar el solapamiento efectivo entre baseline y test
    # (intersección de los rangos del par activo)
    pair_ranges = []
    for r in rows:
        pass
    # Intersección a partir de los rangos globales de baseline/test
    def _global_range(prod):
        rs = [r for r in rows if r[4].startswith(product_label(prod))]
        if not rs:
            return None
        return (min(r[1] for r in rs), max(r[2] for r in rs))
    rng_b = _global_range(BASELINE)
    rng_t = _global_range(TEST)
    if rng_b and rng_t:
        ov_start = max(rng_b[0], rng_t[0])
        ov_end = min(rng_b[1], rng_t[1])
        if ov_start < ov_end:
            ax.axvspan(plt.matplotlib.dates.date2num(ov_start),
                       plt.matplotlib.dates.date2num(ov_end),
                       facecolor="#d62728", edgecolor="#d62728",
                       linewidth=0, zorder=0, alpha=0.07)
            # Líneas verticales en los bordes del solapamiento
            for xb in (ov_start, ov_end):
                ax.axvline(plt.matplotlib.dates.date2num(xb), color="#d62728",
                           linewidth=1.0, linestyle="--", alpha=0.7, zorder=1)
            ax.text(plt.matplotlib.dates.date2num(ov_start +
                    (ov_end - ov_start) / 2), -0.85,
                    f"Compared overlap ({BASELINE_LABEL} \u2229 {TEST_LABEL})",
                    ha="center", va="top", fontsize=FS - 2,
                    color="#d62728", fontweight="bold")

    # Dibujar barras
    for (y, start, end, color, label, ndays) in rows:
        start_num = plt.matplotlib.dates.date2num(start)
        end_num = plt.matplotlib.dates.date2num(end)
        width = max(end_num - start_num, 0.5)
        yy = len(rows) - 1 - y
        ax.barh(yy, width, left=start_num, height=0.55, color=color,
                alpha=0.9, edgecolor="black", linewidth=0.6, zorder=3)
        dur = (end - start).days
        info = f"  {dur} d" + (f" · {ndays} obs" if ndays is not None else "")
        ax.text(end_num, yy, info, va="center", ha="left",
                fontsize=FS - 2, color="#333333")

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(y_labels[::-1], fontsize=FS - 1)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=FS - 1)
    ax.set_xlabel("Time", fontsize=FS)
    ax.set_title(f"Temporal coverage of compared products\n{COMPARISON_TITLE}",
                 fontsize=FS + 1, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3, linewidth=0.5)
    ax.set_ylim(-1.2, len(rows) + 0.3)

    # Leyenda de productos
    handles = [Patch(facecolor=prod_color.get(p, "#777777"),
                     edgecolor="black", label=product_label(p))
               for p in products]
    handles.append(Patch(facecolor="#d62728", edgecolor="#d62728", alpha=0.2,
                         label="Compared overlap"))
    ax.legend(handles=handles, loc="lower right", fontsize=FS - 2,
              framealpha=0.9, ncol=2)

    plt.tight_layout()
    fname = PATHS["output"] / "fig00_temporal_coverage"
    fig.savefig(fname.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname.name}.png")


# =============================================================================
# ORQUESTADOR DE DOS COMPARACIONES
# =============================================================================

def run_comparison(comp_key: str) -> dict:
    """Ejecuta el pipeline P3 completo para una comparación (par baseline/test).
    Reconfigura las variables globales, fija la carpeta de salida y llama a todas
    las funciones de análisis y figuras. Devuelve un resumen numérico.
    """
    global BASELINE, TEST, BASELINE_LABEL, TEST_LABEL, COMPARISON_TITLE, COMPARISON_KEY

    comp = COMPARISONS[comp_key]
    BASELINE = comp["baseline"]
    TEST = comp["test"]
    BASELINE_LABEL = comp["baseline_label"]
    TEST_LABEL = comp["test_label"]
    COMPARISON_TITLE = comp["title"]
    COMPARISON_KEY = comp_key

    # Reconfigurar rutas de salida para esta comparación
    PATHS["output"] = PATHS["output_root"] / comp["outdir"]
    PATHS["figs_maps"] = PATHS["output"] / "maps"
    PATHS["output"].mkdir(parents=True, exist_ok=True)
    PATHS["figs_maps"].mkdir(parents=True, exist_ok=True)

    _reset_results()
    RESULTS["comparison"] = {
        "key": comp_key,
        "baseline": BASELINE, "test": TEST,
        "baseline_label": BASELINE_LABEL, "test_label": TEST_LABEL,
        "common_baseline": COMMON_BASELINE,
        "title": COMPARISON_TITLE,
    }

    print("\n" + "=" * 78)
    print(f"  COMPARACIÓN: {COMPARISON_TITLE}")
    print(f"  baseline = {BASELINE!r} ({BASELINE_LABEL})   |   "
          f"test = {TEST!r} ({TEST_LABEL})")
    print(f"  salida -> {PATHS['output']}")
    print("=" * 78)

    print("\n[0] Loading metrics...")
    df_calval = load_metrics("CalVal", loess=True)
    df_science = load_metrics("Science", loess=True)
    df_combined = load_combined_metrics(loess=True)
    print(f"    CalVal: {len(df_calval)} stations | Science: {len(df_science)} stations")

    print("[0b] Temporal coverage of compared products...")
    try: plot_temporal_coverage_comparison()
    except Exception as e: print(f"    [skip] {e}")

    print("[1] Study area map...")
    try: plot_study_area_map(df_combined)
    except Exception as e: print(f"    [skip] {e}")

    print("[2] Taylor diagrams...")
    try: plot_taylor_diagram(df_calval, df_science)
    except Exception as e: print(f"    [skip] {e}")

    print("[3] Percentage Improvement maps...")
    try: plot_percentage_improvement_maps(df_calval, df_science)
    except Exception as e: print(f"    [skip] {e}")

    print("[4] RMSD absolute maps...")
    try: plot_rmsd_absolute_maps(df_calval, df_science)
    except Exception as e: print(f"    [skip] {e}")

    print("[4b] RMSD scatter by region...")
    try: plot_rmsd_scatter_by_region(df_calval, df_science)
    except Exception as e: print(f"    [skip] {e}")

    print("[5] Metrics vs variability...")
    try: plot_metrics_vs_variability(df_calval, df_science)
    except Exception as e: print(f"    [skip] {e}")

    print("[6] Improvement histograms...")
    try:
        plot_improvement_histograms_simple(df_calval, df_science)
        plot_improvement_histograms_by_region(df_calval, df_science)
    except Exception as e: print(f"    [skip] {e}")

    print("[7] Regional improvement bars...")
    try: plot_regional_improvement_bars(df_calval, df_science)
    except Exception as e: print(f"    [skip] {e}")

    print("[8] Statistical significance (global)...")
    results_calval = paired_significance_tests(df_calval, "CalVal")
    results_science = paired_significance_tests(df_science, "Science")
    try: plot_significance_summary(results_calval, results_science)
    except Exception as e: print(f"    [skip] {e}")
    try: save_summary_table(results_calval, results_science)
    except Exception as e: print(f"    [skip] {e}")

    print("[9] Statistical significance (by region)...")
    regional_calval = regional_significance_tests(df_calval, "CalVal")
    regional_science = regional_significance_tests(df_science, "Science")
    try: plot_regional_significance(regional_calval, regional_science)
    except Exception as e: print(f"    [skip] {e}")
    try: save_regional_significance_table(regional_calval, regional_science)
    except Exception as e: print(f"    [skip] {e}")
    try: plot_regional_heatmap(regional_calval, regional_science)
    except Exception as e: print(f"    [skip] {e}")

    print("[10] CalVal SWOT tracks analysis...")
    swot_l3_dir = Path(PATHS["swot_l3"])  # coerción defensiva (ver nota en PATHS block)
    if swot_l3_dir.exists():
        try:
            swot_tracks = load_swot_tracks(swot_l3_dir)
            if swot_tracks:
                print_article_metrics(df_calval, swot_tracks)
                plot_calval_swot_tracks(df_calval, swot_tracks)
                plot_calval_swot_impact(df_calval, swot_tracks)
                plot_calval_swot_impact_by_region(df_calval, swot_tracks)
                plot_calval_swot_improvement_histograms(df_calval, swot_tracks)
            else:
                print("    Warning: No SWOT tracks loaded")
        except Exception as e:
            print(f"    [skip] {e}")
    else:
        print(f"    {'!'*70}")
        print(f"    [ALERTA] SWOT L3 dir NO ENCONTRADO — análisis Near/Far SWOT OMITIDO")
        print(f"    Ruta buscada: {swot_l3_dir}")
        print(f"    Si tus NetCDF SWOT L3 Expert están en otra ruta, relanza con:")
        print(f"      --data-base-dir /ruta/correcta   (debe ser el --base-dir de P1)")
        print(f"    {'!'*70}")

    print("[11] Combined analysis...")
    try: plot_combined_analysis(df_combined)
    except Exception as e: print(f"    [skip] {e}")

    print("[12] Spatial variability analysis...")
    try:
        plot_variability_analysis(df_calval, df_science)
        plot_variability_difference_histograms(df_calval, df_science)
    except Exception as e: print(f"    [skip] {e}")

    print("[12b] Combined difference maps...")
    try: plot_difference_maps_combined(df_calval, df_science)
    except Exception as e: print(f"    [skip] {e}")

    print("[13] Summary of PI results:")
    for phase, df in [("CalVal", df_calval), ("Science", df_science)]:
        try:
            df_ss = compute_improvement_metrics(df)
            pi_rmsd = df_ss["PI_RMSD"].mean()
            pi_corr = df_ss["PI_Corr"].mean()
            pct = 100 * (df_ss["PI_RMSD"] > 0).sum() / len(df_ss)
            line = (f"    {phase:8s}: PI_RMSD={pi_rmsd:+.2f}% "
                    f"({pct:.0f}% improved), PI_Corr={pi_corr:+.2f}%")
            if "PI_RMSD_vsMY" in df_ss.columns:
                line += f" | vsMY: PI_RMSD={df_ss['PI_RMSD_vsMY'].mean():+.2f}%"
            print(line)
        except Exception as e:
            print(f"    [skip {phase}] {e}")

    print("[14] High-pass analysis...")
    try: run_highpass_analysis()
    except Exception as e: print(f"    [skip] {e}")

    print("[15] Saving numerical results...")
    try: save_all_numerical_results()
    except Exception as e: print(f"    [skip] {e}")

    print(f"\n  ✔ Comparación '{comp_key}' completada -> {PATHS['output']}")
    return dict(RESULTS.get("comparison", {}))


def _run_comparison_worker(comp_key: str) -> dict:
    """Wrapper de nivel-módulo para ProcessPoolExecutor (debe ser 'picklable').
    Cada proceso tiene su propia copia de los globales, así que reconfigurar
    BASELINE/TEST/PATHS dentro de run_comparison() no interfiere entre procesos.
    """
    return run_comparison(comp_key)


def main(parallel: bool = True) -> None:
    print("=" * 78)
    print("  P2 - Advanced Value-Added Analysis (fixed / per_station / per_region)")
    print(f"  radius-mode: {RADIUS_MODE}")
    print("  1) Interpolation effect: MY vs MY-MIOST")
    print("  2) KaRIn added value:    MY-MIOST vs MIOST")
    print("  3) Total effect:         MY vs MIOST_KaRIn")
    print("  + SWOT proximity analysis (Near/Far, global + by-region)")
    print(f"  Modo: {'PARALELO (' + str(len(COMPARISONS)) + ' procesos)' if parallel else 'SECUENCIAL'}")
    print("=" * 78)
    print(f"\nOutput root: {PATHS['output_root']}\n")

    comp_keys = list(COMPARISONS.keys())
    summaries = {}

    if parallel and len(comp_keys) > 1:
        # Las dos comparaciones son independientes -> un proceso por comparación.
        # matplotlib('Agg') es seguro en subprocesos separados; cada proceso
        # escribe en su propia carpeta de salida, sin colisiones.
        import multiprocessing as mp
        ctx = mp.get_context("spawn")  # 'spawn' evita problemas de fork con matplotlib/GEOS
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=len(comp_keys), mp_context=ctx) as executor:
            future_to_key = {
                executor.submit(_run_comparison_worker, k): k for k in comp_keys
            }
            for future in concurrent.futures.as_completed(future_to_key):
                k = future_to_key[future]
                try:
                    summaries[k] = future.result()
                except Exception as e:
                    import traceback
                    print(f"  [ERROR] Comparación '{k}' falló: {e}")
                    print(f"  {'-'*70}")
                    print(f"  Traceback completo (para localizar la línea exacta):")
                    traceback.print_exception(type(e), e, e.__traceback__)
                    print(f"  {'-'*70}")
                    summaries[k] = {"error": str(e),
                                    "traceback": traceback.format_exception(
                                        type(e), e, e.__traceback__)}
    else:
        for comp_key in comp_keys:
            summaries[comp_key] = run_comparison(comp_key)

    # Resumen global de ambas comparaciones
    summary_path = PATHS["output_root"] / "comparisons_overview.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2, default=str)

    print("\n" + "=" * 78)
    print("  AMBAS COMPARACIONES COMPLETADAS")
    for k, v in COMPARISONS.items():
        print(f"    - {k:22s} ({v['baseline']} vs {v['test']}) "
              f"-> {PATHS['output_root'] / v['outdir']}")
    print(f"  Resumen: {summary_path}")
    print("=" * 78)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="P2 - análisis de valor añadido (fixed / per_station / per_region)")
    parser.add_argument("--sequential", action="store_true",
                        help="Ejecutar las comparaciones en secuencia (no en paralelo)")
    parser.add_argument("--radius-mode", default=None,
                        choices=["fixed", "per_station", "per_region"],
                        help="Metodología de radio a usar (def.: variable de "
                             "entorno P2_RADIUS_MODE, o 'fixed' si no está "
                             "definida). IMPORTANTE: si lo cambias aquí, "
                             "P2 se relanza a sí mismo como subproceso para "
                             "que PATHS se reconstruya con la nueva "
                             "configuración desde cero (PATHS ya se calculó "
                             "al importar el módulo, antes de leer argparse).")
    parser.add_argument("--fixed-radius-km", type=int, default=None,
                        help="Radio (km) para --radius-mode fixed (def.: "
                             "P2_FIXED_RADIUS_KM o 50)")
    parser.add_argument("--p1-outdir", type=Path, default=None,
                        help="Carpeta de salida de P1 (def.: "
                             "P2_P1_OUTDIR o ./results_radius_sensitivity_"
                             "commongrid_cipollini_r400)")
    parser.add_argument("--iho-shp", type=Path, default=None,
                        help="Ruta al shapefile IHO v2 (def.: P2_IHO_SHP o "
                             "el default hardcodeado)")
    parser.add_argument("--data-base-dir", type=Path, default=None,
                        help="Carpeta base de los datos crudos (data/raw/swot_l3_expert, "
                             "results/metrics/product_temporal_coverage_v7.csv) — "
                             "debe ser el mismo --base-dir que usaste en P1. "
                             "Def.: P2_DATA_BASE_DIR o "
                             "KARIN_PROJECT_DIR")
    
    # --- NUEVO: Argumento para especificar la carpeta de salida de P2 ---
    parser.add_argument("--outdir", type=Path, default=None,
                        help="Carpeta principal de salida para los resultados de P2 "
                             "(gráficos, tablas y métricas). Def.: ./results_latest")
    args = parser.parse_args()

    # Si el usuario pidió una configuración distinta a la que ya se usó para
    # construir PATHS (leída de os.environ al importar el módulo), hay que
    # RELANZAR el script como subproceso con el entorno actualizado: PATHS es
    # un dict a nivel de módulo ya construido, y algunas figuras/funciones lo
    # capturan en closures — mutarlo a medio camino es frágil. Relanzar es
    # más lento (un proceso extra) pero 100% correcto y simple de razonar.
    needs_relaunch = False
    new_env = dict(os.environ)
    if args.radius_mode is not None and args.radius_mode != RADIUS_MODE:
        new_env["P2_RADIUS_MODE"] = args.radius_mode
        needs_relaunch = True
    if args.fixed_radius_km is not None and args.fixed_radius_km != FIXED_RADIUS_KM:
        new_env["P2_FIXED_RADIUS_KM"] = str(args.fixed_radius_km)
        needs_relaunch = True
    if args.p1_outdir is not None and Path(args.p1_outdir) != P1_OUTDIR:
        new_env["P2_P1_OUTDIR"] = str(args.p1_outdir)
        needs_relaunch = True
    if args.iho_shp is not None and str(args.iho_shp) != str(PATHS["iho_shp"]):
        new_env["P2_IHO_SHP"] = str(args.iho_shp)
        needs_relaunch = True
    if args.data_base_dir is not None and Path(args.data_base_dir) != DATA_BASE_DIR:
        new_env["P2_DATA_BASE_DIR"] = str(args.data_base_dir)
        needs_relaunch = True

    if args.outdir is not None and Path(args.outdir).resolve() != P2_OUTDIR.resolve():
        new_env["P2_OUTDIR"] = str(Path(args.outdir).resolve())
        needs_relaunch = True

    if needs_relaunch:
        import subprocess
        import sys as _sys
        print(f"[P2] Relanzando con radius-mode={new_env.get('P2_RADIUS_MODE', RADIUS_MODE)} "
              f"(PATHS se reconstruye en el subproceso)...")
        cmd = [_sys.executable, __file__]
        if args.sequential:
            cmd.append("--sequential")
        raise SystemExit(subprocess.run(cmd, env=new_env).returncode)

    print(f"[P2] radius-mode = {RADIUS_MODE}  |  metrics = {PATHS['metrics']}  |  "
          f"output_root = {PATHS['output_root']}")
    main(parallel=not args.sequential)