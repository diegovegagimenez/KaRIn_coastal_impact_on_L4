# KaRIn_coastal_impact_on_L4

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23373092.svg)](https://doi.org/10.5281/zenodo.23373092)

Code for the validation of delayed-time Level-4 (L4) sea level products with and
without SWOT KaRIn observations against tide gauges along European coasts
(submitted to *Ocean Science*).

The repository contains the processing pipeline and the analysis and figure
scripts. It does not contain data or results.

## Products

| Label | Product |
|---|---|
| MY | Nadir-only multi-year DUACS L4 (Copernicus Marine Service) |
| MIOST-K | Experimental MIOST L4 merging nadir altimetry and SWOT KaRIn (AVISO+) |
| L3 | SWOT L3 KaRIn low-rate SSH, Expert v3.0 (AVISO+) |

## Data

| Dataset | DOI |
|---|---|
| European Seas gridded L4 SSH, reprocessed (MY) | https://doi.org/10.48670/moi-00141 |
| Global Ocean in situ observations (tide gauges) | https://doi.org/10.48670/moi-00036 |
| Experimental multimission gridded L4 with SWOT (MIOST) | https://doi.org/10.24400/527896/a01-2025.001 |
| SWOT L3 KaRIn low-rate SSH, Expert | https://doi.org/10.24400/527896/A01-2023.018 |
| Dynamic Atmospheric Correction | https://doi.org/10.24400/527896/A01-2022.001 |
| IHO Sea Areas v3 | https://doi.org/10.14284/323 |

## Setup

```bash
conda env create -f environment.yml
conda activate karin-l4
export KARIN_PROJECT_DIR=/path/to/project
export KARIN_IHO_SHP=/path/to/World_Seas_IHO_v2.shp   # optional
```

Layout of `$KARIN_PROJECT_DIR`:

```
data/raw/swot_l3_expert/              SWOT_L3_LR_SSH_Expert_*_v3.0.nc
data/raw/miost_l4/                    MIOST-K daily files
data/raw/duacs_l4_my/                 MY, 0.0625 deg
data/raw/duacs_l4_my_miost/           MY-MIOST, 0.0625 deg
data/raw/dac_l4/                      DAC
data/processed_NRT/tide_gauges_nrt/   one .pkl per station
PIPELINE_FINAL_LIMPIO/                pipeline outputs
```

## Run

1. Pipeline. P1 is run once per extraction radius R = 10, 20, ..., 100:

```bash
python pipeline/P1_altimetry_tidegauge_processing.py --fixed-radius-km R \
       --outdir $KARIN_PROJECT_DIR/PIPELINE_FINAL_LIMPIO/results_periodo_extendido \
       --science-end 2025-07-06 --operational-window
python pipeline/P2_swot_proximity.py
```

2. Analysis (from `scripts/`, in this order):

| Script | Output |
|---|---|
| `final_v1_scale.py` | `final_v1_scale.json` |
| `verify_v3_numbers.py` | `per_station_radius_with_pixels.csv` |
| `regions_v2.py` | `station_regions_v2.csv` |
| `final_v3_regions.py` | `per_station_radius_with_pixels_v2.csv`, `v3_*.csv` |
| `tidal_regime_proxy.py` | `tidal_regime_proxy.csv` |
| `tidal_class_stats.py` | tidal-class statistics |
| `compute_temporal_std_R50.py` | `temporal_std_products_R50.csv` |
| `station_level_stats.py` | station-level statistics |
| `karin_coverage_all_stations.py` | KaRIn coverage per station |
| `l3_near_swath_scale.py` | L3, L4 and tide gauge caches |
| `l3_near_swath_scale_qc.py` | L3 vs L4 (`quality_flag == 0`) |
| `l3_instantaneous.py` | instantaneous L3 vs tide gauge |

3. Figures (`scripts/make_fig*.py`, written to `figures/`):

| Script | Figure |
|---|---|
| `make_fig01_study_area_v2.py` | Fig. 1 |
| `make_fig07b_restyled.py` | Fig. 2 |
| `make_figF_proximity_boxplot_v2.py [R]` | Fig. 4 (R = 50), Fig. 7 (R = 10) |
| `make_figD_timeseries_example_v2.py` | Fig. 5 |
| `make_figE_nearfar_bootstrap.py` | Fig. 6 |
| `make_figN_l3_over_l4_pass16_qc.py` | Fig. 8 |
| `make_figO_l3_over_l4_transition_zone.py [YYYYMMDD]` | Fig. 9 |
| `make_fig07c_corr.py`, `make_fig07d_std.py` | Figs. S1, S2 |

## Conventions

`dRMSD = RMSD_MY - RMSD_MIOST-K` and `PI_RMSD = 100 * dRMSD / RMSD_MY`; positive
values mean that MIOST-K is closer to the tide gauge. Near-swath stations are
within 2 km of the KaRIn swath footprint of the CalVal passes 1, 3, 14 and 16.

## Licence

MIT, see [LICENSE](LICENSE).
