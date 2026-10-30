[![DOI](https://zenodo.org/badge/1227787164.svg)](https://doi.org/10.5281/zenodo.19998726)

# Scenario-Based, Stormwater-Risk-Informed Screening of Spatially Optimized Permeable Pavement
## Reproducibility Package

This repository contains all code, input data, and processed results that reproduce the analyses in:

> Kang, J. et al. (2026). *Scenario-Based, Stormwater-Risk-Informed Screening of Spatially Optimized Permeable Pavement: An Interpretable AI–Hydrology Framework for Seocho-gu, Seoul.* Climate Risk Management. (Manuscript number CLRM-D-26-00388, under revision.)

This repository is permanently archived on Zenodo (concept DOI:
  [10.5281/zenodo.19998726](https://doi.org/10.5281/zenodo.19998726); v1.0.0 DOI:
  [10.5281/zenodo.19998727](https://doi.org/10.5281/zenodo.19998727)). **The release that
  reproduces the fourth-revision (R4) manuscript is v1.1.3** (version DOI:
  [10.5281/zenodo.ZENODO_V113](https://doi.org/10.5281/zenodo.ZENODO_V113)); it adds the
  result-to-script map below, the Figure 9 re-draw (script 21), and the corrected Figure 3 —
  no simulation or numerical output changed relative to v1.1.2. The third-revision (R3)
  snapshot is release **v1.1.2** (version DOI:
  [10.5281/zenodo.21522480](https://doi.org/10.5281/zenodo.21522480)),
  which adds the targeted-resampling experiment (script 18),
  the reduced-permeability re-optimization (script 19), and the SWMM run manifest (script 20).
  The second-revision (R2) snapshot is release **v1.1.1**, which added the multi-return-period
  and climate-stress sensitivity analysis (scripts 15–17) using the official KICT (2019)
  design-rainfall depths.

**Reproducibility claim.** Re-executing these scripts with the pinned package versions and the
  documented random seeds regenerates every number, table, and figure in the manuscript —
  *computational reproducibility under the documented software environment*. Bit-identical
  numerical output across differing operating systems, processors, and numerical-library builds
  is neither claimed nor demonstrated.
---

## Repository Structure

```
.
├── scripts/                    # Python scripts (numbered in pipeline order)
│   ├── 01_gis_preprocessing.py        # OSM + DEM → sub-catchment parameters
│   ├── 02_generate_inp.py             # SWMM .inp file generation
│   ├── 03_run_simulation.py           # Uniform-coverage SWMM scenarios (0–100%)
│   ├── 04_visualize.py                # Runoff-reduction figures
│   ├── 05_surrogate_training.py       # LHS + 3 ML models
│   ├── 05b_kfold_validation.py        # 5-fold CV
│   ├── 06_spatial_optimization.py     # NSGA-II (single run)
│   ├── 06b_pareto_validation.py       # Direct SWMM validation of Pareto solutions
│   ├── 07_shap_analysis.py            # SHAP on Random Forest (legacy)
│   ├── 08_landuse_validation.py       # OSM vs. SG05 imperviousness
│   ├── 09_rational_method_validation.py
│   ├── 10_permeability_sensitivity.py # Clogging analysis
│   ├── 11_residual_diagnostics.py     # NEW (revision): residual plots, normality tests
│   ├── 12_shap_xgboost_consistent.py  # NEW (revision): SHAP on XGBoost (consistent)
│   ├── 13_nsga2_repeated.py           # NEW (R1): 5 NSGA-II runs + hypervolume
│   ├── 14_significance_tests.py       # NEW (R1): bootstrap CI, paired tests
│   ├── 15_climate_stress_scenario.py  # NEW (R2): +30% climate-scaled design storm
│   ├── 16_climate_stress_figure.py    # NEW (R2): multi-return-period + climate figure
│   ├── 17_multi_return_period.py      # NEW (R2): 30/50/100-year design storms
│   ├── 18_targeted_resampling.py      # NEW (R3): 50 low-coverage samples, n=251 retraining
│   ├── 19_clogged_optimization.py     # NEW (R3): full re-optimization at 100 mm/hr
│   ├── 20_run_manifest.py             # NEW (R3): manifest of all 557 SWMM runs (Table 15)
│   └── 21_pareto_front_figure.py      # NEW (R4): Figure 9 incl. SWMM-verified Pareto points
├── data/                       # Pre-computed CSV outputs (scripts write them to results/)
├── figures/                    # Figures 3, 9, 15, 16 (others regenerate from the scripts)
├── model/                      # SWMM input file
│   └── seocho_baseline.inp
├── requirements.txt            # Version-locked Python packages
├── LICENSE                     # MIT
└── README.md                   # This file
```

---

## Software Environment

- **Python**: 3.11–3.14 (tested on 3.14)
- **OS**: Windows 11 (developed) / Linux (CI-tested)
- **Approximate run time** on Intel i7 / 16 GB RAM: SWMM execution is not the bottleneck for this
  model — a single 3-hour simulation of the 18-sub-catchment network takes ~0.02 s of engine time
  (~0.65 s including per-scenario `.inp` generation), so all 557 model executions take ~6 minutes,
  and a single NSGA-II run (60,000 surrogate evaluations) takes ~6 s. End-to-end wall time is
  dominated by the geospatial preprocessing stage (script 01), which depends on OpenStreetMap and
  Copernicus DEM download speed.

Install dependencies in a fresh virtual environment:

```bash
python -m venv .venv
.\.venv\Scripts\activate       # Windows
# source .venv/bin/activate    # Linux / macOS
pip install -r requirements.txt
```

Key packages:

| Package | Pinned version | Purpose |
|---|---|---|
| `pyswmm` | 2.1 | EPA SWMM Python interface |
| `scipy` | 1.13 | Latin Hypercube Sampling, statistics |
| `scikit-learn` | 1.4 | RF, LR, KFold, train/test split |
| `xgboost` | 2.1 | Gradient-boosted-tree surrogate |
| `pymoo` | 0.6.1 | NSGA-II + hypervolume indicator |
| `shap` | 0.46 | SHAP values via TreeExplainer |
| `geopandas` | 0.14 | Vector GIS |
| `osmnx` | 1.9 | OpenStreetMap data |
| `rasterio` | 1.3 | DEM raster I/O |
| `matplotlib` | 3.8 | Plotting |
| `python-docx` | 1.1 | docx export |

---

## Reproducing the Main Analyses

All random seeds are deterministic (see Manuscript Table 7). Scripts read and write `results/`
(CSV) and `results/figures/` (PNG) relative to the repository root; the pre-computed copies are
in `data/` and `figures/` — copy `data/*.csv` to `results/` to run any post-processing script
without re-simulating. Re-running with the same seeds and pinned packages reproduces the
reported numbers under the documented software environment; bit-identical output across
platforms is not claimed.

```bash
# Stage 1 — GIS pre-processing (requires internet for OSM/DEM if data/ is empty)
python scripts/01_gis_preprocessing.py

# Stage 2 — Build SWMM input file
python scripts/02_generate_inp.py

# Stage 3 — Uniform-coverage screening (11 SWMM runs, seconds)
python scripts/03_run_simulation.py
python scripts/04_visualize.py

# Stage 4 — Surrogate training (201 SWMM runs incl. zero-coverage baseline, ~2 min)
python scripts/05_surrogate_training.py
python scripts/05b_kfold_validation.py

# Stage 5 — Spatial optimization
python scripts/06_spatial_optimization.py
python scripts/06b_pareto_validation.py

# Stage 6 — Explainability + auxiliary validations
python scripts/07_shap_analysis.py
python scripts/08_landuse_validation.py
python scripts/09_rational_method_validation.py
python scripts/10_permeability_sensitivity.py

# Stage 7 — REVISION analyses (added in response to reviewer comments)
python scripts/11_residual_diagnostics.py        # CV residual diagnostics
python scripts/12_shap_xgboost_consistent.py     # SHAP on operational XGBoost
python scripts/13_nsga2_repeated.py              # 5 independent NSGA-II runs
python scripts/14_significance_tests.py          # bootstrap CI, paired t-tests

# Stage 8 — SECOND-REVISION analyses (rainfall scenarios / climate framing)
python scripts/15_climate_stress_scenario.py     # +30% climate-scaled 10-yr storm
python scripts/17_multi_return_period.py         # 30/50/100-yr design storms
python scripts/16_climate_stress_figure.py       # Figure 15

# Stage 9 — THIRD-REVISION analyses (reproducibility completion)
python scripts/18_targeted_resampling.py         # 50 targeted SWMM runs + n=251 surrogate (Table 16)
python scripts/19_clogged_optimization.py        # 201 SWMM runs at 100 mm/hr + NSGA-II (Table 17, Fig. 16)
python scripts/20_run_manifest.py                # manifest of all 557 SWMM runs (Table 15)

# Stage 10 — FOURTH-REVISION figure
python scripts/21_pareto_front_figure.py         # Figure 9 (from CSVs only; no re-simulation)
```

---

## Result-to-Script Map (manuscript Appendix A, Table A1)

| Manuscript item | Section | Script(s) | Output file(s) |
|---|---|---|---|
| Figure 1 | 2 | `00_study_area_map.py` | `fig_study_area.png` |
| Table 3 | 3.2 | `01_gis_preprocessing.py` | `subcatchment_params.csv` |
| Tables 4–6 (SWMM input) | 3.3–3.6 | `02_generate_inp.py` | `model/seocho_baseline.inp` |
| Table 11; Figure 3 | 4.2–4.4 | `03_run_simulation.py`; `04_visualize.py` | `simulation_results.csv`; `fig3_combined.png` |
| Table 8; Figures 4–5 | 4.5 | `05_surrogate_training.py`; `05b_kfold_validation.py` | `surrogate_training_data.csv`, `surrogate_model_comparison.csv`, `kfold_cv_results.csv`, `kfold_cv_summary.csv` |
| Figure 6 | 4.5.1 | `11_residual_diagnostics.py` | `residual_diagnostics.csv` |
| Tables 9–10; Figure 12 | 4.5, 4.8.3 | `14_significance_tests.py` | `significance_tests.csv` |
| §4.6 stability; Figures 7–8 | 4.6 | `13_nsga2_repeated.py` | `nsga2_repeated_runs.csv`, `nsga2_convergence.csv` |
| Pareto front; Figure 13 | 4.7 | `06_spatial_optimization.py` | `pareto_solutions.csv`; `fig_optimal_maps.png` |
| Table 12; Figure 10 | 4.7.1 | `06b_pareto_validation.py` | `pareto_validation_results.csv` |
| Figure 9 | 4.7 | `21_pareto_front_figure.py` | `fig_pareto_front.png` |
| Table 16 | 4.7.2 | `18_targeted_resampling.py` | `surrogate_training_data_resampled.csv`, `targeted_resampling_results.csv`, `targeted_resampling_summary.csv` |
| Table 17; Figure 16 | 4.7.3 | `19_clogged_optimization.py` | `surrogate_training_data_midperm.csv`, `pareto_solutions_midperm.csv`, `clogged_optimization_summary.csv`, `clogged_spatial_comparison.csv`; `fig_clogged_optimization.png` |
| Table 13; Figure 11 | 4.8 | `12_shap_xgboost_consistent.py` | `shap_xgboost_importance.csv` |
| Table 14; Figure 15 | 4.9 | `17_multi_return_period.py`; `15_climate_stress_scenario.py`; `16_climate_stress_figure.py` | `simulation_results_return_periods.csv`, `simulation_results_climate130.csv`; `fig_climate_stress.png` |
| §4.1 consistency checks | 4.1 | `08_landuse_validation.py`; `09_rational_method_validation.py` | `landuse_validation.csv`, `rational_method_validation.csv` |
| Figure 14 | 5.6 | `10_permeability_sensitivity.py` | `permeability_sensitivity.csv` |
| Table 15 | 3.13 | `20_run_manifest.py` | `swmm_run_manifest.csv`, `swmm_run_summary.csv` |

The run manifest (`data/swmm_run_manifest.csv`) lists all 557 SWMM executions with block,
generating script, input filename, and SHA-256 checksum; `data/swmm_run_summary.csv` gives the
per-block totals of manuscript Table 15.

---

## Key Random Seeds (also in Manuscript Table 7)

| Component | Seed |
|---|---|
| LHS sampling | 42 |
| train/test split | 42 |
| KFold | 42 |
| Random Forest | 42 |
| XGBoost | 42 |
| NSGA-II (primary) | 42 |
| NSGA-II (replications) | 0, 7, 123, 2025 |
| Bootstrap CI | 42 |

---

## Pre-computed Outputs

If you wish to skip the long SWMM simulations, the `data/` folder already contains the deterministic outputs of every stage. You can run only the post-processing scripts (e.g., `04_visualize.py`, `05b_kfold_validation.py`, `12_shap_xgboost_consistent.py`, `14_significance_tests.py`) directly.

---

## Notes on Re-Use and Transferability

The framework is **LID-type-agnostic** (re-parameterize the LID block in `02_generate_inp.py`) and **district-agnostic** (replace `data/boundary/` and re-run `01_gis_preprocessing.py`). Documented limitations and transferability conditions appear in Section 5.7 and 5.8 of the manuscript.

---

## Citation

If you use this code or data, please cite:

```bibtex
@article{Kang2026CRM,
  title   = {Scenario-Based, Stormwater-Risk-Informed Screening of Spatially Optimized Permeable Pavement: An Interpretable AI–Hydrology Framework for Seocho-gu, Seoul},
  author  = {Kang, Junsuk and others},
  journal = {Climate Risk Management},
  year    = {2026},
  note    = {Manuscript CLRM-D-26-00388, under revision}
}
```

---

## License

MIT License — see `LICENSE` file. The Korean SGIS administrative boundary data, Copernicus DEM, and OpenStreetMap data are subject to their respective licenses (ODbL for OSM, CC-BY-4.0-IGO equivalent for Copernicus, public-domain for SGIS).

---

## Contact

For questions about the code or data, please contact:
**Junsuk Kang** — `junkang@snu.ac.kr`
Department of Landscape Architecture & Rural Systems Engineering, Seoul National University.
