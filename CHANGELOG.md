# Changelog

## 2026-07 — Second revision (CLRM-D-26-00388 R2) — v1.1.0

Added in response to second-round reviewer comments (Reviewer #1-1, #2-1: rainfall scenarios / climate framing):

- `scripts/15_climate_stress_scenario.py` — re-runs the 0–100 % coverage sweep under a +30 % climate-change scaling of the 10-year design storm (rainfall time-series scaled ×1.30; the SWMM rain-gauge SCF is the snow-catch factor and does **not** scale rainfall).
- `scripts/17_multi_return_period.py` — coverage sweep for 30/50/100-year design storms (frequency growth factors ×1.25 / 1.36 / 1.54 relative to the 10-year event; approximate, to be reconciled with official KICT 2019 probability-rainfall values).
- `scripts/16_climate_stress_figure.py` — multi-return-period + climate-stress figure (manuscript Figure 15).
- `data/simulation_results_climate130.csv`, `data/simulation_results_return_periods.csv` — outputs of the new scenarios.
- `figures/fig_climate_stress.png` — rendered figure.

Key result: baseline runoff rises with storm severity (+15.5 % / +19.9 % / +25.2 % at 30/50/100-year; +17.6 % under +30 % climate), and the relative runoff reduction at a given coverage declines monotonically (e.g., 30 % coverage: 18.1 % → 15.2 % / 14.3 % / 13.2 %; 14.8 % under +30 % climate). The +30 % climate storm falls consistently between the 30- and 50-year present-climate storms.

Also in this revision (manuscript-side, no new code): title/abstract/conclusions re-framed to "scenario-based, stormwater-risk-informed screening"; Rational-Method and OSM–land-use comparisons re-labelled as independent consistency/plausibility checks (not validation); new feasibility-constraint discussion (optimized layouts framed as theoretical planning candidates); table numbering reconciled.

## 2026-05 — Major revision (CLRM-D-26-00388 R1)

Added in response to reviewer comments:

- `scripts/11_residual_diagnostics.py` — surrogate residual analysis (residuals vs. predicted, Q–Q plot, Shapiro–Wilk normality, residuals vs. LID area).
- `scripts/12_shap_xgboost_consistent.py` — SHAP applied to the same XGBoost surrogate used in optimization (resolves R2 #12 model inconsistency); includes 3 normalizations (raw, per-imp-area, per-total-area) addressing R2 #13.
- `scripts/13_nsga2_repeated.py` — 5 independent NSGA-II runs with seeds {42, 0, 7, 123, 2025}; tracks generation-by-generation hypervolume and overlays Pareto fronts.
- `scripts/14_significance_tests.py` — bootstrap (n=10,000) 95 % CI for SHAP–physical correlations; Pearson + Spearman; paired-samples *t*-tests across CV folds for surrogate model comparison.
- `data/nsga2_repeated_runs.csv`, `data/nsga2_convergence.csv`, `data/residual_diagnostics.csv`, `data/shap_xgboost_importance.csv`, `data/significance_tests.csv` — outputs of the new scripts.

Updated:

- `README.md` reproducibility instructions and software environment list.
- `requirements.txt` pinned to versions used in revision.

Known limitations (now documented in manuscript Section 5.8):

- Star-topology drainage (no sewer-network detail).
- Administrative-dong sub-catchments (≤ 3 % volume effect vs. hydro-units; see Supplementary Section S5).
- OSM imperviousness underestimation by ~20 percentage points (sensitivity envelope reported).
- Single 10-year 60-min design storm (no continuous simulation).
- 201 LHS samples for an 18-D space (saturation reached; targeted resampling reduces low-area Pareto error from +9.3 % → +5.1 %).
- No probabilistic uncertainty propagation, no economic LCA, no water-quality modeling.

## 2026-03 — Initial submission (CLRM-D-26-00388)

- `scripts/01_*` through `10_*` — main pipeline.
- `model/seocho_baseline.inp` — SWMM input file.
- `data/*` — pre-computed CSV outputs.
