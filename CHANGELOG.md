# Changelog

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
