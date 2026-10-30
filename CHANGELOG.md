# Changelog

## 2026-11 — Fourth revision (CLRM-D-26-00388 R4) — v1.1.3

Documentation and figure release; **no simulation, model input, or numerical output changed** relative to v1.1.2.

- `README.md` — result-to-script map for every manuscript table and figure (= manuscript Appendix A, Table A1); regeneration commands for scripts 18–21; stale run-time and "bit-identical" wording corrected.
- `scripts/21_pareto_front_figure.py` — re-draws manuscript Figure 9 from `pareto_solutions.csv`, `simulation_results.csv`, and `pareto_validation_results.csv`, adding the 10 SWMM-verified Pareto solutions next to the surrogate-predicted front and the uniform-coverage curve.
- `scripts/04_visualize.py` — Figure 3: shaded band and guide lines now mark the 30–50 % range (previously guides at 20 % and 50 % and an unlabelled fill); label overlap removed.
- `figures/fig3_combined.png`, `figures/fig_pareto_front.png` — regenerated.
- `.zenodo.json` — "validation" → "verification" for the surrogate-vs-SWMM comparison.

## 2026-07 — Third revision (CLRM-D-26-00388 R3) — v1.1.2

Added in response to third-round reviewer comments (Reviewer #2-9, #2-10: SWMM run accounting and archive completeness):

- `scripts/18_targeted_resampling.py` — 50 additional SWMM scenarios targeted at the low-coverage / spatially-concentrated region (1-5 active dongs, seed 4218), XGBoost re-training on n = 251, and re-evaluation of the 10 directly simulated Pareto solutions on a **common hold-out** drawn from the original 201 scenarios.
- `scripts/19_clogged_optimization.py` — full workflow repeated at Mid pavement permeability (100 mm/hr): 201 SWMM re-runs, surrogate re-training, NSGA-II re-optimization, Pareto-front and spatial-allocation comparison.
- `data/surrogate_training_data_resampled.csv`, `data/targeted_resampling_results.csv`, `data/targeted_resampling_summary.csv`
- `data/surrogate_training_data_midperm.csv`, `data/pareto_solutions_midperm.csv`, `data/clogged_optimization_summary.csv`, `data/clogged_spatial_comparison.csv`
- `scripts/20_run_manifest.py` — enumerates all 557 SWMM model executions with block, generating script, filename and SHA-256 checksum (manuscript Table 15).
- `data/swmm_run_manifest.csv`, `data/swmm_run_summary.csv`
- `figures/fig_clogged_optimization.png` — manuscript Figure 16.

Note: the 557 generated `.inp` files are regenerable intermediates and remain excluded by `.gitignore`; `swmm_run_manifest.csv` records them so the run accounting stays independently checkable.

**Corrected results.** Both analyses were reported in R2 from preliminary runs whose code was not committed. Re-running them from the now-archived scripts changed two reported numbers:

- Targeted resampling: low-area Pareto error is **9.6 % -> 2.9 %** (mean absolute; max 12.1 % -> 5.6 %), not the previously stated 9.3 % -> 5.1 %.
- Clogging re-optimization: the R2 claim that the S2 spatial pattern is *preserved* under clogging is **not supported**. Individual per-dong rankings are unstable (Spearman rho = +0.25, p = 0.31; 3/5 top-5 overlap); only the cluster-scale over-weighting of the southern Bangbae sub-districts persists (2.21x -> 1.87x its impervious share). The entire clogged Pareto front lies below the design-permeability front (8.6-14.8 % vs. 14.7-22.7 % reduction).

**Run accounting.** The complete study comprises **557 SWMM model executions** (manuscript Table 15): 222 core (201 LHS incl. zero-coverage baseline + 11 uniform + 10 Pareto verification) and 335 sensitivity/re-optimization runs. The figure of 221 quoted in R2 covered only part of the core workflow.

**Reproducibility wording.** The claim is **computational reproducibility under the documented software environment**; bit-identical cross-platform output is neither claimed nor demonstrated.

**Measured computational cost.** A single 3-h SWMM simulation of this 18-sub-catchment model takes ~0.02 s of engine time (~0.65 s including per-scenario .inp generation); all 557 runs take ~6 min. A single NSGA-II run (60,000 surrogate evaluations) takes ~6 s. The R2 figures of "~30 s per SWMM simulation" and "~60 min of SWMM" were incorrect and are corrected in the manuscript.

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
