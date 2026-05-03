"""
14_significance_tests.py
"""
import sys
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
"""
SHAP-physical 상관관계의 통계적 유의성 검정 + 모델 비교 통계 (R1 #3-2, R2 #3 대응)
  - Pearson r + 95% CI + p-value (Bootstrap)
  - Spearman rank correlation (alternative)
  - Paired t-test / Wilcoxon test for CV-fold model comparison
출력:
  - results/significance_tests.csv
  - results/figures/fig_correlation_with_CI.png
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats

matplotlib.rcParams.update({'font.family':'DejaVu Sans','font.size':12,
                            'axes.spines.top':False,'axes.spines.right':False,
                            'figure.dpi':150,'savefig.dpi':300,'savefig.bbox':'tight'})

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHAP_RF    = os.path.join(BASE_DIR, 'results/shap_feature_importance.csv')
SHAP_XGB   = os.path.join(BASE_DIR, 'results/shap_xgboost_importance.csv')
KFOLD_CSV  = os.path.join(BASE_DIR, 'results/kfold_cv_results.csv')
OUT_CSV    = os.path.join(BASE_DIR, 'results/significance_tests.csv')
FIG_DIR    = os.path.join(BASE_DIR, 'results/figures')

BLUE='#2166AC'; ORANGE='#D6604D'; GREEN='#4DAC26'

# ── (A) SHAP-Physical correlation: significance + bootstrap CI ────────────────
print("="*70)
print("(A) SHAP–Physical 상관관계 유의성 검정 (Bootstrap 95% CI)")
print("="*70)

records = []
for shap_csv, model_name in [(SHAP_RF, 'RF'), (SHAP_XGB, 'XGBoost')]:
    if not os.path.exists(shap_csv):
        print(f"  [skip] {shap_csv} 없음")
        continue
    df_imp = pd.read_csv(shap_csv)
    print(f"\n  Model: {model_name}  (N={len(df_imp)})")
    for var, label in [('area_ha',        'Total Area (ha)'),
                        ('impervious_pct', 'Impervious Fraction (%)'),
                        ('avg_slope_deg',  'Mean Slope (°)')]:
        x = df_imp[var].values
        y = df_imp['mean_abs_SHAP'].values

        # Pearson
        pr, pp = stats.pearsonr(x, y)
        # Spearman (rank-based, robust to non-linearity)
        sr, sp = stats.spearmanr(x, y)

        # Bootstrap 95% CI for Pearson (n_boot=10000)
        rng = np.random.default_rng(42)
        n_boot = 10000
        boot_r = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.choice(len(x), len(x), replace=True)
            if len(np.unique(x[idx])) > 1 and len(np.unique(y[idx])) > 1:
                boot_r[b] = np.corrcoef(x[idx], y[idx])[0, 1]
            else:
                boot_r[b] = np.nan
        boot_r = boot_r[~np.isnan(boot_r)]
        ci_lo, ci_hi = np.percentile(boot_r, [2.5, 97.5])

        records.append({
            'Model'          : model_name,
            'Variable'       : label,
            'Pearson_r'      : round(pr, 3),
            'Pearson_p'      : round(pp, 4),
            'Pearson_CI95_lo': round(ci_lo, 3),
            'Pearson_CI95_hi': round(ci_hi, 3),
            'Spearman_rho'   : round(sr, 3),
            'Spearman_p'     : round(sp, 4),
            'N'              : len(x),
        })
        sig = '***' if pp < 0.001 else '**' if pp < 0.01 else '*' if pp < 0.05 else 'n.s.'
        print(f"    {label:<28} Pearson r={pr:+.3f} (95%CI [{ci_lo:+.2f}, {ci_hi:+.2f}], "
              f"p={pp:.4f} {sig}); Spearman ρ={sr:+.3f} (p={sp:.4f})")

# ── (B) CV-fold model comparison (paired tests) ───────────────────────────────
print("\n" + "="*70)
print("(B) Surrogate 모델 간 5-Fold CV 성능 차이 유의성 검정")
print("="*70)

df_cv = pd.read_csv(KFOLD_CSV)
models_list = df_cv['model'].unique().tolist()
for m1, m2 in [('XGBoost','Random Forest'),
                ('XGBoost','Linear Regression'),
                ('Random Forest','Linear Regression')]:
    if m1 not in models_list or m2 not in models_list:
        continue
    r2_1 = df_cv[df_cv['model']==m1].sort_values('fold')['R2'].values
    r2_2 = df_cv[df_cv['model']==m2].sort_values('fold')['R2'].values
    diff = r2_1 - r2_2
    # Paired t-test
    t_stat, t_p = stats.ttest_rel(r2_1, r2_2)
    # Wilcoxon signed-rank (non-parametric)
    try:
        w_stat, w_p = stats.wilcoxon(r2_1, r2_2)
    except Exception:
        w_stat, w_p = np.nan, np.nan
    records.append({
        'Model'   : f'{m1} vs {m2}',
        'Variable': 'CV R² (5 folds)',
        'Pearson_r': '-',
        'Pearson_p': round(t_p, 4),
        'Pearson_CI95_lo': round(np.mean(diff) - 1.96 * np.std(diff)/np.sqrt(5), 3),
        'Pearson_CI95_hi': round(np.mean(diff) + 1.96 * np.std(diff)/np.sqrt(5), 3),
        'Spearman_rho': '-',
        'Spearman_p'  : round(w_p, 4),
        'N': 5,
    })
    sig = '***' if t_p < 0.001 else '**' if t_p < 0.01 else '*' if t_p < 0.05 else 'n.s.'
    print(f"  {m1:<18} vs {m2:<18}: ΔR² mean={np.mean(diff):+.4f}  "
          f"t-test p={t_p:.4f} {sig}  Wilcoxon p={w_p:.4f}")

# ── 저장 ──────────────────────────────────────────────────────────────────────
df_out = pd.DataFrame(records)
df_out.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
print(f"\n저장: {OUT_CSV}")

# ── 시각화: Forest plot of correlations with CI ────────────────────────────────
df_sig = df_out[df_out['Pearson_r'] != '-'].copy()
if len(df_sig) > 0:
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    ax.set_facecolor('white')
    df_sig['Label'] = df_sig['Model'] + ' | ' + df_sig['Variable']
    df_sig = df_sig.sort_values(['Model','Pearson_r'], ascending=[True, False])
    y_pos = np.arange(len(df_sig))
    rs   = df_sig['Pearson_r'].astype(float).values
    los  = df_sig['Pearson_CI95_lo'].astype(float).values
    his  = df_sig['Pearson_CI95_hi'].astype(float).values
    ps   = df_sig['Pearson_p'].astype(float).values
    colors_pt = [BLUE if 'RF' in m else ORANGE for m in df_sig['Model'].values]
    for i, (r, lo, hi, c) in enumerate(zip(rs, los, his, colors_pt)):
        ax.plot([lo, hi], [i, i], '-', color=c, linewidth=2.5, alpha=0.7)
        ax.plot(r, i, 'o', color=c, markersize=10,
                markeredgecolor='white', markeredgewidth=1.5)
    for i, (r, p) in enumerate(zip(rs, ps)):
        sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.'
        ax.text(max(his) + 0.05, i, f'{sig} (p={p:.3f})', va='center', fontsize=10)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sig['Label'].values, fontsize=10)
    ax.set_xlabel('Pearson Correlation Coefficient (Bootstrap 95% CI)')
    ax.set_title('Statistical Significance of SHAP–Physical Correlations\n'
                 '(* p<0.05, ** p<0.01, *** p<0.001)',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='x', linestyle=':', alpha=0.4)
    ax.set_xlim(min(los.min(), -0.3) - 0.05, his.max() + 0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_correlation_with_CI.png'), facecolor='white')
    plt.close(fig)
    print(f"  fig_correlation_with_CI.png 저장")
