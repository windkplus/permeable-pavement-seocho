"""
06_spatial_optimization.py
NSGA-II 기반 다목적 공간 최적화
  - 목적함수 1: 총유출량 최소화 (대리모델 사용)
  - 목적함수 2: LID 설치면적 최소화
  - 출력: Pareto Front + 최적 해 지도
출력:
  - results/pareto_solutions.csv
  - results/figures/fig_pareto_front.png
  - results/figures/fig_optimal_maps.png
"""

import os, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
import matplotlib
matplotlib.rcParams.update({'font.family':'DejaVu Sans','font.size':13,
                            'axes.spines.top':False,'axes.spines.right':False})

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS_CSV   = os.path.join(BASE_DIR, 'results/subcatchment_params.csv')
BOUNDARY_SHP = os.path.join(BASE_DIR, 'data/boundary/bnd_dong_11220_2025_2Q/bnd_dong_11220_2025_2Q.shp')
FIG_DIR      = os.path.join(BASE_DIR, 'results/figures')
OUT_CSV      = os.path.join(BASE_DIR, 'results/pareto_solutions.csv')

df_params = pd.read_csv(PARAMS_CSV)
df_params['ADM_CD'] = df_params['ADM_CD'].astype(str)

# 기준값 (Baseline)
BASELINE_RUNOFF = 614298.0   # m³

# 동별 최대 가용 LID 면적 (불투수면 전체)
df_params['max_lid_m2'] = (df_params['area_ha'] * 10_000 *
                            df_params['impervious_pct'] / 100)
TOTAL_MAX_LID_M2 = df_params['max_lid_m2'].sum()

# ── 대리모델 로드 ──────────────────────────────────────────────────────────────
print("대리모델 로드 중...")
bundle     = joblib.load(os.path.join(BASE_DIR, 'model/surrogate_best.pkl'))
surrogate  = bundle['model']
scaler     = bundle['scaler']
model_name = bundle['name']
dong_codes = bundle['dong_codes']
uses_weighted_feature = bundle.get('uses_weighted_feature', False)
needs_scaling         = bundle.get('needs_scaling', False)
print(f"  모델: {model_name}")

imp_areas = (df_params['area_ha'].values * 10_000 *
             df_params['impervious_pct'].values / 100)

def predict_runoff(rate_array):
    """rate_array: [N, 18] 또는 [18,] → 유출량 예측 (m³)"""
    X = np.atleast_2d(rate_array)
    if uses_weighted_feature:
        wc = (X * imp_areas).sum(axis=1, keepdims=True) / imp_areas.sum()
        X  = np.hstack([X, wc])
    if needs_scaling:
        X = scaler.transform(X)
    return surrogate.predict(X)

# ── 최적화 문제 정의 ───────────────────────────────────────────────────────────
class LIDOptimizationProblem(Problem):
    """
    변수: x[i] = 동 i의 투수성 포장 적용률 (0~100%)
    목적:
      f1 = 총유출량 (m³)              → minimize
      f2 = LID 설치면적 합계 (m²)     → minimize
    """
    def __init__(self):
        n_var = len(dong_codes)
        super().__init__(n_var=n_var,
                         n_obj=2,
                         n_constr=0,
                         xl=np.zeros(n_var),
                         xu=np.full(n_var, 100.0))

    def _evaluate(self, X, out, *args, **kwargs):
        # f1: 유출량 (대리모델)
        f1 = predict_runoff(X)    # shape [pop_size,]

        # f2: LID 설치면적 합계
        max_areas = df_params['max_lid_m2'].values   # [18,]
        f2 = (X / 100.0 * max_areas).sum(axis=1)    # [pop_size,]

        out['F'] = np.column_stack([f1, f2])

# ── NSGA-II 실행 ───────────────────────────────────────────────────────────────
print("NSGA-II 최적화 실행 중 (약 1~2분)...")

problem = LIDOptimizationProblem()

algorithm = NSGA2(
    pop_size    = 200,
    sampling    = FloatRandomSampling(),
    crossover   = SBX(prob=0.9, eta=15),
    mutation    = PM(eta=20),
    eliminate_duplicates=True
)

termination = get_termination('n_gen', 300)

result = minimize(problem, algorithm, termination,
                  seed=42, verbose=False)

print(f"  완료: Pareto 해 {len(result.F)}개")

# ── Pareto 해 정리 ─────────────────────────────────────────────────────────────
F = result.F    # [N_pareto, 2]: [runoff_m3, lid_area_m2]
X = result.X    # [N_pareto, 18]: coverage rates

runoff_reduction_pct = (BASELINE_RUNOFF - F[:, 0]) / BASELINE_RUNOFF * 100
lid_area_pct         = F[:, 1] / TOTAL_MAX_LID_M2 * 100

# 결과 DataFrame
df_pareto = pd.DataFrame(X, columns=[f'rate_{c}' for c in dong_codes])
df_pareto['runoff_m3']          = F[:, 0].round(1)
df_pareto['lid_area_m2']        = F[:, 1].round(1)
df_pareto['runoff_red_pct']     = runoff_reduction_pct.round(2)
df_pareto['lid_area_pct']       = lid_area_pct.round(2)
df_pareto = df_pareto.sort_values('runoff_red_pct').reset_index(drop=True)
df_pareto.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
print(f"  Pareto 해 저장 → {OUT_CSV}")

# ── 3개 대표 해 선택 ───────────────────────────────────────────────────────────
# S1: 저감율 최대 해
# S2: 설치면적 최소 해
# S3: 균형 해 (저감율/설치면적 비율 최대)
idx_max_red  = df_pareto['runoff_red_pct'].idxmax()
idx_min_area = df_pareto['lid_area_pct'].idxmin()
ratio        = df_pareto['runoff_red_pct'] / (df_pareto['lid_area_pct'] + 1e-6)
idx_balanced = ratio.idxmax()

sol_labels = {'S1 (Max Reduction)': idx_max_red,
              'S2 (Min Area)'      : idx_min_area,
              'S3 (Balanced)'      : idx_balanced}

print("\n=== 대표 Pareto 해 ===")
for label, idx in sol_labels.items():
    r = df_pareto.loc[idx]
    print(f"  {label}: 저감율={r['runoff_red_pct']:.1f}%  "
          f"설치면적={r['lid_area_pct']:.1f}% of max")

# ── 시각화 A: Pareto Front ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6), facecolor='white')
ax.set_facecolor('white')

sc = ax.scatter(df_pareto['lid_area_pct'], df_pareto['runoff_red_pct'],
                c=df_pareto['runoff_red_pct'], cmap='Blues',
                s=40, alpha=0.7, edgecolors='none', zorder=2,
                label='Pareto Solutions')

# 균일 시나리오 비교점 (기존 결과)
unif_csv = os.path.join(BASE_DIR, 'results/simulation_results.csv')
df_unif  = pd.read_csv(unif_csv)
unif_x   = df_unif['lid_rate_pct']   # ← 균일 적용률 자체를 x축에 표시
# 균일 적용 시 실제 설치면적 비율 계산
for _, row in df_unif.iterrows():
    rate      = row['lid_rate_pct'] / 100
    unif_area = (df_params['max_lid_m2'] * rate).sum()
    unif_area_pct = unif_area / TOTAL_MAX_LID_M2 * 100
    ax.scatter(unif_area_pct, row['vol_reduction_pct'],
               marker='D', s=60, color='#D6604D',
               edgecolors='white', linewidth=0.5, zorder=3)

# 대표 해 마킹
marker_styles = {'S1 (Max Reduction)': ('*', '#FFD700', 18),
                 'S2 (Min Area)'      : ('s', '#4DAC26', 12),
                 'S3 (Balanced)'      : ('^', '#FF6B35', 13)}
for label, idx in sol_labels.items():
    r = df_pareto.loc[idx]
    m, c, ms = marker_styles[label]
    ax.scatter(r['lid_area_pct'], r['runoff_red_pct'],
               marker=m, s=ms**2, color=c, edgecolors='black',
               linewidth=1.2, zorder=5, label=label)

plt.colorbar(sc, ax=ax, label='Runoff Reduction Rate (%)', pad=0.02)

handles, labels = ax.get_legend_handles_labels()
red_patch = mpatches.Patch(color='#D6604D', label='Uniform Coverage (Baseline Scenarios)')
ax.legend(handles=handles+[red_patch], fontsize=11, loc='lower right')

ax.set_xlabel('LID Installation Area (% of Max. Available Impervious Area)', fontsize=13)
ax.set_ylabel('Runoff Volume Reduction Rate (%)', fontsize=13)
ax.set_title('Pareto Front: Trade-off Between LID Installation Area\n'
             'and Stormwater Runoff Reduction (NSGA-II, Seocho-gu)',
             fontsize=14, fontweight='bold')
ax.grid(linestyle=':', alpha=0.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig_pareto_front.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("  Pareto Front 그래프 저장")

# ── 시각화 B: 대표 해의 공간 분포 지도 ────────────────────────────────────────
dong_gdf = gpd.read_file(BOUNDARY_SHP).to_crs('EPSG:5179')
dong_gdf['ADM_CD'] = dong_gdf['ADM_CD'].astype(str)

fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')
titles = list(sol_labels.keys())
indices= list(sol_labels.values())

for ax2, title, idx in zip(axes2, titles, indices):
    ax2.set_facecolor('white')
    rates_sol = df_pareto.loc[idx, [f'rate_{c}' for c in dong_codes]].values
    dong_gdf['lid_rate'] = rates_sol

    dong_gdf.plot(column='lid_rate', ax=ax2, cmap='YlOrRd',
                  vmin=0, vmax=100, edgecolor='gray', linewidth=0.5)

    red_r = df_pareto.loc[idx, 'runoff_red_pct']
    area_r= df_pareto.loc[idx, 'lid_area_pct']
    ax2.set_title(f'{title}\nReduction={red_r:.1f}%  Area={area_r:.1f}%',
                  fontsize=12, fontweight='bold')
    ax2.axis('off')

# 공통 컬러바
sm = plt.cm.ScalarMappable(cmap='YlOrRd',
                            norm=plt.Normalize(vmin=0, vmax=100))
sm.set_array([])
cbar = fig2.colorbar(sm, ax=axes2, shrink=0.7, pad=0.02)
cbar.set_label('Permeable Pavement Coverage Rate (%)', fontsize=12)

fig2.suptitle('Spatial Distribution of Optimal Permeable Pavement Coverage\n'
              '(NSGA-II Pareto-Optimal Solutions, Seocho-gu, Seoul)',
              fontsize=14, fontweight='bold', y=1.02)
fig2.savefig(os.path.join(FIG_DIR, 'fig_optimal_maps.png'),
             dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig2)
print("  최적 해 공간분포 지도 저장")

print(f"\n=== 공간 최적화 완료 ===")
print(f"  Pareto 해: {len(df_pareto)}개")
print(f"  결과: {OUT_CSV}")
print(f"  그래프: results/figures/fig_pareto_front.png")
print(f"          results/figures/fig_optimal_maps.png")
