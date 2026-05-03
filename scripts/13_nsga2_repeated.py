"""
13_nsga2_repeated.py
"""
import sys
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
"""
NSGA-II 반복 실행 + 수렴/안정성 진단 (R2 #10, #20 대응)
  - 5개 독립 random seed (42, 0, 7, 123, 2025) 로 반복 실행
  - 각 run의 hypervolume / spacing / num-Pareto 변화 추적
  - 5개 Pareto front 중첩 시각화
출력:
  - results/nsga2_repeated_runs.csv
  - results/nsga2_convergence.csv
  - results/figures/fig_nsga2_convergence.png
  - results/figures/fig_nsga2_pareto_overlay.png
"""

import os, joblib, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.indicators.hv import HV

matplotlib.rcParams.update({'font.family':'DejaVu Sans','font.size':12,
                            'axes.spines.top':False,'axes.spines.right':False,
                            'figure.dpi':150,'savefig.dpi':300,'savefig.bbox':'tight'})

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS_CSV   = os.path.join(BASE_DIR, 'results/subcatchment_params.csv')
FIG_DIR      = os.path.join(BASE_DIR, 'results/figures')
OUT_RUNS     = os.path.join(BASE_DIR, 'results/nsga2_repeated_runs.csv')
OUT_CONV     = os.path.join(BASE_DIR, 'results/nsga2_convergence.csv')

BLUE='#2166AC'; ORANGE='#D6604D'; GREEN='#4DAC26'; PURPLE='#762A83'; GOLD='#E69F00'
COLORS = [BLUE, ORANGE, GREEN, PURPLE, GOLD]

df_params = pd.read_csv(PARAMS_CSV)
df_params['ADM_CD'] = df_params['ADM_CD'].astype(str)
DONG_CODES = df_params['ADM_CD'].tolist()
imp_areas  = (df_params['area_ha'].values * 10_000 *
              df_params['impervious_pct'].values / 100)
TOTAL_MAX_LID_M2 = imp_areas.sum()
BASELINE_RUNOFF  = 614298.0

# ── 대리모델 로드 ──────────────────────────────────────────────────────────────
bundle    = joblib.load(os.path.join(BASE_DIR, 'model/surrogate_best.pkl'))
surrogate = bundle['model']
uses_w    = bundle.get('uses_weighted_feature', False)

def predict_runoff(X):
    X = np.atleast_2d(X)
    if uses_w:
        wc = (X * imp_areas).sum(axis=1, keepdims=True) / imp_areas.sum()
        X  = np.hstack([X, wc])
    return surrogate.predict(X)

class LIDProblem(Problem):
    def __init__(self):
        n = len(DONG_CODES)
        super().__init__(n_var=n, n_obj=2, n_constr=0,
                         xl=np.zeros(n), xu=np.full(n, 100.0))
    def _evaluate(self, X, out, *_, **__):
        f1 = predict_runoff(X)
        f2 = (X / 100.0 * imp_areas).sum(axis=1)
        out['F'] = np.column_stack([f1, f2])

# ── 5회 반복 실행 ──────────────────────────────────────────────────────────────
SEEDS = [42, 0, 7, 123, 2025]
N_GEN = 300
all_runs = []         # 모든 run의 Pareto 해
conv_records = []     # 세대별 hypervolume

# Hypervolume reference: 보수적 상한 (Baseline + 10%, 전 면적)
HV_REF = np.array([BASELINE_RUNOFF * 1.10, TOTAL_MAX_LID_M2 * 1.10])
hv_ind = HV(ref_point=HV_REF)

print("="*70)
print(f"NSGA-II 반복 실행 (5 seeds, {N_GEN} generations each)")
print(f"HV reference point: ({HV_REF[0]:,.0f}, {HV_REF[1]:,.0f})")
print("="*70)

for run_idx, seed in enumerate(SEEDS):
    print(f"\n[Run {run_idx+1}/5] seed={seed}")
    t0 = time.time()
    problem = LIDProblem()
    algorithm = NSGA2(
        pop_size=200,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
        save_history=True,
    )
    res = minimize(problem, algorithm,
                   get_termination('n_gen', N_GEN),
                   seed=seed, verbose=False, save_history=True)
    elapsed = time.time() - t0
    print(f"   완료 ({elapsed:.1f}s) — Pareto 해 {len(res.F)}개")

    # 세대별 HV (10세대 단위로 sampling 비용 절감)
    print(f"   세대별 HV 계산...")
    for gen_idx, h in enumerate(res.history):
        if (gen_idx + 1) % 10 == 0 or gen_idx == 0 or gen_idx == len(res.history)-1:
            try:
                hv = hv_ind(h.opt.get('F'))
            except Exception:
                hv = np.nan
            conv_records.append({
                'run': run_idx+1, 'seed': seed,
                'generation': gen_idx+1,
                'hypervolume': hv,
                'n_pareto': len(h.opt),
            })

    # Pareto 해 저장
    F = res.F  # [N, 2]
    X = res.X
    runoff_red = (BASELINE_RUNOFF - F[:, 0]) / BASELINE_RUNOFF * 100
    area_pct   = F[:, 1] / TOTAL_MAX_LID_M2 * 100
    final_hv   = hv_ind(F)
    print(f"   최종 HV = {final_hv:.4e}")
    for i in range(len(F)):
        all_runs.append({
            'run': run_idx+1, 'seed': seed,
            'pareto_idx': i,
            'runoff_m3': F[i, 0], 'lid_area_m2': F[i, 1],
            'runoff_red_pct': runoff_red[i],
            'lid_area_pct': area_pct[i],
        })

df_runs = pd.DataFrame(all_runs)
df_runs.to_csv(OUT_RUNS, index=False, encoding='utf-8-sig')
df_conv = pd.DataFrame(conv_records)
df_conv.to_csv(OUT_CONV, index=False, encoding='utf-8-sig')

# ── 통계 요약 ──────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("=== 반복 실행 통계 요약 ===")
print("="*70)
final_hvs   = df_conv[df_conv['generation'] == N_GEN]
final_pareto_n = df_runs.groupby('run').size()
final_red_max  = df_runs.groupby('run')['runoff_red_pct'].max()
final_red_min  = df_runs.groupby('run')['runoff_red_pct'].min()
final_area_min = df_runs.groupby('run')['lid_area_pct'].min()

print(f"  Final HV   : {final_hvs['hypervolume'].mean():.4e} ± {final_hvs['hypervolume'].std():.4e}")
print(f"               (CV = {final_hvs['hypervolume'].std()/final_hvs['hypervolume'].mean()*100:.2f}%)")
print(f"  N_Pareto   : {final_pareto_n.mean():.0f} ± {final_pareto_n.std():.1f}")
print(f"  Max Reduction (%): {final_red_max.mean():.2f} ± {final_red_max.std():.3f}")
print(f"  Min Reduction (%): {final_red_min.mean():.2f} ± {final_red_min.std():.3f}")
print(f"  Min Area (%)     : {final_area_min.mean():.2f} ± {final_area_min.std():.3f}")

# ── 시각화: HV 수렴 곡선 ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
ax.set_facecolor('white')
for run_idx, seed in enumerate(SEEDS):
    sub = df_conv[df_conv['run'] == run_idx+1]
    ax.plot(sub['generation'], sub['hypervolume'],
            color=COLORS[run_idx], linewidth=1.8, alpha=0.85,
            label=f'Run {run_idx+1} (seed={seed})')
ax.set_xlabel('Generation')
ax.set_ylabel('Hypervolume')
ax.set_title('NSGA-II Convergence — 5 Independent Runs\n'
             '(Hypervolume Indicator, Reference Point = 110% of upper bound)',
             fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(linestyle=':', alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig_nsga2_convergence.png'), facecolor='white')
plt.close(fig)
print(f"\n그림 저장: fig_nsga2_convergence.png")

# ── 시각화: 5개 Pareto front 중첩 ──────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(10, 7), facecolor='white')
ax2.set_facecolor('white')
for run_idx, seed in enumerate(SEEDS):
    sub = df_runs[df_runs['run'] == run_idx+1].sort_values('runoff_red_pct')
    ax2.plot(sub['lid_area_pct'], sub['runoff_red_pct'],
             '-', color=COLORS[run_idx], linewidth=1.5, alpha=0.7,
             label=f'Run {run_idx+1} (seed={seed})')
    ax2.scatter(sub['lid_area_pct'], sub['runoff_red_pct'],
                s=15, color=COLORS[run_idx], edgecolors='none', alpha=0.5)
ax2.set_xlabel('LID Installation Area (% of Max. Available Impervious)')
ax2.set_ylabel('Runoff Volume Reduction Rate (%)')
ax2.set_title('Stability of NSGA-II Pareto Front Across 5 Independent Runs',
              fontsize=13, fontweight='bold')
ax2.legend(loc='lower right', fontsize=10)
ax2.grid(linestyle=':', alpha=0.4)
fig2.tight_layout()
fig2.savefig(os.path.join(FIG_DIR, 'fig_nsga2_pareto_overlay.png'), facecolor='white')
plt.close(fig2)
print(f"그림 저장: fig_nsga2_pareto_overlay.png")
print(f"\n결과 저장:")
print(f"  {OUT_RUNS}")
print(f"  {OUT_CONV}")
