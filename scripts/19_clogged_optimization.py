"""
19_clogged_optimization.py
Reduced-permeability (clogging) re-optimization
  (manuscript Section 4.7.3)

Rationale
---------
Section 5.6 shows that permeable-pavement effectiveness is strongly conditional on
maintaining surface permeability. This script asks whether the *spatial* conclusions
of the optimization survive clogging: it repeats the entire surrogate + NSGA-II
workflow under the "Mid" clogging assumption (pavement-layer permeability
100 mm/hr instead of the 500 mm/hr design value) and compares the resulting Pareto
front and per-dong allocation pattern against the design-permeability optimization.

Workflow (all seeds identical to the main pipeline)
  1. Re-run the same 201 LHS coverage matrices with PAVEMENT permeability = 100 mm/hr
  2. Re-train the XGBoost surrogate (same hyperparameters, seed 42, same
     physics-informed weighted-coverage feature)
  3. Re-run NSGA-II (pop = 200, 300 generations, seed 42)
  4. Compare Pareto fronts and the spatial allocation pattern of the
     minimum-area (S2) and maximum-reduction (S1) solutions

Outputs
  - results/surrogate_training_data_midperm.csv
  - results/pareto_solutions_midperm.csv
  - results/clogged_optimization_summary.csv
  - results/clogged_spatial_comparison.csv
  - results/figures/fig_clogged_optimization.png
  - model/midperm_scenarios/mp_XXXX.inp
"""

import os, time
import numpy as np
import pandas as pd
from scipy.stats import qmc, spearmanr
from pyswmm import Simulation, Nodes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

matplotlib.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                            'axes.spines.top': False, 'axes.spines.right': False,
                            'savefig.dpi': 300, 'savefig.bbox': 'tight'})

BLUE, ORANGE, GREEN, GRAY = '#2166AC', '#D6604D', '#4DAC26', '#888888'

SEED       = 42
N_SAMPLES  = 200            # 05_surrogate_training.py와 동일 (+ 무적용 baseline 1개)
MID_PERM   = 100            # mm/hr — Section 5.6 "Mid (3-5 yr)" 부분 막힘
BEST_PERM  = 500            # mm/hr — 설계값

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_INP    = os.path.join(BASE_DIR, 'model/seocho_baseline.inp')
SCEN_DIR    = os.path.join(BASE_DIR, 'model/midperm_scenarios')
PARAMS_CSV  = os.path.join(BASE_DIR, 'results/subcatchment_params.csv')
PARETO_BEST = os.path.join(BASE_DIR, 'results/pareto_solutions.csv')
FIG_DIR     = os.path.join(BASE_DIR, 'results/figures')

OUT_TRAIN   = os.path.join(BASE_DIR, 'results/surrogate_training_data_midperm.csv')
OUT_PARETO  = os.path.join(BASE_DIR, 'results/pareto_solutions_midperm.csv')
OUT_SUMMARY = os.path.join(BASE_DIR, 'results/clogged_optimization_summary.csv')
OUT_SPATIAL = os.path.join(BASE_DIR, 'results/clogged_spatial_comparison.csv')

os.makedirs(SCEN_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

BASELINE_RUNOFF = 614298.0   # m³ — 투수율과 무관 (LID 미적용)

df_params = pd.read_csv(PARAMS_CSV)
df_params['ADM_CD'] = df_params['ADM_CD'].astype(str)
DONG_CODES = df_params['ADM_CD'].tolist()
DONG_NAMES = df_params['ADM_NM'].tolist()
N_DONG     = len(DONG_CODES)
rate_cols  = [f'rate_{c}' for c in DONG_CODES]

imp_areas        = (df_params['area_ha'].values * 10_000 *
                    df_params['impervious_pct'].values / 100)
TOTAL_MAX_LID_M2 = imp_areas.sum()

# 그림용 로마자 표기 (DejaVu Sans에 한글 글리프가 없음)
ROMAN = {'11220510': 'Seocho 1', '11220520': 'Seocho 2', '11220530': 'Seocho 3',
         '11220540': 'Seocho 4', '11220550': 'Jamwon',   '11220560': 'Banpo-bon',
         '11220570': 'Banpo 1',  '11220580': 'Banpo 2',  '11220590': 'Banpo 3',
         '11220600': 'Banpo 4',  '11220610': 'Bangbae-bon', '11220620': 'Bangbae 1',
         '11220630': 'Bangbae 2', '11220640': 'Bangbae 3', '11220650': 'Bangbae 4',
         '11220660': 'Yangjae 1', '11220670': 'Yangjae 2', '11220680': 'Naegok'}
DONG_ROMAN = [ROMAN[c] for c in DONG_CODES]


def lid_control_block(perm_mmhr):
    """PAVEMENT 층 투수율만 변경 (10_permeability_sensitivity.py와 동일한 방식)"""
    return f"""
[LID_CONTROLS]
;;Name             Type
PP                 PP

;;Name             Layer      Thick   VoidFrac  Rough    Slope  Side
PP                 SURFACE    0       0         0.013    0      0
PP                 PAVEMENT   100     0.15      0        {perm_mmhr}    0
PP                 STORAGE    200     0.4       50       0
PP                 DRAIN      0       0.5       0        0      0      0
"""


def make_lid_usage_per_dong(rate_array):
    lines = ["[LID_USAGE]",
             ";;Subcatchment        LIDProcess       Number  Area       Width      "
             "InitSat    FromImp    ToPerv     RptFile    DrainTo    FromPerv",
             ";;==================== ================ ======= ========== ========== "
             "========== ========== ========== ========== ========== =========="]
    for code, rate in zip(DONG_CODES, rate_array):
        r       = df_params.loc[df_params['ADM_CD'] == code].iloc[0]
        lid_m2  = float(r['area_ha']) * 10_000 * float(r['impervious_pct']) / 100 * rate / 100
        if lid_m2 < 1:
            continue
        width = max(1.0, round(np.sqrt(lid_m2), 1))
        lines.append(f"SC_{code:<16} PP               1       "
                     f"{lid_m2:<10.1f} {width:<10.1f} "
                     f"0          100        0          *          *          0")
    lines.append("")
    return "\n".join(lines)


def generate_inp(rate_array, out_path, perm_mmhr):
    with open(BASE_INP, 'r', encoding='utf-8') as f:
        content = f.read()
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(content)
        if not np.all(np.asarray(rate_array) == 0):
            f.write("\n" + lid_control_block(perm_mmhr) + "\n")
            f.write(make_lid_usage_per_dong(rate_array))


def run_sim(inp_path):
    total, peak, prev_t = 0.0, 0.0, None
    with Simulation(inp_path) as sim:
        outfall = Nodes(sim)['OUTFALL_1']
        for _ in sim:
            q = outfall.total_inflow
            t = sim.current_time
            if prev_t:
                total += q * (t - prev_t).total_seconds()
            prev_t = t
            if q > peak:
                peak = q
    return total, peak


def build_features(X_raw):
    X_raw = np.atleast_2d(X_raw)
    wc = (X_raw * imp_areas).sum(axis=1) / imp_areas.sum()
    return np.column_stack([X_raw, wc])


# ── Step 1: 동일 LHS 설계를 100 mm/hr 조건에서 재실행 ─────────────────────────
print("=" * 74)
print(f"Step 1: LHS {N_SAMPLES + 1} scenarios re-run at pavement permeability "
      f"= {MID_PERM} mm/hr")
print("=" * 74)

sampler   = qmc.LatinHypercube(d=N_DONG, seed=SEED)      # 원 파이프라인과 동일 시드
rates_all = qmc.scale(sampler.random(n=N_SAMPLES), 0, 100)
rates_all = np.vstack([np.zeros(N_DONG), rates_all])     # baseline 포함

if os.path.exists(OUT_TRAIN):
    print("  기존 결과 로드 (시뮬레이션 건너뜀)")
    df_mid = pd.read_csv(OUT_TRAIN)
else:
    records, t0 = [], time.time()
    for i, rates in enumerate(rates_all):
        inp = os.path.join(SCEN_DIR, f'mp_{i:04d}.inp')
        generate_inp(rates, inp, MID_PERM)
        vol, peak = run_sim(inp)
        records.append(list(rates) + [vol, peak])
        eta = (time.time() - t0) / (i + 1) * (len(rates_all) - i - 1)
        print(f"  [{i+1:3d}/{len(rates_all)}] vol={vol:,.0f} m³  "
              f"ETA {eta:.0f}s", end='\r')
    wall = time.time() - t0
    print(f"\n  완료: {len(rates_all)}회, {wall:.1f}초 "
          f"({wall/len(rates_all):.2f} s/run)")
    df_mid = pd.DataFrame(records, columns=rate_cols + ['total_runoff_m3', 'peak_flow_cms'])
    df_mid.to_csv(OUT_TRAIN, index=False, encoding='utf-8-sig')
    print(f"  저장 → {OUT_TRAIN}")

# ── Step 2: 감소투수율 대리모델 학습 ──────────────────────────────────────────
print("\n" + "=" * 74)
print("Step 2: XGBoost surrogate re-training (Mid permeability)")
print("=" * 74)

X_mid = build_features(df_mid[rate_cols].values)
y_mid = df_mid['total_runoff_m3'].values
X_tr, X_te, y_tr, y_te = train_test_split(X_mid, y_mid, test_size=0.2, random_state=SEED)

surrogate = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                             subsample=0.8, random_state=SEED, verbosity=0)
surrogate.fit(X_tr, y_tr)
pred = surrogate.predict(X_te)
r2_mid   = r2_score(y_te, pred)
rmse_mid = float(np.sqrt(mean_squared_error(y_te, pred)))
print(f"  R²_test = {r2_mid:.4f}   RMSE = {rmse_mid:,.0f} m³")
print(f"  (design-permeability surrogate: R²_test = 0.789, RMSE = 3,718 m³)")

# ── Step 3: NSGA-II 재최적화 ──────────────────────────────────────────────────
print("\n" + "=" * 74)
print("Step 3: NSGA-II re-optimization (Mid permeability surrogate)")
print("=" * 74)


class LIDProblem(Problem):
    def __init__(self):
        super().__init__(n_var=N_DONG, n_obj=2, n_constr=0,
                         xl=np.zeros(N_DONG), xu=np.full(N_DONG, 100.0))

    def _evaluate(self, X, out, *args, **kwargs):
        f1 = surrogate.predict(build_features(X))
        f2 = (X / 100.0 * imp_areas).sum(axis=1)
        out['F'] = np.column_stack([f1, f2])


result = minimize(LIDProblem(),
                  NSGA2(pop_size=200, sampling=FloatRandomSampling(),
                        crossover=SBX(prob=0.9, eta=15), mutation=PM(eta=20),
                        eliminate_duplicates=True),
                  get_termination('n_gen', 300), seed=SEED, verbose=False)

F, X = result.F, result.X
df_mp = pd.DataFrame(X, columns=rate_cols)
df_mp['runoff_m3']      = F[:, 0].round(1)
df_mp['lid_area_m2']    = F[:, 1].round(1)
df_mp['runoff_red_pct'] = ((BASELINE_RUNOFF - F[:, 0]) / BASELINE_RUNOFF * 100).round(2)
df_mp['lid_area_pct']   = (F[:, 1] / TOTAL_MAX_LID_M2 * 100).round(2)
df_mp = df_mp.sort_values('runoff_red_pct').reset_index(drop=True)
df_mp.to_csv(OUT_PARETO, index=False, encoding='utf-8-sig')
print(f"  Pareto 해 {len(df_mp)}개 → {OUT_PARETO}")

# ── Step 4: 설계투수율 Pareto front와 비교 ────────────────────────────────────
print("\n" + "=" * 74)
print("Step 4: Pareto front comparison (500 vs. 100 mm/hr)")
print("=" * 74)

df_bp = pd.read_csv(PARETO_BEST)

def front_stats(df, tag):
    return {
        'front': tag,
        'n_solutions'      : len(df),
        'red_min_pct'      : round(df['runoff_red_pct'].min(), 2),
        'red_max_pct'      : round(df['runoff_red_pct'].max(), 2),
        'area_min_pct'     : round(df['lid_area_pct'].min(), 2),
        'area_max_pct'     : round(df['lid_area_pct'].max(), 2),
    }

# 동일 저감률을 달성하는 데 필요한 설치면적 (선형보간)
def area_for_reduction(df, target_red):
    d = df.sort_values('runoff_red_pct')
    if target_red < d['runoff_red_pct'].min() or target_red > d['runoff_red_pct'].max():
        return np.nan
    return float(np.interp(target_red, d['runoff_red_pct'], d['lid_area_pct']))

rows = [front_stats(df_bp, f'Design permeability ({BEST_PERM} mm/hr)'),
        front_stats(df_mp, f'Mid clogging ({MID_PERM} mm/hr)')]

common_reds = [14, 15, 16, 17]
for tr in common_reds:
    a_b, a_m = area_for_reduction(df_bp, tr), area_for_reduction(df_mp, tr)
    rows.append({'front': f'Area (% of max imp.) required for {tr} % reduction',
                 'n_solutions': '',
                 'red_min_pct': '', 'red_max_pct': '',
                 'area_min_pct': round(a_b, 2) if a_b == a_b else 'n/a',
                 'area_max_pct': round(a_m, 2) if a_m == a_m else 'n/a'})

df_sum = pd.DataFrame(rows)
df_sum.to_csv(OUT_SUMMARY, index=False, encoding='utf-8-sig')
print(df_sum.to_string(index=False))

# ── Step 5: 공간 배분 패턴 비교 ───────────────────────────────────────────────
print("\n" + "=" * 74)
print("Step 5: Spatial allocation pattern — S2 (min area) and S1 (max reduction)")
print("=" * 74)


def solution_rates(df, kind):
    idx = df['lid_area_pct'].idxmin() if kind == 'S2' else df['runoff_red_pct'].idxmax()
    return df.loc[idx, rate_cols].values.astype(float), df.loc[idx]


spatial_rows = []
for kind in ['S2', 'S1']:
    r_best, meta_b = solution_rates(df_bp, kind)
    r_mid,  meta_m = solution_rates(df_mp, kind)
    rho, p_rho = spearmanr(r_best, r_mid)
    top5_b = set(np.argsort(r_best)[::-1][:5])
    top5_m = set(np.argsort(r_mid)[::-1][:5])
    top3_b = set(np.argsort(r_best)[::-1][:3])
    top3_m = set(np.argsort(r_mid)[::-1][:3])
    print(f"\n  [{kind}]  design: red={meta_b['runoff_red_pct']:.2f} % / "
          f"area={meta_b['lid_area_pct']:.2f} %   |   "
          f"mid: red={meta_m['runoff_red_pct']:.2f} % / area={meta_m['lid_area_pct']:.2f} %")
    print(f"    Spearman rho (per-dong coverage) = {rho:+.3f} (p = {p_rho:.4f})")
    print(f"    Top-3 overlap = {len(top3_b & top3_m)}/3   Top-5 overlap = {len(top5_b & top5_m)}/5")
    print(f"    Top-5 design : {[DONG_NAMES[i] for i in np.argsort(r_best)[::-1][:5]]}")
    print(f"    Top-5 mid    : {[DONG_NAMES[i] for i in np.argsort(r_mid)[::-1][:5]]}")
    for i in range(N_DONG):
        spatial_rows.append({'solution': kind, 'ADM_CD': DONG_CODES[i],
                             'ADM_NM': DONG_NAMES[i],
                             'rate_design_pct': round(r_best[i], 2),
                             'rate_midperm_pct': round(r_mid[i], 2)})
    spatial_rows.append({'solution': f'{kind}_STAT', 'ADM_CD': 'spearman_rho',
                         'ADM_NM': f'p={p_rho:.4f}',
                         'rate_design_pct': round(rho, 3),
                         'rate_midperm_pct': f'top3={len(top3_b & top3_m)}/3, '
                                             f'top5={len(top5_b & top5_m)}/5'})

# ── Step 5b: front 전체 평균 배분 패턴 (단일 극단해보다 안정적인 통계) ────────
# 최소면적(S2)·최대저감(S1) 같은 front 끝점은 최적화가 거의 무차별한 영역에 있어
# 개별 동 배분이 불안정하다. 따라서 front 200개 해 전체의 평균 배분으로도 비교한다.
print("\n" + "-" * 74)
print("Step 5b: Front-averaged allocation pattern (all 200 Pareto solutions)")
print("-" * 74)

mean_b = df_bp[rate_cols].values.mean(axis=0)
mean_m = df_mp[rate_cols].values.mean(axis=0)
rho_f, p_f = spearmanr(mean_b, mean_m)
ob, om = np.argsort(mean_b)[::-1], np.argsort(mean_m)[::-1]
print(f"  Spearman rho (front-mean per-dong coverage) = {rho_f:+.3f} (p = {p_f:.4f})")
print(f"  Top-3 overlap = {len(set(ob[:3]) & set(om[:3]))}/3   "
      f"Top-5 overlap = {len(set(ob[:5]) & set(om[:5]))}/5")
print(f"    design top-5 : {[DONG_NAMES[i] for i in ob[:5]]}")
print(f"    mid    top-5 : {[DONG_NAMES[i] for i in om[:5]]}")

# 지역 클러스터 단위 배분 비중 (동 단위 순위보다 계획적으로 의미 있는 집계)
def cluster_of(name):
    if name.startswith('방배'):
        return 'Bangbae (south)'
    if name.startswith('반포'):
        return 'Banpo (north)'
    if name.startswith('서초'):
        return 'Seocho (central)'
    return 'Yangjae / Naegok (large peripheral)'


clusters = {}
for i, nm in enumerate(DONG_NAMES):
    clusters.setdefault(cluster_of(nm), []).append(i)

area_b = df_bp[rate_cols].values / 100 * imp_areas
area_m = df_mp[rate_cols].values / 100 * imp_areas
cluster_rows = []
print(f"\n  {'Cluster':<36}{'imp. share':>12}{'design':>10}{'mid':>10}{'design/imp':>12}{'mid/imp':>10}")
for cname, idx in clusters.items():
    imp_share = imp_areas[idx].sum() / imp_areas.sum() * 100
    sh_b = (area_b[:, idx].sum(axis=1) / area_b.sum(axis=1)).mean() * 100
    sh_m = (area_m[:, idx].sum(axis=1) / area_m.sum(axis=1)).mean() * 100
    print(f"  {cname:<36}{imp_share:>11.1f}%{sh_b:>9.1f}%{sh_m:>9.1f}%"
          f"{sh_b/imp_share:>11.2f}x{sh_m/imp_share:>9.2f}x")
    cluster_rows.append({'solution': 'FRONT_MEAN_CLUSTER', 'ADM_CD': cname,
                         'ADM_NM': f'imp_share={imp_share:.1f}%',
                         'rate_design_pct': round(sh_b, 2),
                         'rate_midperm_pct': round(sh_m, 2)})

for i in range(N_DONG):
    spatial_rows.append({'solution': 'FRONT_MEAN', 'ADM_CD': DONG_CODES[i],
                         'ADM_NM': DONG_NAMES[i],
                         'rate_design_pct': round(mean_b[i], 2),
                         'rate_midperm_pct': round(mean_m[i], 2)})
spatial_rows.append({'solution': 'FRONT_MEAN_STAT', 'ADM_CD': 'spearman_rho',
                     'ADM_NM': f'p={p_f:.4f}',
                     'rate_design_pct': round(rho_f, 3),
                     'rate_midperm_pct': f'top3={len(set(ob[:3]) & set(om[:3]))}/3, '
                                         f'top5={len(set(ob[:5]) & set(om[:5]))}/5'})
spatial_rows.extend(cluster_rows)

pd.DataFrame(spatial_rows).to_csv(OUT_SPATIAL, index=False, encoding='utf-8-sig')
print(f"\n  저장 → {OUT_SPATIAL}")

# ── Step 6: 그림 ──────────────────────────────────────────────────────────────
fig, (axa, axb) = plt.subplots(1, 2, figsize=(13, 5.4), facecolor='white')
for ax in (axa, axb):
    ax.set_facecolor('white')

axa.scatter(df_bp['lid_area_pct'], df_bp['runoff_red_pct'], s=22, alpha=0.75,
            color=BLUE, edgecolors='none', label=f'Design permeability ({BEST_PERM} mm/hr)')
axa.scatter(df_mp['lid_area_pct'], df_mp['runoff_red_pct'], s=22, alpha=0.75,
            color=ORANGE, edgecolors='none', label=f'Mid clogging ({MID_PERM} mm/hr)')
axa.set_xlabel('LID installation area (% of max. available impervious area)')
axa.set_ylabel('Runoff volume reduction (%)')
axa.set_title('(a) Pareto front shift under clogging', fontweight='bold')
axa.legend(fontsize=10, loc='lower right')
axa.grid(linestyle=':', alpha=0.5)

r_best_s2, _ = solution_rates(df_bp, 'S2')
r_mid_s2, _  = solution_rates(df_mp, 'S2')
order = np.argsort(r_best_s2)[::-1]
xpos  = np.arange(N_DONG)
w = 0.4
axb.bar(xpos - w/2, r_best_s2[order], width=w, color=BLUE, alpha=0.85,
        label=f'Design ({BEST_PERM} mm/hr)')
axb.bar(xpos + w/2, r_mid_s2[order], width=w, color=ORANGE, alpha=0.85,
        label=f'Mid clogging ({MID_PERM} mm/hr)')
axb.set_xticks(xpos)
axb.set_xticklabels([DONG_ROMAN[i] for i in order], rotation=60, ha='right', fontsize=8)
axb.set_ylabel('Per-dong PP coverage rate (%)')
axb.set_title('(b) Minimum-area (S2) spatial allocation', fontweight='bold')
axb.legend(fontsize=10)
axb.grid(axis='y', linestyle=':', alpha=0.5)

fig.suptitle('Reduced-permeability (clogging) re-optimization — Seocho-gu',
             fontsize=13, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(FIG_DIR, 'fig_clogged_optimization.png'), facecolor='white')
plt.close(fig)
print(f"  그림 저장 → results/figures/fig_clogged_optimization.png")

print("\n=== 완료 ===")
