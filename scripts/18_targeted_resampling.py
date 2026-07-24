"""
18_targeted_resampling.py
Targeted resampling in the low-coverage / spatially-concentrated design-space region
  (manuscript Section 4.7.2 and Supplementary Section S6)

Rationale
---------
The original training set (05_surrogate_training.py) draws per-dong coverage rates
independently from U[0, 100] via LHS, which produces predominantly *dispersed*
configurations. NSGA-II, when minimising installation area, drives solutions toward
*spatially concentrated* low-area configurations — a region of the 18-D design space
that is under-represented in the training data. This script adds 50 SWMM runs
targeted at that region, re-trains the XGBoost surrogate on n = 251, and re-evaluates
the 10 directly simulated Pareto solutions.

Design of the 50 targeted samples (seed = 4218, deterministic)
  - k active dongs per sample, k drawn cyclically from {1, 2, 3, 4, 5}
  - the k active dongs are selected by an LHS-driven permutation
  - active coverage rates drawn from U[10, 100] via LHS; all other dongs are 0
  This yields spatially concentrated configurations with low district-wide LID area,
  matching the region where the Pareto front concentrates.

Outputs
  - results/surrogate_training_data_resampled.csv   (n = 251)
  - results/targeted_resampling_results.csv         (per-Pareto-solution errors)
  - results/targeted_resampling_summary.csv         (before/after error summary)
  - model/resample_scenarios/rs_XXXX.inp            (50 SWMM input files)
"""

import os, time
import numpy as np
import pandas as pd
from scipy.stats import qmc
from pyswmm import Simulation, Nodes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb

SEED = 42
RESAMPLE_SEED = 4218
N_EXTRA = 50

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_INP    = os.path.join(BASE_DIR, 'model/seocho_baseline.inp')
SCEN_DIR    = os.path.join(BASE_DIR, 'model/resample_scenarios')
PARAMS_CSV  = os.path.join(BASE_DIR, 'results/subcatchment_params.csv')
TRAIN_CSV   = os.path.join(BASE_DIR, 'results/surrogate_training_data.csv')
PARETO_CSV  = os.path.join(BASE_DIR, 'results/pareto_solutions.csv')
VALID_CSV   = os.path.join(BASE_DIR, 'results/pareto_validation_results.csv')

OUT_TRAIN   = os.path.join(BASE_DIR, 'results/surrogate_training_data_resampled.csv')
OUT_RESULT  = os.path.join(BASE_DIR, 'results/targeted_resampling_results.csv')
OUT_SUMMARY = os.path.join(BASE_DIR, 'results/targeted_resampling_summary.csv')

os.makedirs(SCEN_DIR, exist_ok=True)

BASELINE_RUNOFF = 614298.0   # m³ (03_run_simulation.py)
LOW_AREA_THRESHOLD  = 20.0   # % of maximum available impervious area
HIGH_AREA_THRESHOLD = 40.0

df_params = pd.read_csv(PARAMS_CSV)
df_params['ADM_CD'] = df_params['ADM_CD'].astype(str)
DONG_CODES = df_params['ADM_CD'].tolist()
N_DONG     = len(DONG_CODES)

imp_areas = (df_params['area_ha'].values * 10_000 *
             df_params['impervious_pct'].values / 100)      # 동별 불투수면적 (m²)

# ── LID 블록 (05_surrogate_training.py와 동일) ────────────────────────────────
LID_CONTROL = """
[LID_CONTROLS]
;;Name             Type
PP                 PP

;;Name             Layer      Thick   VoidFrac  Rough    Slope  Side
PP                 SURFACE    0       0         0.013    0      0
PP                 PAVEMENT   100     0.15      0        500    0
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
        area_m2 = float(r['area_ha']) * 10_000
        imp_pct = float(r['impervious_pct'])
        lid_m2  = area_m2 * imp_pct / 100 * rate / 100
        if lid_m2 < 1:
            continue
        width = max(1.0, round(np.sqrt(lid_m2), 1))
        lines.append(f"SC_{code:<16} PP               1       "
                     f"{lid_m2:<10.1f} {width:<10.1f} "
                     f"0          100        0          *          *          0")
    lines.append("")
    return "\n".join(lines)


def generate_inp(rate_array, out_path):
    with open(BASE_INP, 'r', encoding='utf-8') as f:
        content = f.read()
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(content)
        if not np.all(np.asarray(rate_array) == 0):
            f.write("\n" + LID_CONTROL + "\n")
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


def weighted_coverage(X):
    """물리 기반 파생 피처: 불투수면적 가중 평균 적용률"""
    X = np.atleast_2d(X)
    return (X * imp_areas).sum(axis=1) / imp_areas.sum()


def build_features(X_raw):
    X_raw = np.atleast_2d(X_raw)
    return np.column_stack([X_raw, weighted_coverage(X_raw)])


def lid_area_pct(rate_array):
    """설치면적 (최대 가용 불투수면적 대비 %)"""
    return (np.atleast_2d(rate_array) / 100.0 * imp_areas).sum(axis=1) / imp_areas.sum() * 100


# ── Step 1: 저면적·공간집중 영역 50개 샘플 설계 ───────────────────────────────
print("=" * 74)
print("Step 1: Targeted resampling design (low-coverage / concentrated region)")
print("=" * 74)

sampler_k    = qmc.LatinHypercube(d=N_DONG, seed=RESAMPLE_SEED)
order_draw   = sampler_k.random(n=N_EXTRA)                    # 활성 동 선택용
sampler_rate = qmc.LatinHypercube(d=5, seed=RESAMPLE_SEED + 1)
rate_draw    = qmc.scale(sampler_rate.random(n=N_EXTRA), 10, 100)

rates_extra = np.zeros((N_EXTRA, N_DONG))
for i in range(N_EXTRA):
    k = (i % 5) + 1                                # 1~5개 동에만 집중 배치
    active = np.argsort(order_draw[i])[:k]          # LHS 기반 결정론적 선택
    rates_extra[i, active] = rate_draw[i, :k]

areas_extra = lid_area_pct(rates_extra)
print(f"  샘플 수         : {N_EXTRA}")
print(f"  활성 동 수 분포 : k = 1~5 (균등)")
print(f"  설치면적 범위   : {areas_extra.min():.2f} - {areas_extra.max():.2f} % "
      f"(평균 {areas_extra.mean():.2f} %)")

# 기존 학습셋의 설치면적 분포와 비교
df_train = pd.read_csv(TRAIN_CSV)
rate_cols = [f'rate_{c}' for c in DONG_CODES]
areas_orig = lid_area_pct(df_train[rate_cols].values)
print(f"  (기존 n=251 중 원 201개 설치면적 범위: "
      f"{areas_orig.min():.2f} - {areas_orig.max():.2f} %, 평균 {areas_orig.mean():.2f} %)")
print(f"  기존 학습셋 중 설치면적 ≤ {LOW_AREA_THRESHOLD:.0f} % 샘플 수: "
      f"{(areas_orig <= LOW_AREA_THRESHOLD).sum()} / {len(areas_orig)}")

# ── Step 2: 50개 추가 SWMM 실행 ───────────────────────────────────────────────
print("\n" + "=" * 74)
print(f"Step 2: SWMM 시뮬레이션 ({N_EXTRA}회)")
print("=" * 74)

records, t0 = [], time.time()
for i, rates in enumerate(rates_extra):
    inp_path = os.path.join(SCEN_DIR, f'rs_{i:04d}.inp')
    generate_inp(rates, inp_path)
    vol, peak = run_sim(inp_path)
    records.append(list(rates) + [vol, peak])
    eta = (time.time() - t0) / (i + 1) * (N_EXTRA - i - 1)
    print(f"  [{i+1:3d}/{N_EXTRA}] area={areas_extra[i]:5.2f}%  "
          f"vol={vol:,.0f} m³  peak={peak:.2f} m³/s  ETA {eta/60:.1f}min", end='\r')
print(f"\n  완료 ({(time.time()-t0)/60:.1f}분 소요)")

cols = rate_cols + ['total_runoff_m3', 'peak_flow_cms']
df_extra = pd.DataFrame(records, columns=cols)
df_resampled = pd.concat([df_train[cols], df_extra], ignore_index=True)
df_resampled.to_csv(OUT_TRAIN, index=False, encoding='utf-8-sig')
print(f"  통합 학습셋 저장 → {OUT_TRAIN}  (n = {len(df_resampled)})")

# ── Step 3: 원본(n=201) / 증강(n=251) XGBoost 재학습 ──────────────────────────
print("\n" + "=" * 74)
print("Step 3: XGBoost 재학습 (하이퍼파라미터·시드 동일)")
print("=" * 74)

XGB_KWARGS = dict(n_estimators=300, learning_rate=0.05, max_depth=4,
                  subsample=0.8, random_state=SEED, verbosity=0)

# 공통 홀드아웃: 원본 201개에 대한 80/20 분할을 고정하고 두 모델을 동일 테스트셋에서 평가.
# (증강 샘플을 테스트셋에 섞으면 저면적·집중 시나리오가 상대적으로 예측하기 쉬워
#  R²가 인위적으로 부풀려지므로, 일반화 성능 비교는 원 설계영역 홀드아웃에서 수행한다.)
X_orig = build_features(df_train[rate_cols].values)
y_orig = df_train['total_runoff_m3'].values
X_tr0, X_te0, y_tr0, y_te0 = train_test_split(X_orig, y_orig,
                                              test_size=0.2, random_state=SEED)

X_extra = build_features(df_extra[rate_cols].values)
y_extra = df_extra['total_runoff_m3'].values
X_tr1   = np.vstack([X_tr0, X_extra])
y_tr1   = np.concatenate([y_tr0, y_extra])


def fit_eval(X_tr, y_tr):
    mdl = xgb.XGBRegressor(**XGB_KWARGS)
    mdl.fit(X_tr, y_tr)
    pred = mdl.predict(X_te0)
    return (mdl, r2_score(y_te0, pred),
            float(np.sqrt(mean_squared_error(y_te0, pred))))


mdl_201, r2_201, rmse_201 = fit_eval(X_tr0, y_tr0)
mdl_251, r2_251, rmse_251 = fit_eval(X_tr1, y_tr1)
print(f"  공통 홀드아웃 (원본 201개의 20 % = {len(y_te0)}개 시나리오)")
print(f"  n = 201 (train {len(y_tr0)}) : R²_test = {r2_201:.4f}   RMSE = {rmse_201:,.0f} m³")
print(f"  n = 251 (train {len(y_tr1)}) : R²_test = {r2_251:.4f}   RMSE = {rmse_251:,.0f} m³")

# ── Step 4: 10개 Pareto 해 재예측 (SWMM 실측값은 06b 결과 재사용) ─────────────
print("\n" + "=" * 74)
print("Step 4: Pareto 해 10개 재예측 (직접 SWMM 실측값 대비)")
print("=" * 74)

df_val    = pd.read_csv(VALID_CSV, encoding='utf-8-sig')
df_pareto = pd.read_csv(PARETO_CSV)

rows = []
for _, v in df_val.iterrows():
    rates = df_pareto.loc[int(v['pareto_idx']), rate_cols].values.astype(float)
    Xf    = build_features(rates)
    swmm  = float(v['swmm_m3'])
    p201  = float(mdl_201.predict(Xf)[0])
    p251  = float(mdl_251.predict(Xf)[0])
    rows.append({
        'label'          : v['label'],
        'pareto_idx'     : int(v['pareto_idx']),
        'lid_area_pct'   : float(v['lid_area_pct']),
        'swmm_m3'        : round(swmm, 1),
        'pred_n201_m3'   : round(p201, 1),
        'pred_n251_m3'   : round(p251, 1),
        # 부호 규약은 06b_pareto_validation.py와 동일:
        # (+) = 대리모델이 유출량을 과소예측 → 저감효과를 과대예측
        'rel_err_n201_pct': round((swmm - p201) / swmm * 100, 2),
        'rel_err_n251_pct': round((swmm - p251) / swmm * 100, 2),
        'red_pct_swmm'   : round((BASELINE_RUNOFF - swmm) / BASELINE_RUNOFF * 100, 2),
    })

df_res = pd.DataFrame(rows)
df_res.to_csv(OUT_RESULT, index=False, encoding='utf-8-sig')

print(f"{'Solution':<18}{'Area%':>8}{'RelErr n=201':>14}{'RelErr n=251':>14}")
print("-" * 56)
for _, r in df_res.iterrows():
    print(f"  {r['label']:<16}{r['lid_area_pct']:>8.2f}"
          f"{r['rel_err_n201_pct']:>+14.2f}{r['rel_err_n251_pct']:>+14.2f}")

low  = df_res[df_res['lid_area_pct'] <= LOW_AREA_THRESHOLD]
high = df_res[df_res['lid_area_pct'] >= HIGH_AREA_THRESHOLD]

summary = pd.DataFrame([
    {'metric': f'Mean relative error, low-area (<= {LOW_AREA_THRESHOLD:.0f} % of max imp.)',
     'n': len(low),
     'n201': round(low['rel_err_n201_pct'].mean(), 2),
     'n251': round(low['rel_err_n251_pct'].mean(), 2)},
    {'metric': f'Mean |relative error|, low-area (<= {LOW_AREA_THRESHOLD:.0f} %)',
     'n': len(low),
     'n201': round(low['rel_err_n201_pct'].abs().mean(), 2),
     'n251': round(low['rel_err_n251_pct'].abs().mean(), 2)},
    {'metric': f'Max |relative error|, low-area (<= {LOW_AREA_THRESHOLD:.0f} %)',
     'n': len(low),
     'n201': round(low['rel_err_n201_pct'].abs().max(), 2),
     'n251': round(low['rel_err_n251_pct'].abs().max(), 2)},
    {'metric': f'Mean |relative error|, high-area (>= {HIGH_AREA_THRESHOLD:.0f} %)',
     'n': len(high),
     'n201': round(high['rel_err_n201_pct'].abs().mean(), 2),
     'n251': round(high['rel_err_n251_pct'].abs().mean(), 2)},
    {'metric': 'Mean |relative error|, all 10 Pareto solutions',
     'n': len(df_res),
     'n201': round(df_res['rel_err_n201_pct'].abs().mean(), 2),
     'n251': round(df_res['rel_err_n251_pct'].abs().mean(), 2)},
    {'metric': 'Hold-out R2_test',
     'n': '', 'n201': round(r2_201, 4), 'n251': round(r2_251, 4)},
    {'metric': 'Hold-out RMSE (m3)',
     'n': '', 'n201': round(rmse_201, 0), 'n251': round(rmse_251, 0)},
])
summary.to_csv(OUT_SUMMARY, index=False, encoding='utf-8-sig')

print("\n" + "=" * 74)
print("Step 5: 요약 (manuscript Section 4.7.2)")
print("=" * 74)
print(summary.to_string(index=False))
print(f"\n저장:\n  {OUT_TRAIN}\n  {OUT_RESULT}\n  {OUT_SUMMARY}")
