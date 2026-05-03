"""
05_surrogate_training.py
ML 대리모델 학습
  - Latin Hypercube Sampling으로 200개 시나리오 생성
  - 동별 개별 투수성포장 적용률 적용 (18차원 입력)
  - Random Forest / XGBoost / MLP 비교
  - 최적 모델 저장
출력:
  - results/surrogate_training_data.csv
  - results/surrogate_model_comparison.csv
  - model/surrogate_rf.pkl  (best model)
  - results/figures/fig_surrogate_*.png
"""

import os, sys, time, joblib
import numpy as np
import pandas as pd
from scipy.stats import qmc
from pyswmm import Simulation, Nodes
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.family':'DejaVu Sans','font.size':13,
                            'axes.spines.top':False,'axes.spines.right':False})

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_INP   = os.path.join(BASE_DIR, 'model/seocho_baseline.inp')
SCEN_DIR   = os.path.join(BASE_DIR, 'model/surrogate_scenarios')
PARAMS_CSV = os.path.join(BASE_DIR, 'results/subcatchment_params.csv')
DATA_CSV   = os.path.join(BASE_DIR, 'results/surrogate_training_data.csv')
COMP_CSV   = os.path.join(BASE_DIR, 'results/surrogate_model_comparison.csv')
FIG_DIR    = os.path.join(BASE_DIR, 'results/figures')
MODEL_DIR  = os.path.join(BASE_DIR, 'model')

os.makedirs(SCEN_DIR, exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR,exist_ok=True)

df_params = pd.read_csv(PARAMS_CSV)
df_params['ADM_CD'] = df_params['ADM_CD'].astype(str)
DONG_CODES = df_params['ADM_CD'].tolist()
DONG_NAMES = df_params['ADM_NM'].tolist()
N_DONG     = len(DONG_CODES)

# ── LID 블록 ───────────────────────────────────────────────────────────────────
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
    """동별 개별 적용률 (rate_array: shape [18,])"""
    lines = ["[LID_USAGE]",
             ";;Subcatchment        LIDProcess       Number  Area       Width      "
             "InitSat    FromImp    ToPerv     RptFile    DrainTo    FromPerv",
             ";;==================== ================ ======= ========== ========== "
             "========== ========== ========== ========== ========== =========="]
    for idx, (code, rate) in enumerate(zip(DONG_CODES, rate_array)):
        r       = df_params.loc[df_params['ADM_CD'] == code].iloc[0]
        area_m2 = float(r['area_ha']) * 10_000
        imp_pct = float(r['impervious_pct'])
        imp_m2  = area_m2 * imp_pct / 100
        lid_m2  = imp_m2  * rate   / 100
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
    if np.all(rate_array == 0):
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(content)
    else:
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(content)
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

# ── Step 1: Latin Hypercube Sampling ─────────────────────────────────────────
N_SAMPLES = 200
print(f"Step 1: Latin Hypercube Sampling ({N_SAMPLES} 시나리오 생성)")
sampler   = qmc.LatinHypercube(d=N_DONG, seed=42)
lhs       = sampler.random(n=N_SAMPLES)
rates_all = qmc.scale(lhs, 0, 100)    # [0, 100] 범위

# Baseline (모두 0) 추가
rates_all = np.vstack([np.zeros(N_DONG), rates_all])

# ── Step 2: 시뮬레이션 실행 ────────────────────────────────────────────────────
cols = [f'rate_{c}' for c in DONG_CODES] + ['total_runoff_m3', 'peak_flow_cms']
if os.path.exists(DATA_CSV):
    print(f"Step 2: 기존 데이터 로드 (시뮬레이션 건너뜀)")
    df_raw = pd.read_csv(DATA_CSV)
    print(f"Step 3: {len(df_raw)}개 시나리오 로드 완료 → {DATA_CSV}")
else:
    print(f"Step 2: 시뮬레이션 실행 (총 {len(rates_all)}개)...")
    records = []
    t0 = time.time()
    for i, rates in enumerate(rates_all):
        inp_path = os.path.join(SCEN_DIR, f'sc_{i:04d}.inp')
        generate_inp(rates, inp_path)
        vol, peak = run_sim(inp_path)
        records.append(list(rates) + [vol, peak])
        elapsed = time.time() - t0
        eta     = elapsed / (i+1) * (len(rates_all) - i - 1)
        print(f"  [{i+1:3d}/{len(rates_all)}] "
              f"vol={vol:,.0f} m³  peak={peak:.2f} m³/s  "
              f"ETA {eta/60:.1f}min", end='\r')
    print(f"\n  완료 ({(time.time()-t0)/60:.1f}분 소요)")
    df_raw = pd.DataFrame(records, columns=cols)
    df_raw.to_csv(DATA_CSV, index=False, encoding='utf-8-sig')
    print(f"Step 3: 데이터 저장 → {DATA_CSV}")

# ── Step 4: 모델 학습 및 비교 ─────────────────────────────────────────────────
print("Step 4: 모델 학습 중...")
X_raw = df_raw[[f'rate_{c}' for c in DONG_CODES]].values
y     = df_raw['total_runoff_m3'].values

# 물리 기반 파생 피처: 가중 평균 적용률 (불투수면적 가중)
imp_areas = (df_params['area_ha'].values * 10_000 *
             df_params['impervious_pct'].values / 100)   # 동별 불투수면적 (m²)
weighted_cov = (X_raw * imp_areas).sum(axis=1) / imp_areas.sum()  # 가중 평균 적용률

# 최종 입력 피처: 18개 동별 적용률 + 가중평균 피처
X = np.column_stack([X_raw, weighted_cov])

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

scaler   = StandardScaler()
X_tr_sc  = scaler.fit_transform(X_tr)
X_te_sc  = scaler.transform(X_te)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest'    : RandomForestRegressor(n_estimators=300, max_features=0.5,
                                               random_state=42, n_jobs=-1),
    'XGBoost'          : xgb.XGBRegressor(n_estimators=300, learning_rate=0.05,
                                           max_depth=4, subsample=0.8,
                                           random_state=42, verbosity=0),
}

comp_rows = []
trained   = {}
for name, mdl in models.items():
    if name in ['Linear Regression']:
        mdl.fit(X_tr_sc, y_tr)
        pred_te = mdl.predict(X_te_sc)
        pred_tr = mdl.predict(X_tr_sc)
    else:
        mdl.fit(X_tr, y_tr)
        pred_te = mdl.predict(X_te)
        pred_tr = mdl.predict(X_tr)

    r2_te  = r2_score(y_te, pred_te)
    rmse   = np.sqrt(mean_squared_error(y_te, pred_te))
    mae    = mean_absolute_error(y_te, pred_te)
    r2_tr  = r2_score(y_tr, pred_tr)

    comp_rows.append({'Model': name, 'R²_train': round(r2_tr,4),
                      'R²_test': round(r2_te,4),
                      'RMSE(m³)': round(rmse,1), 'MAE(m³)': round(mae,1)})
    trained[name] = mdl
    print(f"  {name:<16} R²_test={r2_te:.4f}  RMSE={rmse:,.0f} m³")

df_comp = pd.DataFrame(comp_rows)
df_comp.to_csv(COMP_CSV, index=False)

# 최적 모델 선택 (R²_test 기준)
best_name = df_comp.loc[df_comp['R²_test'].idxmax(), 'Model']
best_mdl  = trained[best_name]
print(f"\n  최적 모델: {best_name} (R²={df_comp.loc[df_comp['R²_test'].idxmax(),'R²_test']:.4f})")

joblib.dump({'model': best_mdl, 'scaler': scaler,
             'name': best_name, 'dong_codes': DONG_CODES,
             'needs_scaling': (best_name == 'Linear Regression'),
             'uses_weighted_feature': True},
            os.path.join(MODEL_DIR, 'surrogate_best.pkl'))
joblib.dump({'model': trained['Random Forest'], 'scaler': scaler,
             'dong_codes': DONG_CODES},
            os.path.join(MODEL_DIR, 'surrogate_rf.pkl'))
print(f"  모델 저장 → model/surrogate_best.pkl")

# ── Step 5: 시각화 ────────────────────────────────────────────────────────────
print("Step 5: 시각화 중...")

# (A) 모델 성능 비교 막대그래프
fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')
for ax in axes:
    ax.set_facecolor('white')

colors_bar = ['#2166AC', '#4DAC26', '#D6604D']
names  = [r['Model']    for r in comp_rows]
r2s    = [r['R²_test']  for r in comp_rows]
rmses  = [r['RMSE(m³)'] for r in comp_rows]

axes[0].bar(names, r2s, color=colors_bar, edgecolor='white', width=0.5)
axes[0].set_ylim(0, max(r2s) * 1.18)
axes[0].set_ylabel('R² (Test Set)', fontsize=13)
axes[0].set_title('(a) Model Accuracy (R²)', fontsize=14, fontweight='bold')
for i, v in enumerate(r2s):
    axes[0].text(i, v + max(r2s) * 0.02, f'{v:.4f}', ha='center', fontsize=12, fontweight='bold')

axes[1].bar(names, rmses, color=colors_bar, edgecolor='white', width=0.5)
axes[1].set_ylabel('RMSE (m³)', fontsize=13)
axes[1].set_title('(b) Prediction Error (RMSE)', fontsize=14, fontweight='bold')
for i, v in enumerate(rmses):
    axes[1].text(i, v + 100, f'{v:,.0f}', ha='center', fontsize=12, fontweight='bold')

fig.suptitle('ML Surrogate Model Performance Comparison\n'
             'for SWMM Stormwater Runoff Prediction (Seocho-gu, Seoul)',
             fontsize=14, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.88])
fig.savefig(os.path.join(FIG_DIR, 'fig_surrogate_comparison.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)

# (B) 예측 vs 실측 scatter (best model)
if best_name == 'MLP':
    pred_all = best_mdl.predict(scaler.transform(X))
else:
    pred_all = best_mdl.predict(X)

fig2, ax2 = plt.subplots(figsize=(7, 7), facecolor='white')
ax2.set_facecolor('white')
ax2.scatter(y/1e6, pred_all/1e6, alpha=0.5, s=30,
            color='#2166AC', edgecolors='none', label='Scenarios')
lim = [min(y.min(), pred_all.min())/1e6 * 0.98,
       max(y.max(), pred_all.max())/1e6 * 1.02]
ax2.plot(lim, lim, 'r--', linewidth=1.5, label='1:1 Line')
ax2.set_xlim(lim); ax2.set_ylim(lim)
ax2.set_xlabel('SWMM Simulated Runoff (million m³)', fontsize=13)
ax2.set_ylabel('ML Predicted Runoff (million m³)',   fontsize=13)
ax2.set_title(f'(c) Simulated vs. Predicted Runoff\n'
              f'({best_name}, R²={max(r2s):.4f})',
              fontsize=14, fontweight='bold')
ax2.legend(fontsize=12)
ax2.grid(linestyle=':', alpha=0.5)
fig2.tight_layout()
fig2.savefig(os.path.join(FIG_DIR, 'fig_surrogate_scatter.png'),
             dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig2)

print("\n=== 대리모델 학습 완료 ===")
print(df_comp.to_string(index=False))
print(f"\n저장 파일:")
print(f"  데이터  : {DATA_CSV}")
print(f"  모델    : model/surrogate_best.pkl")
print(f"  그래프  : results/figures/fig_surrogate_*.png")
