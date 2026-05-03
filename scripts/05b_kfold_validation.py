"""
05b_kfold_validation.py
Surrogate 모델 K-Fold 교차 검증 (V3 검증)
  - 기존 LHS 201개 데이터 재사용 (SWMM 재실행 없음)
  - Linear Regression / Random Forest / XGBoost 5-Fold CV
  - 기존 80/20 단순 분할 결과와 비교
출력:
  - results/kfold_cv_results.csv     (fold별 상세 결과)
  - results/kfold_cv_summary.csv     (모델별 평균 ± 표준편차)
  - results/figures/fig_kfold_boxplot.png
  - results/figures/fig_kfold_scatter.png  (fold별 예측 vs 실측)
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

matplotlib.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'font.size'        : 13,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'figure.dpi'       : 150,
    'savefig.dpi'      : 300,
    'savefig.bbox'     : 'tight',
})

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV   = os.path.join(BASE_DIR, 'results/surrogate_training_data.csv')
PARAMS_CSV = os.path.join(BASE_DIR, 'results/subcatchment_params.csv')
COMP_CSV   = os.path.join(BASE_DIR, 'results/surrogate_model_comparison.csv')
OUT_FOLD   = os.path.join(BASE_DIR, 'results/kfold_cv_results.csv')
OUT_SUMM   = os.path.join(BASE_DIR, 'results/kfold_cv_summary.csv')
FIG_DIR    = os.path.join(BASE_DIR, 'results/figures')
os.makedirs(FIG_DIR, exist_ok=True)

BLUE   = '#2166AC'
GREEN  = '#4DAC26'
ORANGE = '#D6604D'
GRAY   = '#888888'
N_FOLDS = 5

# ── 1. 데이터 로드 및 피처 구성 ────────────────────────────────────────────────
print("1. 데이터 로드 중...")
df_raw    = pd.read_csv(DATA_CSV)
df_params = pd.read_csv(PARAMS_CSV)
df_params['ADM_CD'] = df_params['ADM_CD'].astype(str)
DONG_CODES = df_params['ADM_CD'].tolist()

rate_cols = [f'rate_{c}' for c in DONG_CODES]
X_raw     = df_raw[rate_cols].values
y         = df_raw['total_runoff_m3'].values

# 물리 기반 파생 피처 (05_surrogate_training.py 와 동일)
imp_areas    = (df_params['area_ha'].values * 10_000
                * df_params['impervious_pct'].values / 100)
weighted_cov = (X_raw * imp_areas).sum(axis=1) / imp_areas.sum()
X            = np.column_stack([X_raw, weighted_cov])   # (201, 19)

print(f"   데이터 수: {len(y)}개  |  피처 수: {X.shape[1]}개 (18 동별 적용률 + 가중평균)")

# ── 2. 모델 정의 ───────────────────────────────────────────────────────────────
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest'    : RandomForestRegressor(n_estimators=300, max_features=0.5,
                                               random_state=42, n_jobs=-1),
    'XGBoost'          : xgb.XGBRegressor(n_estimators=300, learning_rate=0.05,
                                           max_depth=4, subsample=0.8,
                                           random_state=42, verbosity=0),
}

# ── 3. 5-Fold 교차 검증 ────────────────────────────────────────────────────────
print(f"\n2. {N_FOLDS}-Fold 교차 검증 실행 중...")
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

fold_records = []
all_preds    = {name: np.full(len(y), np.nan) for name in models}  # 전체 예측값 누적

for name, mdl in models.items():
    print(f"\n   [{name}]")
    for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(X)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # Linear Regression은 스케일링 적용
        if name == 'Linear Regression':
            sc = StandardScaler()
            X_tr_fit = sc.fit_transform(X_tr)
            X_te_fit = sc.transform(X_te)
        else:
            X_tr_fit, X_te_fit = X_tr, X_te

        mdl_clone = type(mdl)(**mdl.get_params())
        mdl_clone.fit(X_tr_fit, y_tr)
        pred = mdl_clone.predict(X_te_fit)

        all_preds[name][te_idx] = pred

        r2   = r2_score(y_te, pred)
        rmse = np.sqrt(mean_squared_error(y_te, pred))
        mae  = mean_absolute_error(y_te, pred)

        fold_records.append({
            'model': name, 'fold': fold_idx + 1,
            'n_test': len(te_idx),
            'R2': round(r2, 4),
            'RMSE': round(rmse, 1),
            'MAE': round(mae, 1),
        })
        print(f"     Fold {fold_idx+1}: R²={r2:.4f}  RMSE={rmse:,.0f} m³  MAE={mae:,.0f} m³")

df_fold = pd.DataFrame(fold_records)
df_fold.to_csv(OUT_FOLD, index=False, encoding='utf-8-sig')

# ── 4. 요약 통계 (평균 ± 표준편차) ────────────────────────────────────────────
print("\n3. 요약 통계 계산 중...")
summary_rows = []
for name in models:
    sub = df_fold[df_fold['model'] == name]
    summary_rows.append({
        'Model'           : name,
        'CV_R2_mean'      : round(sub['R2'].mean(),   4),
        'CV_R2_std'       : round(sub['R2'].std(),    4),
        'CV_RMSE_mean(m³)': round(sub['RMSE'].mean(), 1),
        'CV_RMSE_std(m³)' : round(sub['RMSE'].std(),  1),
        'CV_MAE_mean(m³)' : round(sub['MAE'].mean(),  1),
        'CV_MAE_std(m³)'  : round(sub['MAE'].std(),   1),
    })

df_summ = pd.DataFrame(summary_rows)

# 기존 80/20 단순 분할 결과 병합 (있는 경우)
if os.path.exists(COMP_CSV):
    df_prev = pd.read_csv(COMP_CSV)
    df_prev = df_prev.rename(columns={
        'R²_train' : 'Split_R2_train',
        'R²_test'  : 'Split_R2_test',
        'RMSE(m³)' : 'Split_RMSE(m³)',
        'MAE(m³)'  : 'Split_MAE(m³)',
        'Model'    : 'Model',
    })
    df_summ = df_summ.merge(df_prev, on='Model', how='left')

df_summ.to_csv(OUT_SUMM, index=False, encoding='utf-8-sig')

print("\n=== K-Fold CV 결과 요약 ===")
for _, r in df_summ.iterrows():
    print(f"  {r['Model']:<20}  "
          f"CV R²={r['CV_R2_mean']:.4f}±{r['CV_R2_std']:.4f}  "
          f"CV RMSE={r['CV_RMSE_mean(m³)']:,.0f}±{r['CV_RMSE_std(m³)']:,.0f} m³")

# ── 5. 시각화 ─────────────────────────────────────────────────────────────────
print("\n4. 시각화 중...")
colors_model = [BLUE, GREEN, ORANGE]
names_short  = ['Linear\nRegression', 'Random\nForest', 'XGBoost']

# ── Figure A: R² / RMSE 박스플롯 ──────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')
for ax in (ax1, ax2):
    ax.set_facecolor('white')

r2_by_model   = [df_fold[df_fold['model']==n]['R2'].values   for n in models]
rmse_by_model = [df_fold[df_fold['model']==n]['RMSE'].values for n in models]

bp1 = ax1.boxplot(r2_by_model, patch_artist=True, widths=0.5,
                  medianprops=dict(color='black', linewidth=2))
for patch, color in zip(bp1['boxes'], colors_model):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# 각 fold 점 표시
for i, (r2s, color) in enumerate(zip(r2_by_model, colors_model)):
    ax1.scatter([i+1]*len(r2s), r2s, color=color, zorder=5,
                s=60, edgecolors='white', linewidth=1.2)

ax1.set_xticks([1, 2, 3])
ax1.set_xticklabels(names_short, fontsize=12)
ax1.set_ylabel('R² (5-Fold CV)', fontsize=13)
ax1.set_title('(a) Cross-Validation R²\nby Model', fontsize=14, fontweight='bold')
ax1.grid(axis='y', linestyle=':', alpha=0.5)

bp2 = ax2.boxplot(rmse_by_model, patch_artist=True, widths=0.5,
                  medianprops=dict(color='black', linewidth=2))
for patch, color in zip(bp2['boxes'], colors_model):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

for i, (rmses, color) in enumerate(zip(rmse_by_model, colors_model)):
    ax2.scatter([i+1]*len(rmses), rmses, color=color, zorder=5,
                s=60, edgecolors='white', linewidth=1.2)

ax2.set_xticks([1, 2, 3])
ax2.set_xticklabels(names_short, fontsize=12)
ax2.set_ylabel('RMSE (m³)', fontsize=13)
ax2.set_title('(b) Cross-Validation RMSE\nby Model', fontsize=14, fontweight='bold')
ax2.grid(axis='y', linestyle=':', alpha=0.5)

fig.suptitle(f'{N_FOLDS}-Fold Cross-Validation Performance of ML Surrogate Models\n'
             'Seocho-gu Stormwater Runoff Prediction',
             fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig_kfold_boxplot.png'), facecolor='white')
plt.close(fig)
print("   fig_kfold_boxplot.png 저장 완료")

# ── Figure B: 3-panel 예측 vs 실측 scatter (모델별) ───────────────────────────
fig2, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='white')
for ax in axes:
    ax.set_facecolor('white')

for ax, name, color, short in zip(axes, models.keys(), colors_model, names_short):
    pred = all_preds[name]
    r2_cv = df_summ.loc[df_summ['Model']==name, 'CV_R2_mean'].values[0]
    rmse_cv = df_summ.loc[df_summ['Model']==name, 'CV_RMSE_mean(m³)'].values[0]

    ax.scatter(y/1e6, pred/1e6, alpha=0.5, s=30, color=color,
               edgecolors='none')
    lim = [min(y.min(), np.nanmin(pred))/1e6 * 0.98,
           max(y.max(), np.nanmax(pred))/1e6 * 1.02]
    ax.plot(lim, lim, 'k--', linewidth=1.5, alpha=0.7, label='1:1 line')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('SWMM Simulated (million m³)', fontsize=11)
    ax.set_ylabel('ML Predicted (million m³)', fontsize=11)
    ax.set_title(f'{short.replace(chr(10)," ")}\nCV R²={r2_cv:.3f}  RMSE={rmse_cv:,.0f} m³',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(linestyle=':', alpha=0.4)

fig2.suptitle('Out-of-Fold Predicted vs. Simulated Runoff Volume (5-Fold CV)',
              fontsize=14, fontweight='bold')
fig2.tight_layout()
fig2.savefig(os.path.join(FIG_DIR, 'fig_kfold_scatter.png'), facecolor='white')
plt.close(fig2)
print("   fig_kfold_scatter.png 저장 완료")

# ── 6. 논문용 표 출력 ─────────────────────────────────────────────────────────
print("\n=== 논문 Table 4 업데이트용 (CV 결과 포함) ===")
print(f"{'Model':<22} {'Split R²':>10} {'CV R² (mean±std)':>20} "
      f"{'Split RMSE':>12} {'CV RMSE (mean±std)':>22}")
print("-" * 90)
for _, r in df_summ.iterrows():
    split_r2   = f"{r.get('Split_R2_test', 'N/A')}"
    split_rmse = f"{r.get('Split_RMSE(m³)', 'N/A')}"
    cv_r2   = f"{r['CV_R2_mean']:.3f} ± {r['CV_R2_std']:.3f}"
    cv_rmse = f"{r['CV_RMSE_mean(m³)']:,.0f} ± {r['CV_RMSE_std(m³)']:,.0f}"
    print(f"  {r['Model']:<20} {split_r2:>10} {cv_r2:>20} {split_rmse:>12} {cv_rmse:>22}")

print(f"\n저장 파일:")
print(f"  fold별 결과 : {OUT_FOLD}")
print(f"  요약 통계   : {OUT_SUMM}")
print(f"  그래프      : results/figures/fig_kfold_*.png")
