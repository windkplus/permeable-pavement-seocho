"""
11_residual_diagnostics.py
Surrogate 모델 잔차·오차분포 진단 (R2 #3, R2 #18 대응)
  - 5-Fold OOF 예측값 대비 잔차 (Residual = SWMM - Surrogate)
  - 잔차의 정규성 / 편향(bias) / 평균제곱오차의 공간적 패턴
  - 입력공간(가중평균 적용률·총 LID 면적)에 따른 오차분포
출력:
  - results/residual_diagnostics.csv
  - results/figures/fig_residual_diagnostics.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from scipy import stats

matplotlib.rcParams.update({'font.family':'DejaVu Sans','font.size':12,
                            'axes.spines.top':False,'axes.spines.right':False,
                            'figure.dpi':150,'savefig.dpi':300,'savefig.bbox':'tight'})

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV   = os.path.join(BASE_DIR, 'results/surrogate_training_data.csv')
PARAMS_CSV = os.path.join(BASE_DIR, 'results/subcatchment_params.csv')
OUT_CSV    = os.path.join(BASE_DIR, 'results/residual_diagnostics.csv')
FIG_DIR    = os.path.join(BASE_DIR, 'results/figures')
os.makedirs(FIG_DIR, exist_ok=True)

BLUE='#2166AC'; ORANGE='#D6604D'; GRAY='#888888'

# ── 데이터 ────────────────────────────────────────────────────────────────────
df_raw    = pd.read_csv(DATA_CSV)
df_params = pd.read_csv(PARAMS_CSV)
df_params['ADM_CD'] = df_params['ADM_CD'].astype(str)
DONG_CODES = df_params['ADM_CD'].tolist()

rate_cols = [f'rate_{c}' for c in DONG_CODES]
X_raw = df_raw[rate_cols].values
y     = df_raw['total_runoff_m3'].values

imp_areas    = (df_params['area_ha'].values * 10_000 *
                df_params['impervious_pct'].values / 100)
weighted_cov = (X_raw * imp_areas).sum(axis=1) / imp_areas.sum()
total_lid    = (X_raw / 100 * imp_areas).sum(axis=1)
X = np.column_stack([X_raw, weighted_cov])

# ── 5-Fold OOF 예측 ───────────────────────────────────────────────────────────
print("5-Fold OOF 예측 계산 중 (3 모델)...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'XGBoost': xgb.XGBRegressor(n_estimators=300, learning_rate=0.05,
                                  max_depth=4, subsample=0.8,
                                  random_state=42, verbosity=0),
    'Random Forest': RandomForestRegressor(n_estimators=300, max_features=0.5,
                                             random_state=42, n_jobs=-1),
    'Linear Regression': LinearRegression(),
}

oof_preds = {n: np.full(len(y), np.nan) for n in models}
for name, mdl in models.items():
    for tr, te in kf.split(X):
        if name == 'Linear Regression':
            sc = StandardScaler()
            Xt = sc.fit_transform(X[tr]); Xe = sc.transform(X[te])
        else:
            Xt, Xe = X[tr], X[te]
        m = type(mdl)(**mdl.get_params())
        m.fit(Xt, y[tr])
        oof_preds[name][te] = m.predict(Xe)

# 잔차 = SWMM 실측 - 모델 예측
residuals = {n: y - oof_preds[n] for n in models}

# ── 잔차 통계 ──────────────────────────────────────────────────────────────────
diag_rows = []
for name, res in residuals.items():
    bias = res.mean()
    rmse = np.sqrt((res**2).mean())
    mae  = np.abs(res).mean()
    # Shapiro-Wilk 정규성 검정
    sw_stat, sw_p = stats.shapiro(res)
    # Skewness, Kurtosis
    skew = stats.skew(res)
    kurt = stats.kurtosis(res)
    diag_rows.append({
        'Model'    : name,
        'Bias_m3'  : round(bias, 1),
        'RMSE_m3'  : round(rmse, 1),
        'MAE_m3'   : round(mae,  1),
        'Skewness' : round(skew, 3),
        'Kurtosis' : round(kurt, 3),
        'Shapiro_W': round(sw_stat, 4),
        'Shapiro_p': round(sw_p,    4),
    })

df_diag = pd.DataFrame(diag_rows)
df_diag.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
print(df_diag.to_string(index=False))

# ── 시각화: 4-panel diagnostic plot for XGBoost (operational surrogate) ───────
print("\n진단 그림 생성 중...")
fig, axes = plt.subplots(2, 2, figsize=(13, 10), facecolor='white')
for ax in axes.ravel():
    ax.set_facecolor('white')

res_xgb = residuals['XGBoost']
pred_xgb = oof_preds['XGBoost']

# (a) Residual vs Predicted
ax = axes[0, 0]
ax.scatter(pred_xgb/1e6, res_xgb/1e3, alpha=0.5, s=30, color=BLUE, edgecolors='none')
ax.axhline(0, color='black', linewidth=0.8)
# LOWESS-like trend
from numpy.polynomial import polynomial as P
sort = np.argsort(pred_xgb)
ax.plot(pred_xgb[sort]/1e6,
        np.poly1d(np.polyfit(pred_xgb, res_xgb, 2))(pred_xgb[sort])/1e3,
        color=ORANGE, linewidth=2, label='Quadratic trend')
ax.set_xlabel('Predicted Runoff (million m³)')
ax.set_ylabel('Residual = SWMM − Surrogate (10³ m³)')
ax.set_title('(a) Residuals vs. Predicted (XGBoost, 5-Fold OOF)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(linestyle=':', alpha=0.4)

# (b) Histogram of residuals + Normal fit
ax = axes[0, 1]
counts, bins, _ = ax.hist(res_xgb/1e3, bins=25, color=BLUE, alpha=0.7,
                            edgecolor='white', density=True)
mu, sd = res_xgb.mean()/1e3, res_xgb.std()/1e3
xs = np.linspace(bins[0], bins[-1], 200)
ax.plot(xs, stats.norm.pdf(xs, mu, sd), color=ORANGE, linewidth=2,
        label=f'Normal(μ={mu:.1f}, σ={sd:.1f})')
ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_xlabel('Residual (10³ m³)')
ax.set_ylabel('Density')
sw_p = df_diag[df_diag['Model']=='XGBoost']['Shapiro_p'].values[0]
ax.set_title(f'(b) Residual Distribution (Shapiro p = {sw_p:.3f})',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(linestyle=':', alpha=0.4)

# (c) QQ plot
ax = axes[1, 0]
stats.probplot(res_xgb, dist='norm', plot=ax)
ax.get_lines()[0].set_color(BLUE)
ax.get_lines()[0].set_markersize(5)
ax.get_lines()[1].set_color(ORANGE)
ax.set_title('(c) Normal Q–Q Plot of Residuals', fontsize=12, fontweight='bold')
ax.set_xlabel('Theoretical Quantiles'); ax.set_ylabel('Sample Quantiles')
ax.grid(linestyle=':', alpha=0.4)

# (d) Residual vs LID area (input-space bias)
ax = axes[1, 1]
total_lid_pct = total_lid / imp_areas.sum() * 100
ax.scatter(total_lid_pct, res_xgb/1e3, alpha=0.5, s=30, color=BLUE, edgecolors='none')
ax.axhline(0, color='black', linewidth=0.8)
# 4 bins로 그룹화하여 평균 잔차
bins_e = np.percentile(total_lid_pct, [0, 25, 50, 75, 100])
labels = ['Q1','Q2','Q3','Q4']
binned = pd.cut(total_lid_pct, bins=bins_e, labels=labels, include_lowest=True)
group_means = pd.Series(res_xgb).groupby(binned).mean()
mids = [(bins_e[i]+bins_e[i+1])/2 for i in range(4)]
ax.plot(mids, group_means.values/1e3, '-o', color=ORANGE, linewidth=2,
        markersize=10, markeredgecolor='white', label='Quartile mean')
ax.set_xlabel('Total LID Area (% of Max Available Impervious)')
ax.set_ylabel('Residual = SWMM − Surrogate (10³ m³)')
ax.set_title('(d) Residual vs. Total LID Area', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(linestyle=':', alpha=0.4)

fig.suptitle('Residual Diagnostics for XGBoost Surrogate (5-Fold Out-of-Fold Predictions)',
             fontsize=14, fontweight='bold', y=1.00)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig_residual_diagnostics.png'), facecolor='white')
plt.close(fig)
print(f"   fig_residual_diagnostics.png 저장")
print(f"   {OUT_CSV}")
