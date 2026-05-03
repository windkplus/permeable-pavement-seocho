"""
12_shap_xgboost_consistent.py
"""
import sys
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
"""
SHAP을 운영 surrogate(XGBoost)에 적용 — R2 #12 대응
  - 기존 07_shap_analysis.py 는 RF (R²=0.65) 에 SHAP 적용 → 최적화에 쓰인 XGBoost와 모델 불일치
  - 본 스크립트는 동일한 XGBoost (R²_test=0.79) 에 SHAP 적용하여 일관성 확보
출력:
  - results/shap_xgboost_importance.csv
  - results/figures/fig_shap_xgb_summary.png
  - results/figures/fig_shap_xgb_dependence.png
  - results/figures/fig_shap_xgb_spatial.png
"""

import os, joblib, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib
import geopandas as gpd
import xgboost as xgb
from sklearn.model_selection import train_test_split

matplotlib.rcParams.update({'font.family':'DejaVu Sans','font.size':12,
                            'axes.spines.top':False,'axes.spines.right':False,
                            'figure.dpi':150,'savefig.dpi':300,'savefig.bbox':'tight'})

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV     = os.path.join(BASE_DIR, 'results/surrogate_training_data.csv')
PARAMS_CSV   = os.path.join(BASE_DIR, 'results/subcatchment_params.csv')
BOUNDARY_SHP = os.path.join(BASE_DIR, 'data/boundary/bnd_dong_11220_2025_2Q/bnd_dong_11220_2025_2Q.shp')
FIG_DIR      = os.path.join(BASE_DIR, 'results/figures')
SHAP_CSV     = os.path.join(BASE_DIR, 'results/shap_xgboost_importance.csv')
os.makedirs(FIG_DIR, exist_ok=True)

df_params = pd.read_csv(PARAMS_CSV)
df_params['ADM_CD'] = df_params['ADM_CD'].astype(str)
DONG_CODES = df_params['ADM_CD'].tolist()
DONG_NAMES = df_params['ADM_NM'].tolist()
DONG_EN    = ['Seocho-1','Seocho-2','Seocho-3','Seocho-4',
              'Jamwon','Banpo-bon','Banpo-1','Banpo-2',
              'Banpo-3','Banpo-4','Bangbae-bon',
              'Bangbae-1','Bangbae-2','Bangbae-3','Bangbae-4',
              'Yangjae-1','Yangjae-2','Naegok']

# ── 데이터 ────────────────────────────────────────────────────────────────────
print("1. 데이터 로드 + XGBoost 학습 (R²≈0.79)")
df_raw = pd.read_csv(DATA_CSV)
feat_cols = [f'rate_{c}' for c in DONG_CODES]
X = df_raw[feat_cols].values
y = df_raw['total_runoff_m3'].values

imp_areas = (df_params['area_ha'].values * 10_000 *
             df_params['impervious_pct'].values / 100)
weighted_cov = (X * imp_areas).sum(axis=1) / imp_areas.sum()
X_full = np.column_stack([X, weighted_cov])
feature_names_full = DONG_EN + ['Weighted_Mean_Coverage']

X_tr, X_te, y_tr, y_te = train_test_split(X_full, y, test_size=0.2, random_state=42)
xgb_model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05,
                               max_depth=4, subsample=0.8,
                               random_state=42, verbosity=0)
xgb_model.fit(X_tr, y_tr)
print(f"   R²_test = {xgb_model.score(X_te, y_te):.4f}")

# ── SHAP ──────────────────────────────────────────────────────────────────────
print("2. SHAP 값 계산 중 (TreeExplainer)")
explainer = shap.TreeExplainer(xgb_model)
shap_vals = explainer.shap_values(X_full)   # [201, 19]
print(f"   shap_vals.shape = {shap_vals.shape}")

# 동별 18피처만 기여도 분석
mean_abs_shap = np.abs(shap_vals[:, :18]).mean(axis=0)

df_imp = pd.DataFrame({
    'Dong_KR'        : DONG_NAMES,
    'Dong_EN'        : DONG_EN,
    'ADM_CD'         : DONG_CODES,
    'mean_abs_SHAP'  : mean_abs_shap.round(1),
    'area_ha'        : df_params['area_ha'].values,
    'impervious_pct' : df_params['impervious_pct'].values,
    'avg_slope_deg'  : df_params['avg_slope_deg'].values,
    'imp_area_m2'    : imp_areas.round(0),
})
# 정규화: SHAP per impervious area, per total area
df_imp['shap_per_imp_area'] = (df_imp['mean_abs_SHAP'] /
                                (df_imp['imp_area_m2']/10000)).round(3)
df_imp['shap_per_total_area'] = (df_imp['mean_abs_SHAP'] /
                                   df_imp['area_ha']).round(3)
df_imp = df_imp.sort_values('mean_abs_SHAP', ascending=False)
df_imp.to_csv(SHAP_CSV, index=False, encoding='utf-8-sig')
print(f"   저장 → {SHAP_CSV}")

# ── 시각화 A: SHAP Summary Bar (XGBoost) ──────────────────────────────────────
print("3. 시각화 중...")
sorted_idx = np.argsort(mean_abs_shap)[::-1]
sorted_names = [DONG_EN[i] for i in sorted_idx]
sorted_vals  = mean_abs_shap[sorted_idx]

fig, ax = plt.subplots(figsize=(10, 7), facecolor='white')
ax.set_facecolor('white')
colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.9, len(sorted_vals)))
ax.barh(sorted_names[::-1], sorted_vals[::-1],
        color=colors[::-1], edgecolor='white', height=0.7)
for i, v in enumerate(sorted_vals[::-1]):
    ax.text(v + 5, i, f'{v:,.0f}', va='center', fontsize=10)
ax.set_xlabel('Mean |SHAP Value| (m³)')
ax.set_title('SHAP Feature Importance — XGBoost Surrogate (R²=0.79)\n'
             'Same model as used in NSGA-II Optimization',
             fontsize=13, fontweight='bold')
ax.grid(axis='x', linestyle=':', alpha=0.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig_shap_xgb_summary.png'), facecolor='white')
plt.close(fig)

# ── 시각화 B: SHAP Dependence Plots (Top-4 dongs) ─────────────────────────────
top4 = sorted_idx[:4]
fig2, axes2 = plt.subplots(2, 2, figsize=(13, 10), facecolor='white')
for ax, j in zip(axes2.ravel(), top4):
    ax.set_facecolor('white')
    sc = ax.scatter(X[:, j], shap_vals[:, j], c=weighted_cov, cmap='coolwarm',
                    s=30, alpha=0.7, edgecolors='none', vmin=0, vmax=100)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel(f'{DONG_EN[j]} Coverage Rate (%)')
    ax.set_ylabel(f'SHAP Value for {DONG_EN[j]} (m³)')
    ax.set_title(f'{DONG_EN[j]} (Imp={df_params.iloc[j]["impervious_pct"]:.1f}%, '
                 f'Area={df_params.iloc[j]["area_ha"]:.0f} ha)',
                 fontsize=11, fontweight='bold')
    ax.grid(linestyle=':', alpha=0.4)
    plt.colorbar(sc, ax=ax, label='Weighted Mean Cov. (%)', shrink=0.8)
fig2.suptitle('SHAP Dependence Plots — Top-4 Sub-districts (XGBoost)\n'
              'Color: Weighted Mean Coverage (interaction effect)',
              fontsize=14, fontweight='bold', y=1.00)
fig2.tight_layout()
fig2.savefig(os.path.join(FIG_DIR, 'fig_shap_xgb_dependence.png'), facecolor='white')
plt.close(fig2)

# ── 시각화 C: 정규화 SHAP 비교 (raw vs per-imp-area vs per-total-area) ──────
print("4. 정규화 SHAP 비교 그림 생성 중 (R2 #13 대응)")
fig3, axes3 = plt.subplots(1, 3, figsize=(17, 6), facecolor='white')

for ax, col, title, unit in zip(
        axes3,
        ['mean_abs_SHAP', 'shap_per_imp_area', 'shap_per_total_area'],
        ['(a) Raw SHAP\n(absolute leverage)',
         '(b) SHAP per Impervious Area\n(efficiency per LID-eligible ha)',
         '(c) SHAP per Total Area\n(intrinsic hydrological leverage)'],
        ['m³', 'm³ / ha-imp', 'm³ / ha']):
    ax.set_facecolor('white')
    df_sorted = df_imp.sort_values(col, ascending=False)
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.9, len(df_sorted)))
    ax.barh(df_sorted['Dong_EN'].values[::-1], df_sorted[col].values[::-1],
            color=colors[::-1], edgecolor='white', height=0.7)
    ax.set_xlabel(f'Importance ({unit})')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(axis='x', linestyle=':', alpha=0.4)

fig3.suptitle('Effect of Normalization on SHAP-based Sub-district Ranking\n'
              'Addresses Reviewer #2 Comment: "Large because large" vs. true leverage',
              fontsize=13, fontweight='bold', y=1.00)
fig3.tight_layout()
fig3.savefig(os.path.join(FIG_DIR, 'fig_shap_xgb_normalized.png'), facecolor='white')
plt.close(fig3)

# ── 시각화 D: 공간지도 (XGBoost-SHAP) ───────────────────────────────────────
print("5. 공간지도 생성 중...")
dong_gdf = gpd.read_file(BOUNDARY_SHP).to_crs('EPSG:5179')
dong_gdf['ADM_CD'] = dong_gdf['ADM_CD'].astype(str)
dong_gdf = dong_gdf.merge(
    df_imp[['ADM_CD','mean_abs_SHAP','shap_per_imp_area','shap_per_total_area']],
    on='ADM_CD', how='left')

fig4, axes4 = plt.subplots(1, 3, figsize=(17, 6), facecolor='white')
maps = [('mean_abs_SHAP',     'Reds',   'Raw SHAP (m³)'),
        ('shap_per_imp_area', 'Blues',  'SHAP per Impervious Area\n(m³ / ha-imp)'),
        ('shap_per_total_area','Greens','SHAP per Total Area\n(m³ / ha)')]
for ax, (col, cmap, lab) in zip(axes4, maps):
    dong_gdf.plot(column=col, ax=ax, cmap=cmap, edgecolor='gray',
                  linewidth=0.5, legend=True,
                  legend_kwds={'label': lab, 'shrink':0.7, 'pad':0.02})
    ax.set_title(lab.split('\n')[0], fontsize=12, fontweight='bold')
    ax.axis('off')
fig4.suptitle('Spatial Distribution of SHAP Importance under Different Normalizations\n'
              '(XGBoost surrogate, Seocho-gu)',
              fontsize=13, fontweight='bold', y=1.00)
fig4.tight_layout()
fig4.savefig(os.path.join(FIG_DIR, 'fig_shap_xgb_spatial.png'), facecolor='white')
plt.close(fig4)

# ── 요약 출력 ──────────────────────────────────────────────────────────────────
print("\n=== XGBoost-SHAP Top-5 (raw vs normalized) ===")
for col, lab in [('mean_abs_SHAP','Raw SHAP'),
                  ('shap_per_imp_area','per imp-area'),
                  ('shap_per_total_area','per total-area')]:
    print(f"\n[{lab}]")
    top5 = df_imp.sort_values(col, ascending=False).head(5)
    for _, r in top5.iterrows():
        print(f"  {r['Dong_EN']:13s} {r[col]:>10.2f}  "
              f"(imp={r['impervious_pct']:.1f}%, area={r['area_ha']:.0f} ha)")
