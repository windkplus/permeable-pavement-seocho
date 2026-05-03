"""
07_shap_analysis.py
SHAP 기반 설명가능 AI 민감도 분석
  - 어느 동의 투수성 포장이 유출저감에 가장 기여하는가?
  - 동 특성(면적·불투수율·경사)과 적용률의 기여도 분석
출력:
  - results/shap_feature_importance.csv
  - results/figures/fig_shap_summary.png
  - results/figures/fig_shap_spatial.png
  - results/figures/fig_shap_interaction.png
"""

import os, joblib, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

matplotlib.rcParams.update({'font.family':'DejaVu Sans','font.size':13,
                            'axes.spines.top':False,'axes.spines.right':False})

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV     = os.path.join(BASE_DIR, 'results/surrogate_training_data.csv')
PARAMS_CSV   = os.path.join(BASE_DIR, 'results/subcatchment_params.csv')
BOUNDARY_SHP = os.path.join(BASE_DIR, 'data/boundary/bnd_dong_11220_2025_2Q/bnd_dong_11220_2025_2Q.shp')
FIG_DIR      = os.path.join(BASE_DIR, 'results/figures')
SHAP_CSV     = os.path.join(BASE_DIR, 'results/shap_feature_importance.csv')
os.makedirs(FIG_DIR, exist_ok=True)

df_params = pd.read_csv(PARAMS_CSV)
df_params['ADM_CD'] = df_params['ADM_CD'].astype(str)
DONG_CODES  = df_params['ADM_CD'].tolist()
DONG_NAMES  = df_params['ADM_NM'].tolist()

# ── 데이터 로드 ────────────────────────────────────────────────────────────────
print("1. 데이터 로드 중...")
df_raw = pd.read_csv(DATA_CSV)
feat_cols = [f'rate_{c}' for c in DONG_CODES]
X = df_raw[feat_cols].values
y = df_raw['total_runoff_m3'].values
BASELINE = y[0]

# 영문 동 이름 (지도·그래프용)
DONG_EN = [
    'Seocho-1', 'Seocho-2', 'Seocho-3', 'Seocho-4',
    'Jamwon', 'Banpo-bon', 'Banpo-1', 'Banpo-2',
    'Banpo-3', 'Banpo-4', 'Bangbae-bon',
    'Bangbae-1', 'Bangbae-2', 'Bangbae-3', 'Bangbae-4',
    'Yangjae-1', 'Yangjae-2', 'Naegok'
]

# ── 모델 재학습 (SHAP용 Random Forest) ────────────────────────────────────────
print("2. Random Forest 학습 (SHAP 계산용)...")
imp_areas    = (df_params['area_ha'].values * 10_000 *
                df_params['impervious_pct'].values / 100)
weighted_cov = (X * imp_areas).sum(axis=1) / imp_areas.sum()
X_full = np.column_stack([X, weighted_cov])   # [n_samples, 19]

X_tr, X_te, y_tr, y_te = train_test_split(X_full, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)
print(f"   R² (test) = {rf.score(X_te, y_te):.4f}")

# ── SHAP 값 계산 ───────────────────────────────────────────────────────────────
print("3. SHAP 값 계산 중...")
explainer  = shap.TreeExplainer(rf)
shap_vals  = explainer.shap_values(X_full)    # [n_samples, 19]

# 동별 18 피처의 평균 |SHAP| (weighted_cov 제외)
mean_abs_shap = np.abs(shap_vals[:, :18]).mean(axis=0)

df_importance = pd.DataFrame({
    'Dong_KR'       : DONG_NAMES,
    'Dong_EN'       : DONG_EN,
    'ADM_CD'        : DONG_CODES,
    'mean_abs_SHAP' : mean_abs_shap.round(1),
    'area_ha'       : df_params['area_ha'].values,
    'impervious_pct': df_params['impervious_pct'].values,
    'avg_slope_deg' : df_params['avg_slope_deg'].values,
})
df_importance = df_importance.sort_values('mean_abs_SHAP', ascending=False)
df_importance.to_csv(SHAP_CSV, index=False, encoding='utf-8-sig')
print(f"   저장 → {SHAP_CSV}")

# ── 시각화 A: SHAP Summary Bar ────────────────────────────────────────────────
print("4. 시각화 중...")
sorted_idx = np.argsort(mean_abs_shap)[::-1]   # 내림차순
sorted_names = [DONG_EN[i] for i in sorted_idx]
sorted_shap  = mean_abs_shap[sorted_idx]

fig1, ax1 = plt.subplots(figsize=(10, 7), facecolor='white')
ax1.set_facecolor('white')

colors_bar = plt.cm.RdYlBu_r(np.linspace(0.2, 0.9, len(sorted_shap)))
bars = ax1.barh(sorted_names[::-1], sorted_shap[::-1],
                color=colors_bar[::-1], edgecolor='white', height=0.7)

for bar, val in zip(bars, sorted_shap[::-1]):
    ax1.text(val + 5, bar.get_y() + bar.get_height()/2,
             f'{val:,.0f}', va='center', fontsize=11)

ax1.set_xlabel('Mean |SHAP Value| (Impact on Runoff Prediction, m³)', fontsize=13)
ax1.set_title('SHAP Feature Importance: Contribution of Each\n'
              'Sub-district Coverage Rate to Runoff Reduction',
              fontsize=14, fontweight='bold')
ax1.grid(axis='x', linestyle=':', alpha=0.5)
fig1.tight_layout()
fig1.savefig(os.path.join(FIG_DIR, 'fig_shap_summary.png'),
             dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig1)

# ── 시각화 B: SHAP Beeswarm (상위 10개 동) ───────────────────────────────────
top10_idx   = sorted_idx[:10]
X_top10     = X[:, top10_idx]                  # 원본 18 피처에서 인덱싱
shap_top10  = shap_vals[:, top10_idx]          # 동별 SHAP (인덱스 0-17)
names_top10 = [DONG_EN[i] for i in top10_idx]

fig2, ax2 = plt.subplots(figsize=(11, 7), facecolor='white')
ax2.set_facecolor('white')

# 수동 beeswarm (shap.summary_plot 대신)
for j in range(len(top10_idx)-1, -1, -1):
    vals = shap_top10[:, j]
    feat = X_top10[:, j]
    y_pos = np.full(len(vals), j)
    sc = ax2.scatter(vals, y_pos, c=feat, cmap='coolwarm',
                     s=20, alpha=0.6, vmin=0, vmax=100)

cbar = plt.colorbar(sc, ax=ax2, pad=0.02)
cbar.set_label('Coverage Rate (%)', fontsize=11)
ax2.set_yticks(range(len(names_top10)))
ax2.set_yticklabels(names_top10[::-1], fontsize=12)
ax2.axvline(0, color='black', linewidth=0.8)
ax2.set_xlabel('SHAP Value (Impact on Predicted Runoff, m³)', fontsize=13)
ax2.set_title('SHAP Beeswarm Plot: Top-10 Sub-districts\n'
              '(Color = Coverage Rate; Right = Increases Runoff)',
              fontsize=14, fontweight='bold')
ax2.grid(axis='x', linestyle=':', alpha=0.4)
fig2.tight_layout()
fig2.savefig(os.path.join(FIG_DIR, 'fig_shap_beeswarm.png'),
             dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig2)

# ── 시각화 C: SHAP 공간 분포 지도 ────────────────────────────────────────────
dong_gdf = gpd.read_file(BOUNDARY_SHP).to_crs('EPSG:5179')
dong_gdf['ADM_CD'] = dong_gdf['ADM_CD'].astype(str)
dong_gdf = dong_gdf.merge(
    df_importance[['ADM_CD','mean_abs_SHAP','impervious_pct']],
    on='ADM_CD', how='left')

fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')

# 좌: SHAP 중요도 지도
dong_gdf.plot(column='mean_abs_SHAP', ax=axes3[0], cmap='Reds',
              edgecolor='gray', linewidth=0.5, legend=True,
              legend_kwds={'label':'Mean |SHAP| (m³)',
                           'shrink':0.7, 'pad':0.02})
axes3[0].set_title('(a) SHAP Feature Importance\n(Spatial Distribution)',
                   fontsize=13, fontweight='bold')
axes3[0].axis('off')

# 우: 불투수율 지도 (비교)
dong_gdf.plot(column='impervious_pct', ax=axes3[1], cmap='YlOrBr',
              edgecolor='gray', linewidth=0.5, legend=True,
              legend_kwds={'label':'Impervious Fraction (%)',
                           'shrink':0.7, 'pad':0.02})
axes3[1].set_title('(b) Impervious Surface Fraction\n(Sub-district Level)',
                   fontsize=13, fontweight='bold')
axes3[1].axis('off')

fig3.suptitle('Spatial Comparison: SHAP Importance vs. Impervious Fraction\n'
              '(Seocho-gu, Seoul)',
              fontsize=14, fontweight='bold', y=1.02)
fig3.savefig(os.path.join(FIG_DIR, 'fig_shap_spatial.png'),
             dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig3)

# ── 시각화 D: 동 특성 vs SHAP 산점도 ────────────────────────────────────────
fig4, axes4 = plt.subplots(1, 3, figsize=(16, 5), facecolor='white')
props  = ['area_ha', 'impervious_pct', 'avg_slope_deg']
xlabels= ['Sub-district Area (ha)',
           'Impervious Fraction (%)',
           'Mean Slope (°)']
for ax4, prop, xlabel in zip(axes4, props, xlabels):
    ax4.set_facecolor('white')
    ax4.scatter(df_importance[prop], df_importance['mean_abs_SHAP'],
                s=100, color='#2166AC', edgecolors='white', zorder=3)
    for _, row in df_importance.iterrows():
        ax4.annotate(row['Dong_EN'],
                     (row[prop], row['mean_abs_SHAP']),
                     fontsize=8, ha='left', va='bottom',
                     xytext=(3, 2), textcoords='offset points', color='#555')
    # 추세선
    z = np.polyfit(df_importance[prop], df_importance['mean_abs_SHAP'], 1)
    p = np.poly1d(z)
    xs = np.linspace(df_importance[prop].min(), df_importance[prop].max(), 50)
    ax4.plot(xs, p(xs), '--', color='#D6604D', linewidth=1.5, alpha=0.8)

    corr = df_importance[prop].corr(df_importance['mean_abs_SHAP'])
    ax4.set_xlabel(xlabel, fontsize=12)
    ax4.set_ylabel('Mean |SHAP Value| (m³)', fontsize=12)
    ax4.set_title(f'r = {corr:.2f}', fontsize=13, fontweight='bold')
    ax4.grid(linestyle=':', alpha=0.4)

fig4.suptitle('Correlation Between Sub-district Physical Characteristics\n'
              'and SHAP Feature Importance',
              fontsize=14, fontweight='bold')
fig4.tight_layout()
fig4.savefig(os.path.join(FIG_DIR, 'fig_shap_interaction.png'),
             dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig4)

# ── 결과 요약 ─────────────────────────────────────────────────────────────────
print("\n=== SHAP 분석 완료 ===")
print("\nTop-5 중요 동 (유출저감 효과 큰 순):")
for _, row in df_importance.head(5).iterrows():
    print(f"  {row['Dong_EN']:15s} | SHAP={row['mean_abs_SHAP']:,.0f} m³ | "
          f"Imp={row['impervious_pct']:.1f}% | Area={row['area_ha']:.0f}ha")

print(f"\n저장 파일:")
print(f"  {SHAP_CSV}")
print(f"  results/figures/fig_shap_summary.png")
print(f"  results/figures/fig_shap_beeswarm.png")
print(f"  results/figures/fig_shap_spatial.png")
print(f"  results/figures/fig_shap_interaction.png")
