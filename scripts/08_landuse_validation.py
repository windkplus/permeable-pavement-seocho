"""
08_landuse_validation.py
V1 검증: OSM 기반 불투수율 vs 환경부 토지피복지도 비교
  - 토지피복지도(SG05) L1_CODE=100(시가화건조지역)을 불투수면으로 분류
  - 행정동별 불투수율 산출 후 OSM 추정값과 비교
출력:
  - results/landuse_validation.csv
  - results/figures/fig_landuse_validation.png
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

matplotlib.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'font.size'        : 13,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'figure.dpi'       : 150,
    'savefig.dpi'      : 300,
    'savefig.bbox'     : 'tight',
})

BLUE   = '#2166AC'
ORANGE = '#D6604D'
GREEN  = '#4DAC26'
GRAY   = '#888888'
TARGET_CRS = 'EPSG:5179'

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANDUSE_SHP   = os.path.join(BASE_DIR, 'data/landuse/SG05_37608100_20251113/37608100.shp')
BOUNDARY_SHP  = os.path.join(BASE_DIR, 'data/boundary/bnd_dong_11220_2025_2Q/bnd_dong_11220_2025_2Q.shp')
PARAMS_CSV    = os.path.join(BASE_DIR, 'results/subcatchment_params.csv')
OUT_CSV       = os.path.join(BASE_DIR, 'results/landuse_validation.csv')
FIG_DIR       = os.path.join(BASE_DIR, 'results/figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ── 1. 데이터 로드 ─────────────────────────────────────────────────────────────
print("1. 데이터 로드 중...")
dong     = gpd.read_file(BOUNDARY_SHP).to_crs(TARGET_CRS)
lu       = gpd.read_file(LANDUSE_SHP).to_crs(TARGET_CRS)
df_osm   = pd.read_csv(PARAMS_CSV)
df_osm['ADM_CD'] = df_osm['ADM_CD'].astype(str)
dong['ADM_CD']   = dong['ADM_CD'].astype(str)

print(f"   행정동 {len(dong)}개  |  토지피복 폴리곤 {len(lu)}개")

# 불투수 클래스: L1_CODE='100' (시가화건조지역)
# L2 서브 분류 참고용 (L2_CODE: 110=주거, 120=상업, 130=공업, 140=문화체육,
#                                150=교통, 160=공공시설)
lu_imp = lu[lu['L1_CODE'] == '100'].copy()
print(f"   불투수면 폴리곤 (L1=100): {len(lu_imp)}개")

# ── 2. 동별 토지피복 불투수율 계산 ────────────────────────────────────────────
# 토지피복 타일이 서초구 전체를 커버하지 않으므로:
#   - 각 동과 타일의 교차 영역(overlap)만을 분모로 사용
#   - overlap 비율 < 5%인 동은 제외 (통계적 신뢰성 부족)
print("2. 동별 불투수율 계산 중 (부분 커버리지 보정)...")
lu_union = lu.union_all()   # 전체 토지피복 타일의 유니온 경계
records  = []
MIN_OVERLAP_PCT = 5.0   # 최소 중첩 비율 (%)

for _, row in dong.iterrows():
    dong_name = row['ADM_NM']
    dong_code = row['ADM_CD']
    geom      = row.geometry
    dong_area = geom.area   # m²

    # 타일과 동의 교차 영역
    overlap_geom = geom.intersection(lu_union)
    overlap_area = overlap_geom.area
    overlap_pct  = overlap_area / dong_area * 100

    # OSM 불투수율
    osm_row     = df_osm[df_osm['ADM_CD'] == dong_code]
    osm_imp_pct = float(osm_row['impervious_pct'].values[0]) if len(osm_row) else np.nan

    if overlap_pct < MIN_OVERLAP_PCT:
        print(f"   {dong_name:8s} | 타일 중첩 {overlap_pct:.1f}% → 제외")
        records.append({
            'ADM_CD'        : dong_code,
            'ADM_NM'        : dong_name,
            'area_ha'       : round(dong_area / 10_000, 2),
            'overlap_pct'   : round(overlap_pct, 1),
            'lu_imp_pct'    : np.nan,
            'osm_imp_pct'   : osm_imp_pct,
            'diff_pct'      : np.nan,
        })
        continue

    # 불투수 폴리곤과 교차 (분모 = 중첩 영역)
    imp_clip = lu_imp[lu_imp.geometry.intersects(overlap_geom)].copy()
    if len(imp_clip) == 0:
        imp_area = 0.0
    else:
        imp_area = imp_clip.geometry.intersection(overlap_geom).area.sum()

    lu_imp_pct = imp_area / overlap_area * 100

    # OSM 불투수율도 중첩 영역 기준으로 재계산 (건물·도로 교차)
    # (간략화: OSM 전체 동 값을 대표값으로 사용)
    diff = lu_imp_pct - osm_imp_pct

    records.append({
        'ADM_CD'        : dong_code,
        'ADM_NM'        : dong_name,
        'area_ha'       : round(dong_area / 10_000, 2),
        'overlap_pct'   : round(overlap_pct, 1),
        'lu_imp_pct'    : round(lu_imp_pct, 1),
        'osm_imp_pct'   : osm_imp_pct,
        'diff_pct'      : round(diff, 1),
    })
    print(f"   {dong_name:8s} | 중첩={overlap_pct:5.1f}%  "
          f"토지피복={lu_imp_pct:5.1f}%  OSM={osm_imp_pct:5.1f}%  "
          f"차이={diff:+.1f}%")

df_val = pd.DataFrame(records)
df_val.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')

n_valid = df_val['lu_imp_pct'].notna().sum()
n_skip  = df_val['lu_imp_pct'].isna().sum()
print(f"\n   유효 비교 동: {n_valid}개 (타일 중첩 ≥ {MIN_OVERLAP_PCT}%)")
print(f"   제외된 동  : {n_skip}개 (중첩 부족)")

# ── 3. 오차 통계 ───────────────────────────────────────────────────────────────
valid = df_val.dropna(subset=['lu_imp_pct', 'osm_imp_pct'])
rmse = np.sqrt(mean_squared_error(valid['lu_imp_pct'], valid['osm_imp_pct']))
mae  = mean_absolute_error(valid['lu_imp_pct'], valid['osm_imp_pct'])
bias = (valid['osm_imp_pct'] - valid['lu_imp_pct']).mean()

print(f"\n=== OSM vs 토지피복지도 비교 (유효 {len(valid)}개 동) ===")
print(f"  RMSE : {rmse:.2f} %p")
print(f"  MAE  : {mae:.2f} %p")
print(f"  Bias : {bias:+.2f} %p  ({'OSM 과소' if bias<0 else 'OSM 과대'})")
print(f"  LU 평균  : {valid['lu_imp_pct'].mean():.1f}%")
print(f"  OSM 평균 : {valid['osm_imp_pct'].mean():.1f}%")
print(f"  ※ 토지피복 타일이 서초구 일부만 커버하여 3개 동만 비교 가능")

# ── 4. 시각화 ─────────────────────────────────────────────────────────────────
print("\n3. 시각화 중...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
for ax in axes:
    ax.set_facecolor('white')

# ── Panel (a): Scatter ──────────────────────────────────────────────────────
ax = axes[0]
ax.scatter(valid['lu_imp_pct'], valid['osm_imp_pct'],
           color=BLUE, s=80, edgecolors='white', linewidth=1.2,
           zorder=3, label='Sub-districts')

for _, r in valid.iterrows():
    ax.annotate(r['ADM_NM'].replace('동',''),
                xy=(r['lu_imp_pct'], r['osm_imp_pct']),
                xytext=(3, 3), textcoords='offset points',
                fontsize=8, color='#444444')

lim_max = max(valid['lu_imp_pct'].max(), valid['osm_imp_pct'].max()) * 1.1
lim = [0, lim_max]
ax.plot(lim, lim, 'k--', linewidth=1.5, alpha=0.6, label='1:1 line')
ax.fill_between(lim, [v*0.8 for v in lim], [v*1.2 for v in lim],
                alpha=0.07, color=GRAY, label='±20% band')

ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('Land Use Map Impervious Fraction (%)\n(MOE SG05, 2024)', fontsize=12)
ax.set_ylabel('OSM-Derived Impervious Fraction (%)\n(This Study)', fontsize=12)
ax.set_title(f'(a) OSM vs. Land Use Map\nImpervious Fraction per Sub-district (n={len(valid)})\n'
             f'RMSE={rmse:.1f}%p  Bias={bias:+.1f}%p',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(linestyle=':', alpha=0.4)

# ── Panel (b): Bar 비교 ─────────────────────────────────────────────────────
ax2 = axes[1]
x   = np.arange(len(valid))
w   = 0.38
ax2.bar(x - w/2, valid['lu_imp_pct'], width=w,
        color=ORANGE, alpha=0.85, label='Land Use Map (SG05)')
ax2.bar(x + w/2, valid['osm_imp_pct'], width=w,
        color=BLUE,   alpha=0.85, label='OSM-Derived (This Study)')

short_names = [n.replace('동','') for n in valid['ADM_NM']]
ax2.set_xticks(x)
ax2.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
ax2.set_ylabel('Impervious Fraction (%)', fontsize=12)
ax2.set_title('(b) Sub-district Comparison:\nLand Use Map vs. OSM',
              fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(axis='y', linestyle=':', alpha=0.4)

fig.suptitle('Validation of OSM-Derived Impervious Fraction\nagainst Ministry of Environment Land Use Map (SG05)',
             fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig_landuse_validation.png'), facecolor='white')
plt.close(fig)
print("   fig_landuse_validation.png 저장 완료")

# ── 5. 논문용 표 출력 ─────────────────────────────────────────────────────────
print(f"\n=== 논문 Table 용 비교 결과 ===")
print(f"{'Sub-district':<18} {'LU Map (%)':>12} {'OSM (%)':>10} {'Diff (%p)':>10}")
print("-" * 55)
for _, r in df_val.iterrows():
    print(f"  {r['ADM_NM']:<16} {r['lu_imp_pct']:>12.1f} "
          f"{r['osm_imp_pct']:>10.1f} {r['diff_pct']:>+10.1f}")
print("-" * 55)
print(f"  {'Mean':<16} {valid['lu_imp_pct'].mean():>12.1f} "
      f"{valid['osm_imp_pct'].mean():>10.1f} {bias:>+10.2f}")
print(f"\n  RMSE={rmse:.2f}%p  MAE={mae:.2f}%p  Bias={bias:+.2f}%p  (n={len(valid)})")
print(f"\n저장: {OUT_CSV}")
