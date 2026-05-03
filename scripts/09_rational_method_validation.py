"""
09_rational_method_validation.py
V2 검증: 합리식(Rational Method)으로 SWMM Baseline 유출량 교차 검증
  - 토지피복지도 기반 복합 유출계수(C) 산출
  - 합리식 첨두유량 및 총유출량 추정
  - SWMM Baseline 결과와 비교
출력:
  - results/rational_method_validation.csv
  - results/figures/fig_rational_method_validation.png
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Patch

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
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANDUSE_SHP  = os.path.join(BASE_DIR, 'data/landuse/SG05_37608100_20251113/37608100.shp')
BOUNDARY_SHP = os.path.join(BASE_DIR, 'data/boundary/bnd_dong_11220_2025_2Q/bnd_dong_11220_2025_2Q.shp')
PARAMS_CSV   = os.path.join(BASE_DIR, 'results/subcatchment_params.csv')
LU_VAL_CSV   = os.path.join(BASE_DIR, 'results/landuse_validation.csv')
OUT_CSV      = os.path.join(BASE_DIR, 'results/rational_method_validation.csv')
FIG_DIR      = os.path.join(BASE_DIR, 'results/figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ── 설계강우 ───────────────────────────────────────────────────────────────────
RAINFALL_DEPTH_M  = 0.0676   # 서울 10년 빈도 60분 총강우량 (m)
RAINFALL_INTENS   = 67.6     # mm/hr (평균 강우강도 = 67.6mm / 1hr)
DURATION_HR       = 1.0      # 60분
BASELINE_SWMM_M3  = 614298.0 # SWMM Baseline 총유출량
BASELINE_PEAK_CMS = 85.84    # SWMM Baseline 첨두유량 (m³/s)

# ── 유출계수 (서울시 하수도설계기준, 환경부 토지피복 L1 분류 기준) ────────────
# 참고: 서울시 하수도정비기본계획(2021), 환경부 비점오염원관리지침
C_BY_L1 = {
    '100': 0.80,   # 시가화건조지역 (주거·상업·공업·교통 등) — 불투수 지배
    '200': 0.35,   # 농업지역
    '300': 0.20,   # 산림지역
    '400': 0.30,   # 초지
    '500': 0.15,   # 습지
    '600': 0.50,   # 나지
    '700': 0.95,   # 수역
}

# ── 1. 데이터 로드 ─────────────────────────────────────────────────────────────
print("1. 데이터 로드 중...")
dong   = gpd.read_file(BOUNDARY_SHP).to_crs(TARGET_CRS)
lu     = gpd.read_file(LANDUSE_SHP).to_crs(TARGET_CRS)
dong['ADM_CD'] = dong['ADM_CD'].astype(str)
df_params = pd.read_csv(PARAMS_CSV)
df_params['ADM_CD'] = df_params['ADM_CD'].astype(str)

# LU 검증 결과가 있으면 재사용, 없으면 계산
if os.path.exists(LU_VAL_CSV):
    df_lu_val = pd.read_csv(LU_VAL_CSV)
    df_lu_val['ADM_CD'] = df_lu_val['ADM_CD'].astype(str)
    print("   토지피복 검증 데이터 재사용 (08_landuse_validation.py 결과)")
else:
    df_lu_val = None
    print("   토지피복 검증 데이터 없음 — 직접 계산합니다")

# ── 2. 동별 복합 유출계수 계산 ─────────────────────────────────────────────────
print("2. 동별 복합 유출계수(C) 계산 중...")
records = []

for _, row in dong.iterrows():
    dong_name = row['ADM_NM']
    dong_code = row['ADM_CD']
    geom      = row.geometry
    dong_area = geom.area   # m²

    # 각 L1 클래스별 면적 계산
    c_weighted_sum = 0.0
    area_by_class  = {}
    for l1_code, c_val in C_BY_L1.items():
        clip_lu   = lu[lu['L1_CODE'] == l1_code]
        clip_lu   = clip_lu[clip_lu.geometry.intersects(geom)]
        if len(clip_lu) > 0:
            cls_area = clip_lu.geometry.intersection(geom).area.sum()
        else:
            cls_area = 0.0
        area_by_class[l1_code] = cls_area
        c_weighted_sum += c_val * cls_area

    # 분류되지 않은 면적 처리 (잔여 면적 → C=0.30 기본값)
    classified_area = sum(area_by_class.values())
    residual_area   = max(0.0, dong_area - classified_area)
    c_weighted_sum += 0.30 * residual_area

    C_composite = c_weighted_sum / dong_area

    # 합리식 첨두유량: Q(m³/s) = C * I(mm/hr) * A(ha) / 360
    area_ha  = dong_area / 10_000
    Q_peak   = C_composite * RAINFALL_INTENS * area_ha / 360  # m³/s

    # 합리식 총유출량: V(m³) = C * P(m) * A(m²)
    V_total  = C_composite * RAINFALL_DEPTH_M * dong_area     # m³

    # SWMM 파라미터 비교용
    osm_row  = df_params[df_params['ADM_CD'] == dong_code]
    osm_imp  = float(osm_row['impervious_pct'].values[0]) if len(osm_row) else np.nan

    # 토지피복 불투수율
    if df_lu_val is not None:
        lu_row  = df_lu_val[df_lu_val['ADM_CD'] == dong_code]
        lu_imp  = float(lu_row['lu_imp_pct'].values[0]) if len(lu_row) else np.nan
    else:
        lu_imp  = area_by_class.get('100', 0.0) / dong_area * 100

    records.append({
        'ADM_CD'        : dong_code,
        'ADM_NM'        : dong_name,
        'area_ha'       : round(area_ha, 2),
        'C_composite'   : round(C_composite, 3),
        'Q_peak_cms'    : round(Q_peak, 3),
        'V_runoff_m3'   : round(V_total, 1),
        'lu_imp_pct'    : round(lu_imp, 1) if lu_imp is not None else np.nan,
        'osm_imp_pct'   : osm_imp,
    })

    print(f"   {dong_name:8s} | C={C_composite:.3f}  "
          f"Q_peak={Q_peak:.2f} m³/s  V={V_total:,.0f} m³")

df_rm = pd.DataFrame(records)
df_rm.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')

# ── 3. 지구 전체 집계 및 SWMM 비교 ────────────────────────────────────────────
total_area_ha = df_rm['area_ha'].sum()
total_V_rm    = df_rm['V_runoff_m3'].sum()
total_Q_rm    = df_rm['Q_peak_cms'].sum()   # 단순 합 (동시 첨두 가정)
C_district    = total_V_rm / (RAINFALL_DEPTH_M * total_area_ha * 10_000)

vol_diff_pct  = (total_V_rm - BASELINE_SWMM_M3) / BASELINE_SWMM_M3 * 100
peak_diff_pct = (total_Q_rm - BASELINE_PEAK_CMS) / BASELINE_PEAK_CMS * 100

print(f"\n=== 합리식 vs SWMM Baseline 비교 ===")
print(f"  서초구 전체 면적       : {total_area_ha:,.1f} ha")
print(f"  복합 유출계수 (C)      : {C_district:.3f}")
print(f"")
print(f"  {'항목':<22} {'합리식':>12} {'SWMM':>12} {'차이(%)':>10}")
print(f"  {'-'*58}")
print(f"  {'총유출량 (m³)':<22} {total_V_rm:>12,.0f} "
      f"{BASELINE_SWMM_M3:>12,.0f} {vol_diff_pct:>+10.1f}%")
print(f"  {'첨두유량 (m³/s)':<22} {total_Q_rm:>12.1f} "
      f"{BASELINE_PEAK_CMS:>12.2f} {peak_diff_pct:>+10.1f}%")

# ── 4. 시각화 ─────────────────────────────────────────────────────────────────
print("\n3. 시각화 중...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
for ax in axes:
    ax.set_facecolor('white')

short_names = [n.replace('동','') for n in df_rm['ADM_NM']]
x = np.arange(len(df_rm))

# ── Panel (a): 동별 유출계수 ────────────────────────────────────────────────
ax1 = axes[0]
bars = ax1.bar(x, df_rm['C_composite'], color=BLUE, alpha=0.8,
               edgecolor='white', linewidth=0.5)
# 전체 복합 C 기준선
ax1.axhline(C_district, color=ORANGE, linestyle='--', linewidth=1.8,
            label=f'District C = {C_district:.3f}')
ax1.set_xticks(x)
ax1.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
ax1.set_ylabel('Composite Runoff Coefficient (C)', fontsize=12)
ax1.set_ylim(0, 0.6)
ax1.set_title('(a) Composite Runoff Coefficient\nper Sub-district (Rational Method)',
              fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(axis='y', linestyle=':', alpha=0.4)

# ── Panel (b): 유출량 비교 ──────────────────────────────────────────────────
ax2 = axes[1]
categories = ['Total Runoff\nVolume (m³)', 'Peak Discharge\n(m³/s)']
rm_vals   = [total_V_rm,    total_Q_rm]
swmm_vals = [BASELINE_SWMM_M3, BASELINE_PEAK_CMS]
# 스케일 통일을 위해 정규화 (SWMM=1.0 기준)
norm_rm   = [r/s for r, s in zip(rm_vals, swmm_vals)]
norm_swmm = [1.0, 1.0]

xb = np.arange(2)
w  = 0.32
ax2.bar(xb - w/2, norm_swmm, width=w, color=ORANGE, alpha=0.85,
        label=f'SWMM Baseline\n({BASELINE_SWMM_M3:,.0f} m³ / {BASELINE_PEAK_CMS:.1f} m³/s)')
ax2.bar(xb + w/2, norm_rm,   width=w, color=BLUE,   alpha=0.85,
        label=f'Rational Method\n({total_V_rm:,.0f} m³ / {total_Q_rm:.1f} m³/s)')

# 값 표시
ylim_top = 1.5
for i, (ns, nr) in enumerate(zip(norm_swmm, norm_rm)):
    ax2.text(i - w/2, min(ns, ylim_top - 0.12) + 0.02, '1.00',
             ha='center', fontsize=10, fontweight='bold')
    ax2.text(i + w/2, min(nr, ylim_top - 0.12) + 0.02, f'{nr:.2f}',
             ha='center', fontsize=10, fontweight='bold',
             color=GREEN if abs(nr-1) < 0.1 else ORANGE)

ax2.axhline(1.0, color=GRAY, linestyle='--', linewidth=1.2, alpha=0.6)
ax2.set_xticks(xb)
ax2.set_xticklabels(categories, fontsize=12)
ax2.set_ylabel('Normalized Value (SWMM = 1.0)', fontsize=12)
ax2.set_ylim(0, 1.5)
ax2.set_title('(b) Rational Method vs. SWMM\n(Normalized by SWMM Baseline)',
              fontsize=12, fontweight='bold')
ax2.legend(fontsize=9, loc='upper right')
ax2.grid(axis='y', linestyle=':', alpha=0.4)

fig.suptitle('Rational Method Cross-Validation of SWMM Baseline Results\n'
             'Seocho-gu, Seoul (10-yr Design Storm, 67.6 mm / 60 min)',
             fontsize=13, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.88])
fig.savefig(os.path.join(FIG_DIR, 'fig_rational_method_validation.png'), facecolor='white')
plt.close(fig)
print("   fig_rational_method_validation.png 저장 완료")

# ── 5. 논문용 요약 출력 ────────────────────────────────────────────────────────
print(f"\n=== 논문 서술용 요약 ===")
print(f"  설계강우 : 10년 빈도 60분, 총강우량 {RAINFALL_DEPTH_M*1000:.1f} mm")
print(f"  서초구 복합 C : {C_district:.3f}")
print(f"  합리식 총유출량 : {total_V_rm:,.0f} m³  (SWMM 대비 {vol_diff_pct:+.1f}%)")
print(f"  합리식 첨두유량 : {total_Q_rm:.1f} m³/s  (SWMM 대비 {peak_diff_pct:+.1f}%)")
print(f"\n저장: {OUT_CSV}")
