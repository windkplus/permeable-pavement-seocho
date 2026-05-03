"""
00_study_area_map.py
연구 대상지 위치도 생성
  - Panel (a): 서울시 전체 + 서초구 강조 (contextily 배경)
  - Panel (b): 서초구 행정동 상세도 (18개 동 레이블)
출력:
  - results/figures/fig_study_area.png
"""

import os
import numpy as np
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import contextily as ctx
import osmnx as ox
from shapely.geometry import box

matplotlib.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'font.size'        : 11,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'figure.dpi'       : 150,
    'savefig.dpi'      : 300,
    'savefig.bbox'     : 'tight',
})

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DONG_SHP     = os.path.join(BASE_DIR, 'data/boundary/bnd_dong_11220_2025_2Q/bnd_dong_11220_2025_2Q.shp')
SIGUNGU_SHP  = os.path.join(BASE_DIR, 'data/boundary/bnd_sigungu_11220_2025_2Q/bnd_sigungu_11220_2025_2Q.shp')
PARAMS_CSV   = os.path.join(BASE_DIR, 'results/subcatchment_params.csv')
FIG_DIR      = os.path.join(BASE_DIR, 'results/figures')
os.makedirs(FIG_DIR, exist_ok=True)

WGS84  = 'EPSG:4326'
WEB    = 'EPSG:3857'   # contextily basemap CRS
KR_TM  = 'EPSG:5179'  # 국내 표준 TM

BLUE_DARK  = '#08519C'
BLUE_MID   = '#2171B5'
BLUE_LIGHT = '#BDD7E7'
ORANGE     = '#D94801'
GRAY_FILL  = '#F0F0F0'
GRAY_BORD  = '#AAAAAA'

# ── 데이터 로드 ────────────────────────────────────────────────────────────────
print("1. 데이터 로드 중...")
dong    = gpd.read_file(DONG_SHP).to_crs(KR_TM)
seocho  = gpd.read_file(SIGUNGU_SHP).to_crs(KR_TM)

# OSMnx로 서울시 경계
print("2. 서울시 경계 다운로드 중...")
seoul_wgs = ox.geocode_to_gdf('Seoul, South Korea')
seoul_km  = seoul_wgs.to_crs(KR_TM)

# 서초구 중심점
seocho_centroid = seocho.geometry.centroid.values[0]

# 임시: 동별 불투수율 색상용
import pandas as pd
try:
    df_params = pd.read_csv(PARAMS_CSV)
    df_params['ADM_CD'] = df_params['ADM_CD'].astype(str)
    dong['ADM_CD'] = dong['ADM_CD'].astype(str)
    dong = dong.merge(df_params[['ADM_CD', 'impervious_pct']], on='ADM_CD', how='left')
    use_imp_color = True
except Exception:
    use_imp_color = False

# ── 레이블용 동 이름 정리 ──────────────────────────────────────────────────────
KR_TO_EN = {
    '서초1동':  'Seocho 1',
    '서초2동':  'Seocho 2',
    '서초3동':  'Seocho 3',
    '서초4동':  'Seocho 4',
    '잠원동':   'Jamwon',
    '반포본동': 'Banpo',
    '반포1동':  'Banpo 1',
    '반포2동':  'Banpo 2',
    '반포3동':  'Banpo 3',
    '반포4동':  'Banpo 4',
    '방배본동': 'Bangbae',
    '방배1동':  'Bangbae 1',
    '방배2동':  'Bangbae 2',
    '방배3동':  'Bangbae 3',
    '방배4동':  'Bangbae 4',
    '양재1동':  'Yangjae 1',
    '양재2동':  'Yangjae 2',
    '내곡동':   'Naegok',
}

def shorten_name(nm):
    return KR_TO_EN.get(nm, nm.replace('동', ''))

dong['label'] = dong['ADM_NM'].apply(shorten_name)

# ── 그림 구성 ──────────────────────────────────────────────────────────────────
print("3. 그림 생성 중...")
fig = plt.figure(figsize=(16, 8), facecolor='white')

# 왼쪽 여백(0.00~0.11)에 인셋 공간 확보, ax1은 0.12부터 시작
ax1 = fig.add_axes([0.12, 0.04, 0.35, 0.84])
ax2 = fig.add_axes([0.52, 0.04, 0.46, 0.84])

for ax in (ax1, ax2):
    ax.set_facecolor('white')

# ════════════════════════════════════════════════════════════════════════════════
# Panel (a): 서울시 + 서초구 강조
# ════════════════════════════════════════════════════════════════════════════════
ax1.set_aspect('equal')

# 서울시 전체 (연회색)
seoul_km.plot(ax=ax1, facecolor=GRAY_FILL, edgecolor='#888888',
              linewidth=0.8, zorder=1)

# 서초구 강조 (진한 파란색 채우기)
seocho.plot(ax=ax1, facecolor=BLUE_MID, edgecolor='white',
            linewidth=1.5, zorder=2, alpha=0.85)

# 서초구 경계 추가 강조
seocho.boundary.plot(ax=ax1, color=BLUE_DARK, linewidth=2.0, zorder=3)

# 서초구 레이블
ax1.annotate('Seocho-gu',
             xy=(seocho_centroid.x, seocho_centroid.y),
             xytext=(seocho_centroid.x + 8000, seocho_centroid.y + 5000),
             fontsize=11, fontweight='bold', color=BLUE_DARK,
             arrowprops=dict(arrowstyle='->', color=BLUE_DARK, lw=1.5),
             zorder=5)

# 한강 대략 위치 표시 (선)
# 서울시 bbox 기준 참고용 텍스트
minx, miny, maxx, maxy = seoul_km.total_bounds
ax1.text(minx + (maxx - minx) * 0.55, miny + (maxy - miny) * 0.52,
         'Han River', fontsize=8.5, color='#3a86b0', style='italic', zorder=4)

# 나침반
ax1.annotate('N', xy=(0.94, 0.92), xycoords='axes fraction',
             fontsize=13, fontweight='bold', ha='center', va='bottom')
ax1.annotate('', xy=(0.94, 0.92), xycoords='axes fraction',
             xytext=(0.94, 0.83), textcoords='axes fraction',
             arrowprops=dict(arrowstyle='->', color='black', lw=2))

# 스케일바 (~ 10 km)
scale_len = 10_000  # 10 km in meters (KR_TM is meter-based)
sb_x  = minx + (maxx - minx) * 0.07
sb_y  = miny + (maxy - miny) * 0.04
ax1.plot([sb_x, sb_x + scale_len], [sb_y, sb_y], 'k-', linewidth=3, zorder=5)
ax1.text(sb_x + scale_len / 2, sb_y + 800, '10 km',
         ha='center', va='bottom', fontsize=8, zorder=5)

ax1.set_xlim(minx - 3000, maxx + 3000)
ax1.set_ylim(miny - 3000, maxy + 3000)
ax1.set_xticks([]); ax1.set_yticks([])
# (a) 제목: fig.text()로 (b) 제목과 동일 높이 (axes 상단 y≈0.90)에 배치
ax1.set_title('')   # 기존 제목 제거

# 범례
legend_patches = [
    mpatches.Patch(facecolor=GRAY_FILL, edgecolor='#888888', label='Seoul (other gu)'),
    mpatches.Patch(facecolor=BLUE_MID,  edgecolor=BLUE_DARK, label='Seocho-gu'),
]
ax1.legend(handles=legend_patches, fontsize=9, loc='lower right',
           framealpha=0.9, edgecolor='#cccccc')

# 한국 위치도 인셋 → 왼쪽 여백(x: 0.01~0.10, y: 중상단)
ax_inset = fig.add_axes([0.01, 0.62, 0.09, 0.20])
ax_inset.set_facecolor('#d6eaf8')
try:
    korea = ox.geocode_to_gdf('South Korea')
    korea_km = korea.to_crs(KR_TM)
    korea_km.plot(ax=ax_inset, facecolor='#f5f5f5', edgecolor='#888', linewidth=0.8)
    # 서울 위치 점
    seoul_pt_x = seocho_centroid.x
    seoul_pt_y = seocho_centroid.y
    ax_inset.plot(seoul_pt_x, seoul_pt_y, 'r*', markersize=10, zorder=5)
    ax_inset.set_xticks([]); ax_inset.set_yticks([])
    ax_inset.set_title('South Korea', fontsize=7, pad=2)
    for spine in ax_inset.spines.values():
        spine.set_edgecolor('#aaaaaa')
        spine.set_linewidth(0.8)
except Exception as e:
    ax_inset.text(0.5, 0.5, 'South Korea', ha='center', va='center', fontsize=7)
    ax_inset.set_xticks([]); ax_inset.set_yticks([])

# ════════════════════════════════════════════════════════════════════════════════
# Panel (b): 서초구 행정동 상세도
# ════════════════════════════════════════════════════════════════════════════════
ax2.set_aspect('equal')

# 동별 채우기: 불투수율 or 단일 색상
if use_imp_color:
    dong_plot = dong.copy()
    vmin, vmax = dong_plot['impervious_pct'].min(), dong_plot['impervious_pct'].max()
    cmap = plt.cm.YlOrRd
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    dong_plot.plot(ax=ax2, column='impervious_pct', cmap=cmap, norm=norm,
                   edgecolor='white', linewidth=1.2, zorder=2)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax2, orientation='vertical', shrink=0.55,
                        pad=0.02, aspect=20)
    cbar.set_label('Impervious Fraction (%)', fontsize=9)
    cbar.ax.tick_params(labelsize=8)
else:
    dong.plot(ax=ax2, facecolor=BLUE_LIGHT, edgecolor='white',
              linewidth=1.2, zorder=2)

# 서초구 외곽 경계선 강조
seocho.boundary.plot(ax=ax2, color=BLUE_DARK, linewidth=2.2, zorder=3)

# 동 레이블
centroids = dong.copy()
centroids['cx'] = dong.geometry.centroid.x
centroids['cy'] = dong.geometry.centroid.y

# 수동 오프셋 조정이 필요한 동 (겹침 방지)
OFFSETS = {
    'Seocho 1':  (  500,    0),
    'Seocho 2':  (  500, -400),
    'Seocho 3':  ( -500,    0),
    'Seocho 4':  (    0,  300),
    'Jamwon':    (    0,    0),
    'Banpo':     ( -800,    0),
    'Banpo 1':   (  800,    0),
    'Banpo 2':   (-800,     0),
    'Banpo 3':   (    0,  300),
    'Banpo 4':   (    0, -300),
    'Bangbae':   (    0,    0),
    'Bangbae 1': (  300,    0),
    'Bangbae 2': (    0,    0),
    'Bangbae 3': (    0,    0),
    'Bangbae 4': (    0,  300),
    'Yangjae 1': (    0,    0),
    'Yangjae 2': (    0,    0),
    'Naegok':    (    0,    0),
}

for _, row in centroids.iterrows():
    lbl = row['label']
    dx, dy = OFFSETS.get(lbl, (0, 0))
    cx, cy = row['cx'] + dx, row['cy'] + dy
    ax2.text(cx, cy, lbl,
             ha='center', va='center',
             fontsize=7.5, fontweight='bold',
             color='#1a1a1a', zorder=5,
             bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                       alpha=0.65, edgecolor='none'))

# 나침반
d2_minx, d2_miny, d2_maxx, d2_maxy = dong.total_bounds
ax2.annotate('N', xy=(0.96, 0.95), xycoords='axes fraction',
             fontsize=13, fontweight='bold', ha='center', va='bottom')
ax2.annotate('', xy=(0.96, 0.95), xycoords='axes fraction',
             xytext=(0.96, 0.86), textcoords='axes fraction',
             arrowprops=dict(arrowstyle='->', color='black', lw=2))

# 스케일바 (~ 2 km)
scale2_len = 2_000
sb2_x = d2_minx + (d2_maxx - d2_minx) * 0.06
sb2_y = d2_miny + (d2_maxy - d2_miny) * 0.03
ax2.plot([sb2_x, sb2_x + scale2_len], [sb2_y, sb2_y], 'k-', linewidth=3, zorder=5)
ax2.text(sb2_x + scale2_len / 2, sb2_y + 300, '2 km',
         ha='center', va='bottom', fontsize=8, zorder=5)

# 좌표계 그리드 (WGS84 위경도 눈금 대신 간단한 박스)
pad = 600
ax2.set_xlim(d2_minx - pad, d2_maxx + pad + 2000)
ax2.set_ylim(d2_miny - pad, d2_maxy + pad + 800)
ax2.set_xticks([]); ax2.set_yticks([])

ax2.set_title('')   # fig.text()로 대체

# 패널 (b) 범례
if not use_imp_color:
    legend2 = [
        mpatches.Patch(facecolor=BLUE_LIGHT, edgecolor='white', label='Administrative dong'),
        mpatches.Patch(facecolor='none', edgecolor=BLUE_DARK,
                       linewidth=2, label='Seocho-gu boundary'),
    ]
    ax2.legend(handles=legend2, fontsize=9, loc='lower right',
               framealpha=0.9, edgecolor='#cccccc')

# ── 패널 제목 및 전체 제목 (fig.text로 동일 높이 배치) ──────────────────────
TITLE_Y = 0.965   # 두 패널 제목 공통 y 위치

# (a) 제목: ax1 중앙 x 기준
ax1_cx = 0.12 + 0.35 / 2   # 0.295
fig.text(ax1_cx, TITLE_Y,
         '(a) Seoul Metropolitan City\nand Seocho-gu Study Area',
         ha='center', va='top', fontsize=12, fontweight='bold')

# (b) 제목: ax2 중앙 x 기준
ax2_cx = 0.52 + 0.46 / 2   # 0.75
fig.text(ax2_cx, TITLE_Y,
         '(b) Seocho-gu: 18 Sub-catchments\n(Administrative Dong Boundaries)',
         ha='center', va='top', fontsize=12, fontweight='bold')

# 전체 제목 (더 위)
fig.text(0.5, 0.998,
         'Study Area: Seocho-gu, Seoul, South Korea',
         ha='center', va='top', fontsize=14, fontweight='bold')

out_path = os.path.join(FIG_DIR, 'fig_study_area.png')
fig.savefig(out_path, facecolor='white')
plt.close(fig)
print(f"저장 완료: {out_path}")
