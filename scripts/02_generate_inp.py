"""
02_generate_inp.py
서초구 SWMM 기본 모델 (.inp) 파일 생성
  - 소유역 18개 (행정동 단위)
  - 설계강우: 서울 10년 빈도 60분 (Huff 4분위, P=67.6mm)
  - 침투: Green-Ampt
  - LID 없음 (Baseline)
출력: model/seocho_baseline.inp
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS_CSV   = os.path.join(BASE_DIR, 'results/subcatchment_params.csv')
BOUNDARY_SHP = os.path.join(BASE_DIR, 'data/boundary/bnd_dong_11220_2025_2Q/bnd_dong_11220_2025_2Q.shp')
DEM_MERGED   = os.path.join(BASE_DIR, 'data/dem/dem_merged_5179.tif')
OUTPUT_INP   = os.path.join(BASE_DIR, 'model/seocho_baseline.inp')

os.makedirs(os.path.join(BASE_DIR, 'model'), exist_ok=True)
TARGET_CRS = 'EPSG:5179'

# ── 1. 파라미터 및 경계 로드 ───────────────────────────────────────────────────
print("1. 데이터 로드 중...")
df   = pd.read_csv(PARAMS_CSV)
dong = gpd.read_file(BOUNDARY_SHP).to_crs(TARGET_CRS)
df['ADM_CD'] = df['ADM_CD'].astype(str)
dong['ADM_CD'] = dong['ADM_CD'].astype(str)
dong = dong.merge(df[['ADM_CD','area_ha','impervious_pct','avg_slope_deg']],
                  on='ADM_CD')
dong['cx'] = dong.geometry.centroid.x
dong['cy'] = dong.geometry.centroid.y

# SWMM 객체명 (ADM_CD 기반, 한글 대신 코드 사용 → SWMM 호환성 확보)
dong['SC'] = 'SC_' + dong['ADM_CD'].astype(str)   # 소유역명
dong['JN'] = 'J_'  + dong['ADM_CD'].astype(str)   # 절점명
dong['CO'] = 'C_'  + dong['ADM_CD'].astype(str)   # 관로명

# ── 2. DEM에서 각 동 centroid 고도 추출 ───────────────────────────────────────
print("2. DEM 고도 추출 중...")
with rasterio.open(DEM_MERGED) as dem:
    coords = list(zip(dong['cx'], dong['cy']))
    dong['elev'] = [list(v)[0] for v in dem.sample(coords)]
dong['elev'] = dong['elev'].clip(lower=0).round(1)

# ── 3. 파생 파라미터 계산 ──────────────────────────────────────────────────────
dong['area_m2']  = dong['area_ha'] * 10_000
dong['width']    = (np.sqrt(dong['area_m2']) / 2).round(1)   # 특성폭
dong['slope_pct']= (np.tan(np.radians(dong['avg_slope_deg'].fillna(5))) * 100).clip(lower=0.5).round(2)

# 표시 좌표 (SWMM MAP 용, m 단위 오프셋)
x0, y0 = dong['cx'].min() - 500, dong['cy'].min() - 500
dong['px'] = (dong['cx'] - x0).round(1)
dong['py'] = (dong['cy'] - y0).round(1)

outfall_elev = max(0.0, dong['elev'].min() - 3.0)

print(f"   소유역 {len(dong)}개 | 절점 {len(dong)}개 | 관로 {len(dong)}개 | Outfall 1개")

# ── 4. 설계강우 (서울 10년 빈도 60분, Huff 4분위) ─────────────────────────────
# 6분 간격 강우강도 (mm/hr), 총 강우량 = 67.6 mm
# 참고: 서울시 하수도설계기준 / 한국건설기술연구원 확률강우량도
huff_times_min  = list(range(0, 67, 6))          # 0,6,12,...,60,66(종료용)
huff_intensity  = [33.8, 33.8, 33.8, 33.8,       # 0~24분 (약한 비)
                   54.1, 81.1,                     # 24~36분
                   135.2, 135.2,                   # 36~48분 (피크)
                   87.9, 47.3,                     # 48~60분
                   0.0]                            # 60분 이후 종료

# ── 5. .inp 파일 작성 ──────────────────────────────────────────────────────────
print("3. .inp 파일 작성 중...")

def section(title):
    return [f"[{title}]"]

def divider(cols):
    return ";;" + " ".join(f"{'':=<{w}}" for w in cols)

L = []   # 모든 라인을 담을 리스트

# ── [TITLE] ──
L += ["[TITLE]",
      "서초구 투수성포장 LID 유출저감 분석",
      "Baseline 모델 (현황, LID 미적용)",
      "강우: 서울 10년 빈도 60분 Huff 4분위 (67.6mm)",
      ""]

# ── [OPTIONS] ──
L += ["[OPTIONS]",
      f"{'FLOW_UNITS':<20} CMS",
      f"{'INFILTRATION':<20} GREEN_AMPT",
      f"{'FLOW_ROUTING':<20} KINWAVE",
      f"{'LINK_OFFSETS':<20} DEPTH",
      f"{'MIN_SLOPE':<20} 0",
      f"{'ALLOW_PONDING':<20} NO",
      f"{'SKIP_STEADY_STATE':<20} NO",
      f"{'START_DATE':<20} 01/01/2024",
      f"{'START_TIME':<20} 00:00:00",
      f"{'REPORT_START_DATE':<20} 01/01/2024",
      f"{'REPORT_START_TIME':<20} 00:00:00",
      f"{'END_DATE':<20} 01/01/2024",
      f"{'END_TIME':<20} 03:00:00",
      f"{'SWEEP_START':<20} 01/01",
      f"{'SWEEP_END':<20} 12/31",
      f"{'DRY_DAYS':<20} 0",
      f"{'REPORT_STEP':<20} 00:01:00",
      f"{'WET_STEP':<20} 00:01:00",
      f"{'DRY_STEP':<20} 01:00:00",
      f"{'ROUTING_STEP':<20} 0:00:30",
      f"{'VARIABLE_STEP':<20} 0.75",
      f"{'MINIMUM_STEP':<20} 0.5",
      f"{'INERTIAL_DAMPING':<20} PARTIAL",
      f"{'NORMAL_FLOW_LIMITED':<20} BOTH",
      f"{'FORCE_MAIN_EQUATION':<20} H-W",
      ""]

# ── [EVAPORATION] ──
L += ["[EVAPORATION]",
      "CONSTANT         0.0",
      "DRY_ONLY         NO",
      ""]

# ── [RAINGAGES] ──
L += ["[RAINGAGES]",
      ";;Name             Format     Interval   SCF      Source",
      ";;================ ========== ========== ======== ================",
      f"{'RG1':<17} {'INTENSITY':<10} {'0:06':<10} {'1.0':<8} TIMESERIES TS_RAIN",
      ""]

# ── [SUBCATCHMENTS] ──
L += ["[SUBCATCHMENTS]",
      ";;Name                 Rain Gage    Outlet               Area       %Imperv    Width      %Slope     CurbLen",
      ";;==================== ============ ==================== ========== ========== ========== ========== =========="]
for _, r in dong.iterrows():
    L.append(f"{r['SC']:<21} {'RG1':<12} {r['JN']:<21} "
             f"{r['area_ha']:<10.3f} {r['impervious_pct']:<10.1f} "
             f"{r['width']:<10.1f} {r['slope_pct']:<10.2f} {'0'}")
L.append("")

# ── [SUBAREAS] ──
L += ["[SUBAREAS]",
      ";;Subcatchment        N-Imperv     N-Perv       S-Imperv     S-Perv       PctZero    RouteTo",
      ";;==================== ============ ============ ============ ============ ========== =========="]
for _, r in dong.iterrows():
    L.append(f"{r['SC']:<21} {'0.011':<12} {'0.15':<12} "
             f"{'1.52':<12} {'5.08':<12} {'25':<10} {'OUTLET'}")
L.append("")

# ── [INFILTRATION] Green-Ampt (서울 도시토양 기준) ──
# Suction=110mm, Ksat=3.0mm/hr, IMD=0.26
L += ["[INFILTRATION]",
      ";;Subcatchment        Suction      Ksat         IMD",
      ";;==================== ============ ============ ============"]
for _, r in dong.iterrows():
    L.append(f"{r['SC']:<21} {'110':<12} {'3.0':<12} {'0.26'}")
L.append("")

# ── [JUNCTIONS] ──
L += ["[JUNCTIONS]",
      ";;Name                 Elevation    MaxDepth     InitDepth    SurDepth     Aponded",
      ";;==================== ============ ============ ============ ============ ============"]
for _, r in dong.iterrows():
    L.append(f"{r['JN']:<21} {r['elev']:<12.1f} {'5.0':<12} {'0':<12} {'0':<12} {'0'}")
L.append("")

# ── [OUTFALLS] ──
L += ["[OUTFALLS]",
      ";;Name                 Elevation    Type         Stage Data       Gated",
      ";;==================== ============ ============ ================ ======",
      f"{'OUTFALL_1':<21} {outfall_elev:<12.1f} {'FREE':<12} {'':16} {'NO'}",
      ""]

# ── [CONDUITS] ── (각 junction → OUTFALL_1, star topology)
L += ["[CONDUITS]",
      ";;Name                 From Node            To Node              Length     Manning N    InOffset     OutOffset    InitFlow     MaxFlow",
      ";;==================== ==================== ==================== ========== ============ ============ ============ ============ ============"]
for _, r in dong.iterrows():
    L.append(f"{r['CO']:<21} {r['JN']:<21} {'OUTFALL_1':<21} "
             f"{'500':<10} {'0.013':<12} {'0':<12} {'0':<12} {'0':<12} {'0'}")
L.append("")

# ── [XSECTIONS] ──
L += ["[XSECTIONS]",
      ";;Link                 Shape        Geom1      Geom2    Geom3    Geom4    Barrels",
      ";;==================== ============ ========== ======== ======== ======== ======="]
for _, r in dong.iterrows():
    L.append(f"{r['CO']:<21} {'CIRCULAR':<12} {'1.0':<10} {'0':<8} {'0':<8} {'0':<8} {'1'}")
L.append("")

# ── [TIMESERIES] Huff 4분위 설계강우 ──
L += ["[TIMESERIES]",
      ";;Name                 Date         Time       Value",
      ";;==================== ============ ========== =========="]
for i, intensity in enumerate(huff_intensity):
    t_min = huff_times_min[i]
    t_str = f"{t_min // 60:02d}:{t_min % 60:02d}"
    L.append(f"{'TS_RAIN':<21} {'01/01/2024':<12} {t_str:<10} {intensity:.1f}")
L.append("")

# ── [REPORT] ──
L += ["[REPORT]",
      "SUBCATCHMENTS ALL",
      "NODES ALL",
      "LINKS ALL",
      ""]

# ── [COORDINATES] ──
L += ["[COORDINATES]",
      ";;Node                 X-Coord          Y-Coord",
      ";;==================== ================ ================"]
for _, r in dong.iterrows():
    L.append(f"{r['JN']:<21} {r['px']:<16.1f} {r['py']:.1f}")
cx_mean = dong['px'].mean()
cy_min  = dong['py'].min() - 1000
L.append(f"{'OUTFALL_1':<21} {cx_mean:<16.1f} {cy_min:.1f}")
L.append("")

# ── [SYMBOLS] ──
L += ["[SYMBOLS]",
      ";;Gage                 X-Coord          Y-Coord",
      ";;==================== ================ ================",
      f"{'RG1':<21} {cx_mean:<16.1f} {dong['py'].max() + 500:.1f}",
      ""]

# ── 파일 저장 ──
with open(OUTPUT_INP, 'w', encoding='utf-8') as f:
    f.write("\n".join(L))

# ── 결과 요약 출력 ──
print(f"\n완료! → {OUTPUT_INP}")
print(f"\n{'='*55}")
print(f"{'항목':<20} {'내용'}")
print(f"{'='*55}")
print(f"{'소유역 수':<20} {len(dong)}개 (행정동 단위)")
print(f"{'절점(Junction) 수':<20} {len(dong)}개")
print(f"{'관로(Conduit) 수':<20} {len(dong)}개")
print(f"{'Outfall 수':<20} 1개")
print(f"{'강우 입력':<20} Huff 4분위 60분 (67.6mm)")
print(f"{'침투 모델':<20} Green-Ampt")
print(f"{'시뮬레이션 시간':<20} 3시간 (강우 60분+추가 2시간)")
print(f"{'LID 적용':<20} 없음 (Baseline)")
print(f"{'='*55}")
print(f"\n서초구 소유역 파라미터:")
print(dong[['ADM_NM','area_ha','impervious_pct','slope_pct','elev']].to_string(index=False))
