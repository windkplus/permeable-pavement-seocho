"""
03_run_simulation.py
pySWMM 시뮬레이션 실행
  - Baseline (LID 없음)
  - 투수성 포장 적용률 10~100% (10% 단위) 시나리오
출력: results/simulation_results.csv
"""

import os
import re
import shutil
import numpy as np
import pandas as pd
from pyswmm import Simulation, Subcatchments, Nodes

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_INP    = os.path.join(BASE_DIR, 'model/seocho_baseline.inp')
SCEN_DIR    = os.path.join(BASE_DIR, 'model/scenarios')
OUTPUT_CSV  = os.path.join(BASE_DIR, 'results/simulation_results.csv')
PARAMS_CSV  = os.path.join(BASE_DIR, 'results/subcatchment_params.csv')

os.makedirs(SCEN_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)

# ── LID 파라미터 (투수성 포장: Permeable Pavement) ────────────────────────────
# SWMM LID_CONTROLS 섹션 — 서울 일반 투수성 아스팔트 기준
LID_CONTROL_BLOCK = """
[LID_CONTROLS]
;;Name             Type
;;================ ====
PP                 PP

;;Name             Layer      Thick   VoidFrac  Rough/ImperFrac  Slope/Permeab  SideSlope/Clog
;;================ ========== ======= ========= ================ ============== ==============
;;  SURFACE : Thickness VoidFrac Roughness SurfSlope SideSlope
PP                 SURFACE    0       0         0.013            0              0
;;  PAVEMENT: Thickness VoidFrac ImperFrac Permeability ClogFactor
PP                 PAVEMENT   100     0.15      0                500            0
;;  STORAGE : Thickness VoidFrac Ksat ClogFactor
PP                 STORAGE    200     0.4       50               0
;;  DRAIN   : Coeff Expon Offset Delay OpenHead CloseHead
PP                 DRAIN      0       0.5       0                0              0              0
"""

def make_lid_usage(df_params, lid_rate_pct):
    """
    LID_USAGE 섹션 생성
    lid_rate_pct: 각 소유역 불투수면 중 투수성 포장으로 전환할 비율 (%)
    """
    lines = [
        "[LID_USAGE]",
        ";;Subcatchment        LIDProcess       Number  Area       Width      InitSat    FromImp    ToPerv     RptFile    DrainTo    FromPerv",
        ";;==================== ================ ======= ========== ========== ========== ========== ========== ========== ========== =========="
    ]
    for _, r in df_params.iterrows():
        sc_name   = 'SC_' + str(r['ADM_CD'])
        area_m2   = r['area_ha'] * 10_000
        imp_area  = area_m2 * (r['impervious_pct'] / 100)
        lid_area  = imp_area * (lid_rate_pct / 100)   # m² (투수성 포장 면적)

        if lid_area < 1:
            continue   # 면적이 너무 작으면 생략

        width = max(1.0, round(np.sqrt(lid_area), 1))
        lines.append(
            f"{sc_name:<21} {'PP':<16} {'1':<7} "
            f"{lid_area:<10.1f} {width:<10.1f} "
            f"{'0':<10} {'100':<10} {'0':<10} "
            f"{'*':<10} {'*':<10} {'0'}"
        )
    lines.append("")
    return "\n".join(lines)


def generate_scenario_inp(base_inp, output_inp, lid_rate_pct, df_params):
    """기본 .inp에 LID 섹션을 추가한 시나리오 파일 생성"""
    with open(base_inp, 'r', encoding='utf-8') as f:
        content = f.read()

    if lid_rate_pct == 0:
        # Baseline: LID 섹션 없음
        with open(output_inp, 'w', encoding='utf-8') as f:
            f.write(content)
    else:
        lid_usage = make_lid_usage(df_params, lid_rate_pct)
        with open(output_inp, 'w', encoding='utf-8') as f:
            f.write(content)
            f.write("\n")
            f.write(LID_CONTROL_BLOCK)
            f.write("\n")
            f.write(lid_usage)


def run_simulation(inp_path):
    """
    시뮬레이션 실행 → 총 유출량(m³) 및 첨두유량(m³/s) 반환
    """
    total_runoff_m3 = 0.0
    peak_flow_cms   = 0.0
    prev_time       = None

    with Simulation(inp_path) as sim:
        nodes   = Nodes(sim)
        outfall = nodes['OUTFALL_1']

        for _ in sim:
            flow      = outfall.total_inflow      # m³/s
            curr_time = sim.current_time           # datetime

            if prev_time is not None:
                dt = (curr_time - prev_time).total_seconds()
                total_runoff_m3 += flow * dt

            prev_time = curr_time
            if flow > peak_flow_cms:
                peak_flow_cms = flow

    return total_runoff_m3, peak_flow_cms


# ── 시뮬레이션 실행 ────────────────────────────────────────────────────────────
print("=" * 60)
print("pySWMM 시뮬레이션 시작")
print("=" * 60)

df_params  = pd.read_csv(PARAMS_CSV)
lid_rates  = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
results    = []

for rate in lid_rates:
    label    = f"LID_{rate:03d}pct"
    inp_path = os.path.join(SCEN_DIR, f"{label}.inp")

    generate_scenario_inp(BASE_INP, inp_path, rate, df_params)

    print(f"\n[{rate:3d}%] 시뮬레이션 실행 중...", end=" ", flush=True)
    total_m3, peak_cms = run_simulation(inp_path)
    print(f"총유출={total_m3:,.0f} m³  |  첨두유량={peak_cms:.3f} m³/s")

    results.append({
        'lid_rate_pct'   : rate,
        'total_runoff_m3': round(total_m3, 1),
        'peak_flow_cms'  : round(peak_cms, 4),
    })

# ── 유출저감량 계산 ────────────────────────────────────────────────────────────
df_res = pd.DataFrame(results)
baseline_vol  = df_res.loc[df_res['lid_rate_pct'] == 0, 'total_runoff_m3'].values[0]
baseline_peak = df_res.loc[df_res['lid_rate_pct'] == 0, 'peak_flow_cms'].values[0]

df_res['vol_reduction_m3']  = (baseline_vol  - df_res['total_runoff_m3']).round(1)
df_res['vol_reduction_pct'] = ((df_res['vol_reduction_m3'] / baseline_vol) * 100).round(1)
df_res['peak_reduction_pct']= (((baseline_peak - df_res['peak_flow_cms']) / baseline_peak) * 100).round(1)

df_res.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

# ── 결과 요약 출력 ─────────────────────────────────────────────────────────────
print("\n")
print("=" * 70)
print("시뮬레이션 결과 요약")
print("=" * 70)
print(f"{'적용률':>6} {'총유출량(m³)':>14} {'유출저감량(m³)':>14} {'저감율(%)':>10} {'첨두저감율(%)':>13}")
print("-" * 70)
for _, r in df_res.iterrows():
    print(f"{r['lid_rate_pct']:>5}%  "
          f"{r['total_runoff_m3']:>14,.1f}  "
          f"{r['vol_reduction_m3']:>14,.1f}  "
          f"{r['vol_reduction_pct']:>9.1f}%  "
          f"{r['peak_reduction_pct']:>12.1f}%")
print("=" * 70)
print(f"\n결과 저장 → {OUTPUT_CSV}")
