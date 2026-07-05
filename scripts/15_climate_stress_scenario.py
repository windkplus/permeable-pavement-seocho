"""
15_climate_stress_scenario.py
Climate-change stress test (Reviewer #1-1, #2-1).
Re-runs the baseline + uniform PP coverage sweep (0-100%) under a
climate-adjusted design storm obtained by scaling the 10-yr 60-min Huff
4th-quartile rainfall intensities by a climate-change factor CCF = 1.30
(+30 %, consistent with the ~+30 % increase in 100-yr 60-min rainfall
depth projected for the Seoul metropolitan region by mid-21st century;
Ministry of Environment of Korea, 2021).

The scaling is applied through the SWMM rain-gauge Storm Coefficient
Factor (SCF), leaving the temporal (Huff) pattern unchanged.

Output: results/simulation_results_climate130.csv
"""

import os
import numpy as np
import pandas as pd
from pyswmm import Simulation, Nodes

# Reuse LID logic from 03_run_simulation.py
import importlib.util

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_INP   = os.path.join(BASE_DIR, 'model/seocho_baseline.inp')
SCEN_DIR   = os.path.join(BASE_DIR, 'model/scenarios_climate130')
PARAMS_CSV = os.path.join(BASE_DIR, 'results/subcatchment_params.csv')
OUTPUT_CSV = os.path.join(BASE_DIR, 'results/simulation_results_climate130.csv')

CCF = 1.30   # climate-change scaling factor (+30 %)

os.makedirs(SCEN_DIR, exist_ok=True)

# import make_lid_usage + LID_CONTROL_BLOCK from script 03
spec = importlib.util.spec_from_file_location(
    "s03", os.path.join(BASE_DIR, 'scripts', '03_run_simulation.py'))
# NOTE: importing 03 would execute its simulation loop; instead we copy the
# two needed objects manually to avoid side effects.

LID_CONTROL_BLOCK = """
[LID_CONTROLS]
PP                 PP
PP                 SURFACE    0       0         0.013            0              0
PP                 PAVEMENT   100     0.15      0                500            0
PP                 STORAGE    200     0.4       50               0
PP                 DRAIN      0       0.5       0                0              0              0
"""

def make_lid_usage(df_params, lid_rate_pct):
    lines = ["[LID_USAGE]"]
    for _, r in df_params.iterrows():
        sc_name  = 'SC_' + str(r['ADM_CD'])
        area_m2  = r['area_ha'] * 10_000
        imp_area = area_m2 * (r['impervious_pct'] / 100)
        lid_area = imp_area * (lid_rate_pct / 100)
        if lid_area < 1:
            continue
        width = max(1.0, round(np.sqrt(lid_area), 1))
        lines.append(
            f"{sc_name:<21} {'PP':<16} {'1':<7} "
            f"{lid_area:<10.1f} {width:<10.1f} "
            f"{'0':<10} {'100':<10} {'0':<10} "
            f"{'*':<10} {'*':<10} {'0'}")
    lines.append("")
    return "\n".join(lines)


def make_climate_base(base_inp, ccf):
    """Return the baseline .inp content with the TS_RAIN rainfall-intensity
    ordinates scaled by ccf. (Note: the rain-gauge SCF field is the SWMM snow
    catch factor and does NOT scale rainfall, so the time series itself is
    scaled instead.)"""
    with open(base_inp, 'r', encoding='utf-8') as f:
        content = f.read()
    out_lines = []
    for line in content.splitlines():
        s = line.strip()
        # TS_RAIN  01/01/2024  00:00  33.8
        if s.startswith('TS_RAIN') and not s.startswith('TS_RAIN ' + ';'):
            parts = line.split()
            if len(parts) == 4:
                try:
                    val = float(parts[3]) * ccf
                    line = (f"{parts[0]:<21} {parts[1]:<12} "
                            f"{parts[2]:<10} {val:.4f}")
                except ValueError:
                    pass
        out_lines.append(line)
    return "\n".join(out_lines)


def generate_scenario_inp(climate_base_content, output_inp, lid_rate_pct, df_params):
    with open(output_inp, 'w', encoding='utf-8') as f:
        f.write(climate_base_content)
        if lid_rate_pct != 0:
            f.write("\n")
            f.write(LID_CONTROL_BLOCK)
            f.write("\n")
            f.write(make_lid_usage(df_params, lid_rate_pct))


def run_simulation(inp_path):
    total_runoff_m3 = 0.0
    peak_flow_cms   = 0.0
    prev_time       = None
    with Simulation(inp_path) as sim:
        nodes   = Nodes(sim)
        outfall = nodes['OUTFALL_1']
        for _ in sim:
            flow      = outfall.total_inflow
            curr_time = sim.current_time
            if prev_time is not None:
                dt = (curr_time - prev_time).total_seconds()
                total_runoff_m3 += flow * dt
            prev_time = curr_time
            if flow > peak_flow_cms:
                peak_flow_cms = flow
    return total_runoff_m3, peak_flow_cms


print("=" * 66)
print(f"Climate-stress scenario (CCF = {CCF:.2f}, +{(CCF-1)*100:.0f}% design rainfall)")
print("=" * 66)

df_params      = pd.read_csv(PARAMS_CSV)
climate_base   = make_climate_base(BASE_INP, CCF)
lid_rates      = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
results        = []

for rate in lid_rates:
    inp_path = os.path.join(SCEN_DIR, f"LID_{rate:03d}pct_c130.inp")
    generate_scenario_inp(climate_base, inp_path, rate, df_params)
    print(f"[{rate:3d}%] running...", end=" ", flush=True)
    total_m3, peak_cms = run_simulation(inp_path)
    print(f"total={total_m3:,.0f} m3 | peak={peak_cms:.3f} m3/s")
    results.append({'lid_rate_pct': rate,
                    'total_runoff_m3': round(total_m3, 1),
                    'peak_flow_cms': round(peak_cms, 4)})

df_res = pd.DataFrame(results)
base_vol  = df_res.loc[df_res['lid_rate_pct'] == 0, 'total_runoff_m3'].values[0]
base_peak = df_res.loc[df_res['lid_rate_pct'] == 0, 'peak_flow_cms'].values[0]
df_res['vol_reduction_m3']   = (base_vol - df_res['total_runoff_m3']).round(1)
df_res['vol_reduction_pct']  = ((df_res['vol_reduction_m3'] / base_vol) * 100).round(1)
df_res['peak_reduction_pct'] = (((base_peak - df_res['peak_flow_cms']) / base_peak) * 100).round(1)
df_res.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

print("\n" + "=" * 66)
print(f"{'rate':>5} {'total(m3)':>13} {'red(m3)':>12} {'red%':>7} {'peakRed%':>9}")
print("-" * 66)
for _, r in df_res.iterrows():
    print(f"{r['lid_rate_pct']:>4}% {r['total_runoff_m3']:>13,.0f} "
          f"{r['vol_reduction_m3']:>12,.0f} {r['vol_reduction_pct']:>6.1f}% "
          f"{r['peak_reduction_pct']:>8.1f}%")
print("=" * 66)
print(f"saved -> {OUTPUT_CSV}")
