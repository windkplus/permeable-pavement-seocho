"""
06b_pareto_validation.py
Pareto 해 직접 SWMM 시뮬레이션 검증 (V4 검증)
  - Pareto front에서 대표 해 10개 선택
  - 각 해에 대해 SWMM 직접 실행
  - Surrogate 예측값 vs SWMM 실측값 비교
출력:
  - results/pareto_validation_results.csv
  - results/figures/fig_pareto_validation_scatter.png
  - results/figures/fig_pareto_validation_error.png
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib
from pyswmm import Simulation, Nodes

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

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS_CSV    = os.path.join(BASE_DIR, 'results/subcatchment_params.csv')
PARETO_CSV    = os.path.join(BASE_DIR, 'results/pareto_solutions.csv')
BASE_INP      = os.path.join(BASE_DIR, 'model/seocho_baseline.inp')
VAL_SCEN_DIR  = os.path.join(BASE_DIR, 'model/pareto_validation')
OUT_CSV       = os.path.join(BASE_DIR, 'results/pareto_validation_results.csv')
FIG_DIR       = os.path.join(BASE_DIR, 'results/figures')

os.makedirs(VAL_SCEN_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

BASELINE_RUNOFF = 614298.0  # m³

# ── 1. 데이터 로드 ─────────────────────────────────────────────────────────────
print("1. 데이터 로드 중...")
df_params = pd.read_csv(PARAMS_CSV)
df_params['ADM_CD'] = df_params['ADM_CD'].astype(str)
DONG_CODES = df_params['ADM_CD'].tolist()

df_pareto = pd.read_csv(PARETO_CSV)
rate_cols = [f'rate_{c}' for c in DONG_CODES]

# ── 2. 대표 해 10개 선택 ────────────────────────────────────────────────────────
print("2. 대표 Pareto 해 선택 중...")

# Pareto front를 저감율 기준으로 정렬
df_pareto = df_pareto.sort_values('runoff_red_pct').reset_index(drop=True)

# S1, S2 고정 + 저감율 균등 간격 8개 추가 = 총 10개
idx_max = df_pareto['runoff_red_pct'].idxmax()   # S1: 최대 저감
idx_min = df_pareto['lid_area_pct'].idxmin()     # S2: 최소 면적

# 저감율 구간을 8등분하여 중간해 선택
red_min = df_pareto['runoff_red_pct'].min()
red_max = df_pareto['runoff_red_pct'].max()
target_reds = np.linspace(red_min, red_max, 10)

selected_indices = set([idx_max, idx_min])
for tr in target_reds:
    closest = (df_pareto['runoff_red_pct'] - tr).abs().idxmin()
    selected_indices.add(closest)

selected_indices = sorted(selected_indices)[:10]  # 최대 10개
print(f"   선택된 해: {len(selected_indices)}개")
for i, idx in enumerate(selected_indices):
    r = df_pareto.loc[idx]
    print(f"   해 {i+1:2d}: 저감율={r['runoff_red_pct']:.1f}%  "
          f"설치면적={r['lid_area_pct']:.1f}%  "
          f"Surrogate 예측={r['runoff_m3']:,.0f} m³")

# ── LID 블록 ───────────────────────────────────────────────────────────────────
LID_CONTROL = """
[LID_CONTROLS]
;;Name             Type
PP                 PP

;;Name             Layer      Thick   VoidFrac  Rough    Slope  Side
PP                 SURFACE    0       0         0.013    0      0
PP                 PAVEMENT   100     0.15      0        500    0
PP                 STORAGE    200     0.4       50       0
PP                 DRAIN      0       0.5       0        0      0      0
"""

def make_lid_usage(rate_array):
    lines = [
        "[LID_USAGE]",
        ";;Subcatchment        LIDProcess       Number  Area       Width      "
        "InitSat    FromImp    ToPerv     RptFile    DrainTo    FromPerv",
        ";;==================== ================ ======= ========== ========== "
        "========== ========== ========== ========== ========== =========="
    ]
    for code, rate in zip(DONG_CODES, rate_array):
        r       = df_params.loc[df_params['ADM_CD'] == code].iloc[0]
        area_m2 = float(r['area_ha']) * 10_000
        imp_pct = float(r['impervious_pct'])
        lid_m2  = area_m2 * imp_pct / 100 * rate / 100
        if lid_m2 < 1:
            continue
        width = max(1.0, round(np.sqrt(lid_m2), 1))
        lines.append(
            f"SC_{code:<16} PP               1       "
            f"{lid_m2:<10.1f} {width:<10.1f} "
            f"0          100        0          *          *          0"
        )
    lines.append("")
    return "\n".join(lines)

def generate_inp(rate_array, out_path):
    with open(BASE_INP, 'r', encoding='utf-8') as f:
        content = f.read()
    if np.all(np.array(rate_array) == 0):
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(content)
    else:
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(content)
            f.write("\n" + LID_CONTROL + "\n")
            f.write(make_lid_usage(rate_array))

def run_swmm(inp_path):
    total, peak, prev_t = 0.0, 0.0, None
    with Simulation(inp_path) as sim:
        outfall = Nodes(sim)['OUTFALL_1']
        for _ in sim:
            q = outfall.total_inflow
            t = sim.current_time
            if prev_t:
                total += q * (t - prev_t).total_seconds()
            prev_t = t
            if q > peak:
                peak = q
    return total, peak

# ── 3. SWMM 직접 시뮬레이션 ───────────────────────────────────────────────────
print("\n3. SWMM 직접 시뮬레이션 실행 중...")
records = []

for sim_num, idx in enumerate(selected_indices):
    row       = df_pareto.loc[idx]
    rates     = row[rate_cols].values.astype(float)
    surr_pred = float(row['runoff_m3'])
    red_pct   = float(row['runoff_red_pct'])
    area_pct  = float(row['lid_area_pct'])

    # 시나리오 구분명 부여
    if idx == idx_max:
        label = 'S1_MaxReduction'
    elif idx == idx_min:
        label = 'S2_MinArea'
    else:
        label = f'P{sim_num+1:02d}'

    inp_path = os.path.join(VAL_SCEN_DIR, f'{label}.inp')
    generate_inp(rates, inp_path)

    swmm_vol, swmm_peak = run_swmm(inp_path)

    abs_err  = swmm_vol - surr_pred
    rel_err  = abs_err / swmm_vol * 100          # SWMM 대비 오차 (%)
    red_swmm = (BASELINE_RUNOFF - swmm_vol) / BASELINE_RUNOFF * 100

    records.append({
        'label'         : label,
        'pareto_idx'    : idx,
        'runoff_red_pct_surrogate': round(red_pct, 2),
        'lid_area_pct'  : round(area_pct, 2),
        'surrogate_m3'  : round(surr_pred, 1),
        'swmm_m3'       : round(swmm_vol, 1),
        'runoff_red_pct_swmm': round(red_swmm, 2),
        'abs_error_m3'  : round(abs_err, 1),
        'rel_error_pct' : round(rel_err, 2),
    })
    print(f"   [{sim_num+1:2d}/{len(selected_indices)}] {label:<18} "
          f"Surrogate={surr_pred:>10,.0f} m³  "
          f"SWMM={swmm_vol:>10,.0f} m³  "
          f"RelErr={rel_err:+.2f}%")

df_val = pd.DataFrame(records)
df_val.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')

# ── 4. 오차 요약 ───────────────────────────────────────────────────────────────
mae  = df_val['abs_error_m3'].abs().mean()
rmse = np.sqrt((df_val['abs_error_m3']**2).mean())
mre  = df_val['rel_error_pct'].abs().mean()
bias = df_val['abs_error_m3'].mean()

print(f"\n=== 검증 오차 요약 ===")
print(f"  MAE  : {mae:,.0f} m³  ({mae/BASELINE_RUNOFF*100:.2f}% of baseline)")
print(f"  RMSE : {rmse:,.0f} m³  ({rmse/BASELINE_RUNOFF*100:.2f}% of baseline)")
print(f"  MRE  : {mre:.2f}%")
print(f"  Bias : {bias:+,.0f} m³  ({'과대' if bias > 0 else '과소'} 예측)")
print(f"  Max |RelErr| : {df_val['rel_error_pct'].abs().max():.2f}%")

# ── 5. 시각화 ─────────────────────────────────────────────────────────────────
print("\n4. 시각화 중...")

# ── Figure A: Surrogate 예측 vs SWMM 실측 scatter ─────────────────────────────
fig1, ax1 = plt.subplots(figsize=(7, 7), facecolor='white')
ax1.set_facecolor('white')

colors_pt = [ORANGE if 'S1' in r['label'] else
             GREEN  if 'S2' in r['label'] else BLUE
             for _, r in df_val.iterrows()]

for (_, r), c in zip(df_val.iterrows(), colors_pt):
    ax1.scatter(r['swmm_m3']/1e6, r['surrogate_m3']/1e6,
                color=c, s=90, edgecolors='white', linewidth=1.2,
                zorder=3)
    ax1.annotate(r['label'],
                 xy=(r['swmm_m3']/1e6, r['surrogate_m3']/1e6),
                 xytext=(4, 4), textcoords='offset points',
                 fontsize=9, color='#333333')

# 1:1 선
lim_vals = pd.concat([df_val['swmm_m3'], df_val['surrogate_m3']]) / 1e6
lim = [lim_vals.min() * 0.995, lim_vals.max() * 1.005]
ax1.plot(lim, lim, 'k--', linewidth=1.5, alpha=0.7, label='1:1 line')

# ±5% 허용 오차 밴드
ax1.fill_between(lim, [v*0.95 for v in lim], [v*1.05 for v in lim],
                 alpha=0.08, color=GRAY, label='±5% error band')

ax1.set_xlim(lim); ax1.set_ylim(lim)
ax1.set_xlabel('SWMM Simulated Runoff (million m³)', fontsize=13)
ax1.set_ylabel('Surrogate Predicted Runoff (million m³)', fontsize=13)
ax1.set_title(f'Pareto Solution Validation:\nSurrogate Prediction vs. Direct SWMM Simulation\n'
              f'MAE={mae:,.0f} m³  RMSE={rmse:,.0f} m³  MRE={mre:.1f}%',
              fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(linestyle=':', alpha=0.4)

# 범례 패치 추가
from matplotlib.patches import Patch
legend_add = [Patch(color=ORANGE, label='S1 (Max Reduction)'),
              Patch(color=GREEN,  label='S2 (Min Area)'),
              Patch(color=BLUE,   label='Intermediate solutions')]
ax1.legend(handles=ax1.get_legend_handles_labels()[0] + legend_add, fontsize=10)

fig1.tight_layout()
fig1.savefig(os.path.join(FIG_DIR, 'fig_pareto_validation_scatter.png'), facecolor='white')
plt.close(fig1)
print("   fig_pareto_validation_scatter.png 저장 완료")

# ── Figure B: 저감율 Surrogate vs SWMM + 상대 오차 막대 ────────────────────────
fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(10, 8), facecolor='white',
                                   gridspec_kw={'height_ratios': [3, 1]})
for ax in (ax2a, ax2b):
    ax.set_facecolor('white')

x_pos = np.arange(len(df_val))
labels_x = df_val['label'].tolist()

w = 0.35
bars_s = ax2a.bar(x_pos - w/2, df_val['runoff_red_pct_surrogate'], width=w,
                   color=BLUE, alpha=0.8, label='Surrogate Prediction')
bars_w = ax2a.bar(x_pos + w/2, df_val['runoff_red_pct_swmm'], width=w,
                   color=ORANGE, alpha=0.8, label='SWMM Simulation')

ax2a.set_xticks(x_pos)
ax2a.set_xticklabels(labels_x, rotation=30, ha='right', fontsize=10)
ax2a.set_ylabel('Runoff Reduction Rate (%)', fontsize=12)
ax2a.set_title('Runoff Reduction Rate: Surrogate Prediction vs. SWMM Simulation\n'
               'for 10 Selected Pareto-Optimal Solutions',
               fontsize=13, fontweight='bold')
ax2a.legend(fontsize=11)
ax2a.grid(axis='y', linestyle=':', alpha=0.5)

# 상대 오차 막대
colors_err = [ORANGE if e > 0 else GREEN for e in df_val['rel_error_pct']]
ax2b.bar(x_pos, df_val['rel_error_pct'], color=colors_err, alpha=0.8, width=0.6)
ax2b.axhline(0, color='black', linewidth=0.8)
ax2b.axhline( 5, color=GRAY, linewidth=1.0, linestyle='--', alpha=0.6, label='+5%')
ax2b.axhline(-5, color=GRAY, linewidth=1.0, linestyle='--', alpha=0.6, label='-5%')
ax2b.set_xticks(x_pos)
ax2b.set_xticklabels(labels_x, rotation=30, ha='right', fontsize=10)
ax2b.set_ylabel('Relative Error (%)', fontsize=12)
ax2b.set_title('(Surrogate − SWMM) / SWMM × 100', fontsize=11)
ax2b.legend(fontsize=10)
ax2b.grid(axis='y', linestyle=':', alpha=0.5)

fig2.tight_layout()
fig2.savefig(os.path.join(FIG_DIR, 'fig_pareto_validation_error.png'), facecolor='white')
plt.close(fig2)
print("   fig_pareto_validation_error.png 저장 완료")

# ── 6. 논문용 표 출력 ─────────────────────────────────────────────────────────
print("\n=== 논문 Table 용 검증 결과 ===")
print(f"{'Solution':<18} {'Surr. Red(%)':>12} {'SWMM Red(%)':>12} "
      f"{'Surr.(m³)':>12} {'SWMM(m³)':>12} {'Rel.Err(%)':>11}")
print("-" * 80)
for _, r in df_val.iterrows():
    print(f"  {r['label']:<16} {r['runoff_red_pct_surrogate']:>12.1f} "
          f"{r['runoff_red_pct_swmm']:>12.1f} "
          f"{r['surrogate_m3']:>12,.0f} {r['swmm_m3']:>12,.0f} "
          f"{r['rel_error_pct']:>+11.2f}")
print("-" * 80)
print(f"  {'MAE':>16}: {mae:>12,.0f} m³   MRE: {mre:.2f}%")
print(f"  {'RMSE':>16}: {rmse:>12,.0f} m³   Bias: {bias:+,.0f} m³")
print(f"\n저장 파일:")
print(f"  결과 CSV : {OUT_CSV}")
print(f"  그래프   : results/figures/fig_pareto_validation_*.png")
