"""
10_permeability_sensitivity.py
투수성 포장 투수율 민감도 분석 (Clogging 효과)
  - 포장층 투수율: 500 / 100 / 30 / 5 mm/hr (신설→노화→막힘)
  - 커버리지: 0, 10, 20, 30, 40, 50, 70, 100%
출력:
  - results/permeability_sensitivity.csv
  - results/figures/fig_permeability_sensitivity.png
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pyswmm import Simulation, Nodes

matplotlib.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'font.size'        : 12,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'figure.dpi'       : 150,
    'savefig.dpi'      : 300,
    'savefig.bbox'     : 'tight',
})

# ── 경로 ───────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_INP   = os.path.join(BASE_DIR, 'model/seocho_baseline.inp')
PARAMS_CSV = os.path.join(BASE_DIR, 'results/subcatchment_params.csv')
SCEN_DIR   = os.path.join(BASE_DIR, 'model/perm_sensitivity')
OUT_CSV    = os.path.join(BASE_DIR, 'results/permeability_sensitivity.csv')
FIG_DIR    = os.path.join(BASE_DIR, 'results/figures')
os.makedirs(SCEN_DIR, exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)

# ── 시나리오 설정 ───────────────────────────────────────────────────────────────
PERM_SCENARIOS = [
    (500, 'Best (New)',    '#2166AC'),   # 신설
    (100, 'Mid (3-5 yr)', '#4DAC26'),   # 부분 막힘
    ( 30, 'Low (Aged)',   '#D6604D'),   # 노화
    (  5, 'Clogged',      '#888888'),   # 사실상 불투수
]
LID_RATES = [0, 10, 20, 30, 40, 50, 70, 100]

df_params = pd.read_csv(PARAMS_CSV)
BASELINE_VOL = 614298.0   # 03_run_simulation.py 결과


def make_lid_control_block(perm_mmhr):
    return f"""
[LID_CONTROLS]
;;Name             Type
;;================ ====
PP                 PP

;;Name             Layer      Thick   VoidFrac  Rough/ImperFrac  Slope/Permeab  SideSlope/Clog
;;================ ========== ======= ========= ================ ============== ==============
PP                 SURFACE    0       0         0.013            0              0
PP                 PAVEMENT   100     0.15      0                {perm_mmhr}    0
PP                 STORAGE    200     0.4       50               0
PP                 DRAIN      0       0.5       0                0              0              0
"""


def make_lid_usage(df_params, lid_rate_pct):
    lines = [
        "[LID_USAGE]",
        ";;Subcatchment        LIDProcess       Number  Area       Width      InitSat    FromImp    ToPerv     RptFile    DrainTo    FromPerv",
        ";;==================== ================ ======= ========== ========== ========== ========== ========== ========== ========== =========="
    ]
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
            f"{'*':<10} {'*':<10} {'0'}"
        )
    lines.append("")
    return "\n".join(lines)


def generate_inp(perm_mmhr, lid_rate_pct):
    with open(BASE_INP, 'r', encoding='utf-8') as f:
        content = f.read()
    fname = os.path.join(SCEN_DIR, f"perm{perm_mmhr:04d}_lid{lid_rate_pct:03d}.inp")
    with open(fname, 'w', encoding='utf-8') as f:
        f.write(content)
        if lid_rate_pct > 0:
            f.write("\n")
            f.write(make_lid_control_block(perm_mmhr))
            f.write("\n")
            f.write(make_lid_usage(df_params, lid_rate_pct))
    return fname


def run_sim(inp_path):
    total = 0.0
    peak  = 0.0
    prev  = None
    with Simulation(inp_path) as sim:
        nodes   = Nodes(sim)
        outfall = nodes['OUTFALL_1']
        for _ in sim:
            flow = outfall.total_inflow
            curr = sim.current_time
            if prev is not None:
                total += flow * (curr - prev).total_seconds()
            prev = curr
            if flow > peak:
                peak = flow
    return total, peak


# ── 시뮬레이션 실행 ────────────────────────────────────────────────────────────
print("=" * 65)
print("투수율 민감도 분석 시작")
print("=" * 65)

records = []
for perm, label, _ in PERM_SCENARIOS:
    print(f"\n[{label} - {perm} mm/hr]")
    for rate in LID_RATES:
        if rate == 0 and perm != 500:
            # baseline은 투수율 무관하게 동일 → 500 케이스 재사용
            vol  = BASELINE_VOL
            peak = 85.84
        else:
            inp  = generate_inp(perm if rate > 0 else 500, rate)
            vol, peak = run_sim(inp)
        red_pct = (BASELINE_VOL - vol) / BASELINE_VOL * 100
        print(f"  LID {rate:3d}%  →  유출={vol:,.0f} m³  저감={red_pct:.2f}%")
        records.append({
            'perm_mmhr'     : perm,
            'perm_label'    : label,
            'lid_rate_pct'  : rate,
            'total_runoff_m3': round(vol, 1),
            'reduction_pct' : round(red_pct, 2),
        })

df = pd.DataFrame(records)
df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
print(f"\n저장 완료: {OUT_CSV}")

# ── 시각화 ─────────────────────────────────────────────────────────────────────
print("\n시각화 중...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
for ax in axes:
    ax.set_facecolor('white')

# ── Panel (a): 커버리지 vs 저감률 (4개 투수율 라인) ──────────────────────────
ax1 = axes[0]
for perm, label, color in PERM_SCENARIOS:
    sub = df[df['perm_mmhr'] == perm].sort_values('lid_rate_pct')
    lw  = 2.5 if perm == 500 else 1.8
    ls  = '-'  if perm == 500 else '--' if perm == 100 else '-.' if perm == 30 else ':'
    ax1.plot(sub['lid_rate_pct'], sub['reduction_pct'],
             color=color, linewidth=lw, linestyle=ls,
             marker='o', markersize=5, label=f'{label} ({perm} mm/hr)')

ax1.axvline(30, color='gray', linestyle=':', linewidth=1.2, alpha=0.7)
ax1.axvline(50, color='gray', linestyle=':', linewidth=1.2, alpha=0.7)
ax1.text(30.5, 0.5, '30%', fontsize=9, color='gray')
ax1.text(50.5, 0.5, '50%', fontsize=9, color='gray')
ax1.set_xlabel('Permeable Pavement Coverage Rate (% of Impervious Area)', fontsize=12)
ax1.set_ylabel('Runoff Volume Reduction Rate (%)', fontsize=12)
ax1.set_title('(a) Runoff Reduction vs. Coverage Rate\nby Pavement Permeability Level',
              fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(linestyle=':', alpha=0.4)
ax1.set_xlim(0, 100)
ax1.set_ylim(0, None)

# ── Panel (b): 30% / 50% 커버리지에서 투수율별 저감률 비교 바차트 ────────────
ax2 = axes[1]
target_rates = [30, 50]
x     = np.arange(len(PERM_SCENARIOS))
width = 0.32
colors_perm = [c for _, _, c in PERM_SCENARIOS]
labels_perm = [f'{l}\n({p} mm/hr)' for p, l, _ in PERM_SCENARIOS]

for i, cov in enumerate(target_rates):
    vals = [df[(df['perm_mmhr'] == p) & (df['lid_rate_pct'] == cov)]['reduction_pct'].values[0]
            for p, _, _ in PERM_SCENARIOS]
    offset = (i - 0.5) * width
    bars = ax2.bar(x + offset, vals, width=width * 0.9,
                   color=colors_perm, alpha=0.75 if i == 0 else 0.45,
                   edgecolor='white', linewidth=0.8,
                   label=f'{cov}% coverage' if i == 0 else f'{cov}% coverage (lighter)')
    for xi, v in zip(x + offset, vals):
        ax2.text(xi, v + 0.1, f'{v:.1f}%', ha='center', va='bottom', fontsize=8.5)

# 범례 수동 구성
from matplotlib.patches import Patch
legend_els = [
    Patch(facecolor='gray', alpha=0.75, label='30% coverage'),
    Patch(facecolor='gray', alpha=0.45, label='50% coverage'),
]
ax2.legend(handles=legend_els, fontsize=10)
ax2.set_xticks(x)
ax2.set_xticklabels(labels_perm, fontsize=9)
ax2.set_ylabel('Runoff Volume Reduction Rate (%)', fontsize=12)
ax2.set_title('(b) Reduction Rate at 30% and 50% Coverage\nby Pavement Permeability Level',
              fontsize=12, fontweight='bold')
ax2.grid(axis='y', linestyle=':', alpha=0.4)
ax2.set_ylim(0, None)

fig.suptitle('Sensitivity of Runoff Reduction to Permeable Pavement Permeability\n'
             '(Pavement Clogging Effect) — Seocho-gu, 10-yr Design Storm',
             fontsize=13, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.88])
fig.savefig(os.path.join(FIG_DIR, 'fig_permeability_sensitivity.png'), facecolor='white')
plt.close(fig)
print("   fig_permeability_sensitivity.png 저장 완료")

# ── 논문용 결과 출력 ───────────────────────────────────────────────────────────
print("\n=== 논문 서술용 요약 ===")
for cov in [30, 50]:
    print(f"\n  커버리지 {cov}%:")
    print(f"  {'투수율':<18} {'저감률(%)':>10}")
    print(f"  {'-'*30}")
    for perm, label, _ in PERM_SCENARIOS:
        val = df[(df['perm_mmhr'] == perm) & (df['lid_rate_pct'] == cov)]['reduction_pct'].values[0]
        ref = df[(df['perm_mmhr'] == 500) & (df['lid_rate_pct'] == cov)]['reduction_pct'].values[0]
        diff = val - ref
        print(f"  {label:<18} {val:>10.1f}%   ({diff:+.1f}%p vs Best)")
