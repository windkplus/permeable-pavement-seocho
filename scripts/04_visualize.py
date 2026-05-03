"""
04_visualize.py
논문용 결과 시각화 (흰색 배경, 영문 레이블)
출력: results/figures/
  - fig1_runoff_reduction.png  : 적용률 vs 유출저감율 (메인)
  - fig2_total_runoff_bar.png  : 시나리오별 총유출량 비교
  - fig3_combined.png          : 두 그래프 합본 (논문 Figure)
"""

import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV   = os.path.join(BASE_DIR, 'results/simulation_results.csv')
FIG_DIR     = os.path.join(BASE_DIR, 'results/figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ── 전역 스타일 설정 ───────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'font.size'        : 14,
    'axes.titlesize'   : 16,
    'axes.labelsize'   : 15,
    'xtick.labelsize'  : 13,
    'ytick.labelsize'  : 13,
    'legend.fontsize'  : 13,
    'figure.dpi'       : 150,
    'savefig.dpi'      : 300,
    'savefig.bbox'     : 'tight',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
})

BLUE   = '#2166AC'
ORANGE = '#D6604D'
GREEN  = '#4DAC26'
GRAY   = '#888888'

# ── 데이터 로드 ────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
rates     = df['lid_rate_pct'].values
vol_m3    = df['total_runoff_m3'].values
red_m3    = df['vol_reduction_m3'].values
red_pct   = df['vol_reduction_pct'].values
peak_red  = df['peak_reduction_pct'].values
baseline  = vol_m3[0]

# ── Figure 1: 적용률 vs 유출저감율 ────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(9, 6), facecolor='white')
ax1.set_facecolor('white')

# 저감율 곡선
ax1.plot(rates, red_pct, 'o-', color=BLUE, linewidth=2.5,
         markersize=9, markerfacecolor='white', markeredgewidth=2.5,
         label='Runoff Volume Reduction Rate (%)', zorder=3)

# 각 점에 값 표시
for x, y in zip(rates, red_pct):
    if x in [0, 20, 50, 100]:
        offset = 1.5 if y < 20 else -2.0
        ax1.annotate(f'{y:.1f}%',
                     xy=(x, y), xytext=(x + 1.5, y + offset),
                     fontsize=12, color=BLUE, fontweight='bold')

# 음영 채우기
ax1.fill_between(rates, red_pct, alpha=0.12, color=BLUE)

# 점선 참조선 (20%, 50%)
for ref_x in [20, 50]:
    ref_y = red_pct[rates == ref_x][0]
    ax1.axvline(ref_x, color=GRAY, linestyle='--', linewidth=1.2, alpha=0.6)
    ax1.axhline(ref_y, color=GRAY, linestyle='--', linewidth=1.2, alpha=0.6)

ax1.set_xlim(-2, 105)
ax1.set_ylim(-1, 30)
ax1.set_xlabel('Permeable Pavement Coverage Rate (%)', fontsize=15, labelpad=8)
ax1.set_ylabel('Runoff Volume Reduction Rate (%)', fontsize=15, labelpad=8)
ax1.set_title('Effect of Permeable Pavement Coverage on\nStormwater Runoff Reduction in Seocho-gu, Seoul',
              fontsize=16, fontweight='bold', pad=14)
ax1.set_xticks(rates)
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
ax1.grid(axis='y', linestyle=':', linewidth=1.0, alpha=0.6)

# 텍스트 박스: 설계조건
info = ('Design Storm: 10-yr Return Period\n'
        'Duration: 60 min  |  Total Depth: 67.6 mm\n'
        'Rainfall Distribution: Huff 4th Quartile')
ax1.text(0.98, 0.05, info, transform=ax1.transAxes,
         fontsize=11, verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                   edgecolor='#CCCC99', alpha=0.9))

ax1.legend(loc='upper left', frameon=True, framealpha=0.9)
fig1.tight_layout()
fig1.savefig(os.path.join(FIG_DIR, 'fig1_runoff_reduction.png'))
plt.close(fig1)
print("fig1 저장 완료")

# ── Figure 2: 시나리오별 총유출량 비교 (막대 그래프) ──────────────────────────
fig2, ax2 = plt.subplots(figsize=(11, 6), facecolor='white')
ax2.set_facecolor('white')

colors = [ORANGE if r == 0 else BLUE for r in rates]
bars = ax2.bar(rates, vol_m3 / 1e6, width=7, color=colors,
               edgecolor='white', linewidth=0.8, zorder=2)

# 막대 위 값 표시
for bar, v, r in zip(bars, vol_m3, rates):
    if r in [0, 20, 50, 100]:
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005,
                 f'{v/1e6:.3f}',
                 ha='center', va='bottom', fontsize=11,
                 color='black', fontweight='bold')

# Baseline 기준선
ax2.axhline(baseline / 1e6, color=ORANGE, linestyle='--',
            linewidth=1.8, alpha=0.7, label=f'Baseline: {baseline/1e6:.3f} M m³')

ax2.set_xlim(-7, 107)
ax2.set_ylim(0, baseline / 1e6 * 1.18)
ax2.set_xlabel('Permeable Pavement Coverage Rate (%)', fontsize=15, labelpad=8)
ax2.set_ylabel('Total Runoff Volume (million m³)', fontsize=15, labelpad=8)
ax2.set_title('Total Stormwater Runoff Volume by\nPermeable Pavement Coverage Scenario',
              fontsize=16, fontweight='bold', pad=14)
ax2.set_xticks(rates)
ax2.grid(axis='y', linestyle=':', linewidth=1.0, alpha=0.6)
ax2.legend(fontsize=13)

# 범례용 패치
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=ORANGE, label='Baseline (No LID)'),
    Patch(facecolor=BLUE,   label='With Permeable Pavement'),
]
ax2.legend(handles=legend_elements, fontsize=13)

fig2.tight_layout()
fig2.savefig(os.path.join(FIG_DIR, 'fig2_total_runoff_bar.png'))
plt.close(fig2)
print("fig2 저장 완료")

# ── Figure 3: 합본 (2-panel, 논문 제출용) ─────────────────────────────────────
fig3, (axA, axB) = plt.subplots(1, 2, figsize=(16, 7), facecolor='white')
fig3.subplots_adjust(wspace=0.35, top=0.79)

# ── Panel (a): 저감율 곡선 ──────────────────────────────────────────────────
axA.set_facecolor('white')
axA.plot(rates, red_pct, 'o-', color=BLUE, linewidth=2.5,
         markersize=9, markerfacecolor='white', markeredgewidth=2.5)
axA.fill_between(rates, red_pct, alpha=0.12, color=BLUE)

for ref_x in [20, 50]:
    ref_y = red_pct[rates == ref_x][0]
    axA.axvline(ref_x, color=GRAY, linestyle='--', linewidth=1.2, alpha=0.6)
    axA.axhline(ref_y, color=GRAY, linestyle='--', linewidth=1.2, alpha=0.6)

for x, y in zip(rates, red_pct):
    if x in [0, 30, 60, 100]:
        axA.annotate(f'{y:.1f}%',
                     xy=(x, y), xytext=(x + 2, y + 1.2),
                     fontsize=11, color=BLUE, fontweight='bold')

axA.set_xlim(-2, 105)
axA.set_ylim(-1, 30)
axA.set_xlabel('Coverage Rate (%)', fontsize=14, labelpad=8)
axA.set_ylabel('Runoff Volume Reduction Rate (%)', fontsize=14, labelpad=8)
axA.set_title('(a) Runoff Reduction Rate vs.\nCoverage Rate', fontsize=15, fontweight='bold', pad=12)
axA.set_xticks(rates)
axA.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
axA.grid(axis='y', linestyle=':', linewidth=1.0, alpha=0.6)

# ── Panel (b): 총유출량 막대 ────────────────────────────────────────────────
axB.set_facecolor('white')
colors_b = [ORANGE if r == 0 else BLUE for r in rates]
axB.bar(rates, vol_m3 / 1e6, width=7, color=colors_b,
        edgecolor='white', linewidth=0.8, zorder=2)
axB.axhline(baseline / 1e6, color=ORANGE, linestyle='--',
            linewidth=1.8, alpha=0.7)

axB.set_xlim(-7, 107)
axB.set_ylim(0, baseline / 1e6 * 1.15)
axB.set_xlabel('Coverage Rate (%)', fontsize=14, labelpad=8)
axB.set_ylabel('Total Runoff Volume (million m³)', fontsize=14, labelpad=8)
axB.set_title('(b) Total Runoff Volume by\nCoverage Scenario', fontsize=15, fontweight='bold', pad=12)
axB.set_xticks(rates)
axB.grid(axis='y', linestyle=':', linewidth=1.0, alpha=0.6)

legend_b = [Patch(facecolor=ORANGE, label='Baseline (No LID)'),
            Patch(facecolor=BLUE,   label='With Permeable Pavement')]
axB.legend(handles=legend_b, fontsize=12, loc='upper right')

# ── 공통 제목 ──────────────────────────────────────────────────────────────
fig3.suptitle(
    'Stormwater Runoff Reduction by Permeable Pavement LID in Seocho-gu, Seoul\n'
    'Design Storm: 10-yr Return Period, 60-min Duration, Huff 4th Quartile (67.6 mm)',
    fontsize=14, fontweight='bold', y=0.98
)

fig3.savefig(os.path.join(FIG_DIR, 'fig3_combined.png'),
             bbox_inches='tight', facecolor='white')
plt.close(fig3)
print("fig3 저장 완료")

# ── 결과 테이블 출력 ───────────────────────────────────────────────────────────
print("\n=== Simulation Results Summary ===")
print(f"{'Coverage':>10} {'Total Runoff':>15} {'Reduction':>12} {'Red. Rate':>11}")
print(f"{'Rate (%)':>10} {'Volume (m³)':>15} {'Volume (m³)':>12} {'(%)':>11}")
print("-" * 52)
for _, r in df.iterrows():
    print(f"{r['lid_rate_pct']:>9.0f}%  "
          f"{r['total_runoff_m3']:>14,.0f}  "
          f"{r['vol_reduction_m3']:>11,.0f}  "
          f"{r['vol_reduction_pct']:>10.1f}%")
print("-" * 52)
print(f"\nFigures saved to: {FIG_DIR}")
print("  - fig1_runoff_reduction.png")
print("  - fig2_total_runoff_bar.png")
print("  - fig3_combined.png  ← 논문 제출용")
