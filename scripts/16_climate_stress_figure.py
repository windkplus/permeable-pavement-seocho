"""
16_climate_stress_figure.py
Figure for the multi-return-period + climate-stress sensitivity (Section 4.9).
Panel (a): runoff reduction rate vs coverage for 10/30/50/100-yr design storms,
           with the +30% climate-stress scaling of the 10-yr storm (dashed).
Panel (b): total runoff volume vs coverage for the same storms.
Output: results/figures/fig_climate_stress.png  (white background, 300 dpi)
"""

import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RP_CSV   = os.path.join(BASE_DIR, 'results/simulation_results_return_periods.csv')
CLM_CSV  = os.path.join(BASE_DIR, 'results/simulation_results_climate130.csv')
FIG_DIR  = os.path.join(BASE_DIR, 'results/figures')
os.makedirs(FIG_DIR, exist_ok=True)

matplotlib.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 14, 'axes.titlesize': 16,
    'axes.labelsize': 15, 'xtick.labelsize': 13, 'ytick.labelsize': 13,
    'legend.fontsize': 11, 'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'axes.spines.top': False, 'axes.spines.right': False,
})

# colorblind-safe sequential (viridis-like) for return periods
RP_COLORS = {10: '#2166AC', 30: '#4393C3', 50: '#D6A600', 100: '#B2182B'}
CLM_COLOR = '#000000'

rp = pd.read_csv(RP_CSV)
clm = pd.read_csv(CLM_CSV)
depths = rp.drop_duplicates('return_period_yr').set_index('return_period_yr')['depth_mm'].to_dict()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))

for period in [10, 30, 50, 100]:
    d = rp[rp.return_period_yr == period]
    lab = f'{period}-yr ({depths[period]:.0f} mm)'
    ax1.plot(d['lid_rate_pct'], d['vol_reduction_pct'], 'o-',
             color=RP_COLORS[period], lw=2.2, ms=6, label=lab)
    ax2.plot(d['lid_rate_pct'], d['total_runoff_m3'] / 1e3, 'o-',
             color=RP_COLORS[period], lw=2.2, ms=6, label=lab)
# climate-stress overlay (dashed black)
ax1.plot(clm['lid_rate_pct'], clm['vol_reduction_pct'], 's--', color=CLM_COLOR,
         lw=2.0, ms=6, mfc='white', label='10-yr +30% climate (88 mm)')
ax2.plot(clm['lid_rate_pct'], clm['total_runoff_m3'] / 1e3, 's--', color=CLM_COLOR,
         lw=2.0, ms=6, mfc='white', label='10-yr +30% climate (88 mm)')

ax1.axvspan(30, 50, color='grey', alpha=0.12)
ax1.set_xlabel('PP coverage rate (% of impervious area)')
ax1.set_ylabel('Runoff volume reduction (%)')
ax1.set_title('(a) Relative reduction by return period')
ax1.set_xticks(range(0, 101, 20))
ax1.legend(frameon=False, loc='lower right')
ax1.grid(alpha=0.25)

ax2.set_xlabel('PP coverage rate (% of impervious area)')
ax2.set_ylabel('Total runoff volume (10³ m³)')
ax2.set_title('(b) Total runoff volume by return period')
ax2.set_xticks(range(0, 101, 20))
ax2.legend(frameon=False, loc='upper right')
ax2.grid(alpha=0.25)

fig.tight_layout()
out = os.path.join(FIG_DIR, 'fig_climate_stress.png')
fig.savefig(out)
print('saved ->', out)
