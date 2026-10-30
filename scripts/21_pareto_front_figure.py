"""
21_pareto_front_figure.py
Fourth revision (R4): re-draws manuscript Figure 9 from archived outputs only
(no SWMM or NSGA-II re-run), adding the 10 directly SWMM-verified Pareto
solutions so that the surrogate-predicted front and the verified values are
shown together, alongside the uniform-coverage scenarios.

Inputs : results/pareto_solutions.csv          (06_spatial_optimization.py)
         results/simulation_results.csv        (03_run_simulation.py)
         results/pareto_validation_results.csv (06b_pareto_validation.py)
Output : results/figures/fig_pareto_front.png
"""
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE_DIR, 'results')
FIG_DIR = os.path.join(RES, 'figures')

plt.rcParams.update({'font.family': 'DejaVu Sans', 'savefig.dpi': 300})

df_pareto = pd.read_csv(os.path.join(RES, 'pareto_solutions.csv'))
df_unif = pd.read_csv(os.path.join(RES, 'simulation_results.csv'))
df_val = pd.read_csv(os.path.join(RES, 'pareto_validation_results.csv'))

fig, ax = plt.subplots(figsize=(9, 6), facecolor='white')

sc = ax.scatter(df_pareto['lid_area_pct'], df_pareto['runoff_red_pct'],
                c=df_pareto['runoff_red_pct'], cmap='Blues', vmin=10,
                s=40, alpha=0.8, edgecolors='none', zorder=2,
                label='Pareto front (surrogate-predicted)')

# uniform coverage: installation area (% of max. impervious) equals the coverage rate
ax.plot(df_unif['lid_rate_pct'], df_unif['vol_reduction_pct'], '--', color='#D6604D',
        linewidth=1.0, zorder=1)
ax.scatter(df_unif['lid_rate_pct'], df_unif['vol_reduction_pct'], marker='D', s=55,
           color='#D6604D', edgecolors='white', linewidth=0.5, zorder=3,
           label='Uniform coverage (direct SWMM)')

# directly verified Pareto solutions: surrogate value -> SWMM value
for _, r in df_val.iterrows():
    ax.annotate('', xy=(r['lid_area_pct'], r['runoff_red_pct_swmm']),
                xytext=(r['lid_area_pct'], r['runoff_red_pct_surrogate']),
                arrowprops=dict(arrowstyle='->', color='0.35', lw=0.9), zorder=4)
ax.scatter(df_val['lid_area_pct'], df_val['runoff_red_pct_swmm'], marker='o', s=70,
           facecolors='none', edgecolors='black', linewidth=1.4, zorder=5,
           label='Verified Pareto solutions (direct SWMM)')

s1 = df_val[df_val['label'].str.startswith('S1')].iloc[0]
s2 = df_val[df_val['label'].str.startswith('S2')].iloc[0]
ax.scatter(s1['lid_area_pct'], s1['runoff_red_pct_surrogate'], marker='*', s=320,
           color='#FFD700', edgecolors='black', linewidth=1.2, zorder=6,
           label='S1 (max. reduction)')
ax.scatter(s2['lid_area_pct'], s2['runoff_red_pct_surrogate'], marker='s', s=140,
           color='#4DAC26', edgecolors='black', linewidth=1.2, zorder=6,
           label='S2 = S3 (min. area / balanced)')

plt.colorbar(sc, ax=ax, label='Surrogate-predicted reduction (%)', pad=0.02)
ax.legend(fontsize=10, loc='lower right')
ax.set_xlabel('LID installation area (% of max. available impervious area)', fontsize=13)
ax.set_ylabel('Runoff volume reduction rate (%)', fontsize=13)
ax.set_xlim(-3, 103)
ax.set_ylim(-1, 27)
ax.grid(linestyle=':', alpha=0.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig_pareto_front.png'), facecolor='white')
print('saved results/figures/fig_pareto_front.png')
