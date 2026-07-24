"""
20_run_manifest.py
Generate a machine-readable manifest of every SWMM model execution in the study
  (manuscript Table 15; Reviewer #2 third-round Comment 9)

The generated .inp files are regenerable intermediates and are excluded from the
archive by .gitignore. This script records, for each run, the block it belongs to,
the script that generates it, the .inp filename, and a SHA-256 checksum of the file
as generated locally — so that the run accounting in Table 15 is independently
checkable and any re-run can be verified against the original.

Output
  - results/swmm_run_manifest.csv   (one row per model execution)
  - results/swmm_run_summary.csv    (per-block totals = manuscript Table 15)
"""

import os, glob, hashlib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
OUT_MANIFEST = os.path.join(BASE_DIR, 'results/swmm_run_manifest.csv')
OUT_SUMMARY  = os.path.join(BASE_DIR, 'results/swmm_run_summary.csv')

# (directory, block label, manuscript section, generating script, core?)
BLOCKS = [
    ('surrogate_scenarios',      'LHS surrogate training (200 LHS + 1 zero-coverage baseline)',
     '3.8.1', '05_surrogate_training.py',      True),
    ('scenarios',                'Uniform-coverage sweep (0-100 %, 11 steps)',
     '3.7, 4.3', '03_run_simulation.py',       True),
    ('pareto_validation',        'Pareto-solution verification (10 representative solutions)',
     '4.7.1', '06b_pareto_validation.py',      True),
    ('scenarios_return_periods', 'Multi-return-period sweep (4 storms x 11 coverages)',
     '4.9', '17_multi_return_period.py',       False),
    ('scenarios_climate130',     'Climate-stress sweep (+30 % scaled 10-year storm)',
     '4.9', '15_climate_stress_scenario.py',   False),
    ('perm_sensitivity',         'Pavement-permeability (clogging) sweep',
     '5.6', '10_permeability_sensitivity.py',  False),
    ('resample_scenarios',       'Targeted resampling in the low-coverage region',
     '4.7.2', '18_targeted_resampling.py',     False),
    ('midperm_scenarios',        'Reduced-permeability re-optimization (LHS re-run at 100 mm/hr)',
     '4.7.3', '19_clogged_optimization.py',    False),
]


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


rows, summary = [], []
for subdir, label, section, script, is_core in BLOCKS:
    paths = sorted(glob.glob(os.path.join(MODEL_DIR, subdir, '*.inp')))
    if not paths:
        print(f"  WARNING: no .inp files found in model/{subdir} - run {script} first")
    for p in paths:
        rows.append({
            'block'      : label,
            'section'    : section,
            'script'     : script,
            'workflow'   : 'core' if is_core else 'sensitivity',
            'inp_relpath': os.path.relpath(p, BASE_DIR).replace('\\', '/'),
            'size_bytes' : os.path.getsize(p),
            'sha256'     : sha256(p),
        })
    summary.append({'block': label, 'section': section, 'script': script,
                    'workflow': 'core' if is_core else 'sensitivity',
                    'runs': len(paths)})

df = pd.DataFrame(rows)
df.to_csv(OUT_MANIFEST, index=False, encoding='utf-8-sig')

df_sum = pd.DataFrame(summary)
core = df_sum.loc[df_sum.workflow == 'core', 'runs'].sum()
sens = df_sum.loc[df_sum.workflow == 'sensitivity', 'runs'].sum()
df_sum = pd.concat([df_sum, pd.DataFrame([
    {'block': 'CORE surrogate-development and Pareto-verification workflow',
     'section': '', 'script': '', 'workflow': 'subtotal', 'runs': core},
    {'block': 'SENSITIVITY and re-optimization analyses',
     'section': '', 'script': '', 'workflow': 'subtotal', 'runs': sens},
    {'block': 'TOTAL SWMM model executions',
     'section': '', 'script': '', 'workflow': 'total', 'runs': core + sens},
])], ignore_index=True)
df_sum.to_csv(OUT_SUMMARY, index=False, encoding='utf-8-sig')

print(df_sum[['block', 'workflow', 'runs']].to_string(index=False))
print(f"\n  manifest ({len(df)} runs) -> {OUT_MANIFEST}")
print(f"  summary                  -> {OUT_SUMMARY}")
