"""Coverage-gap under new models: does the spread signal survive RandomForest? (HSJ black-box)
Reads results.parquet. -> plots/cg_rf_vs_tree.png"""
import os
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_v] = '1'
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, polars as pl
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PLOTS = HERE / 'plots'; PLOTS.mkdir(exist_ok=True)
d = pl.read_parquet(HERE / 'results.parquet').filter(pl.col('defect') == 'coverage_gap')
BIAS = [0.1, 0.3, 0.5, 0.7, 0.9]
MCOL = {'rf': '#1e5eff', 'tree': '#c0392b'}
MLAB = {'rf': 'RandomForest (60 trees)', 'tree': 'single tree'}


def curve(model, col):
    xs, ms, cis = [], [], []
    for b in BIAS:
        v = d.filter((pl.col('model') == model) & (pl.col('bias') == b))[col].drop_nans().drop_nulls()
        xs.append(b); ms.append(float(v.mean()) if len(v) else np.nan)
        cis.append(1.96 * float(v.std()) / np.sqrt(len(v)) if len(v) > 1 else 0)
    return np.array(xs), np.array(ms), np.array(cis)


fig, ax = plt.subplots(figsize=(7.5, 4.6))
for m in ['rf', 'tree']:
    x, y, ci = curve(m, 'mean_dist')
    ax.plot(x, y, marker='o', lw=2, color=MCOL[m], label=MLAB[m] + ' — spread')
    ax.fill_between(x, y - ci, y + ci, color=MCOL[m], alpha=0.15)
ax.set_xlabel('coverage-gap bias (fraction of class deleted)')
ax.set_ylabel('adversarial spread')
ax2 = ax.twinx()
for m in ['rf', 'tree']:
    x, y, _ = curve(m, 'vacc')
    ax2.plot(x, y, ls=':', marker='.', color=MCOL[m], alpha=0.5)
ax2.set_ylabel('test accuracy', color='0.4'); ax2.set_ylim(0.0, 1.03)
ax.set_title('Coverage gap under black-box HSJ: RF spread RISES with bias (signal survives\n'
             'bagging) while accuracy stays flat; single tree+HSJ stays flat (known weak combo)',
             fontsize=9.5)
ax.legend(loc='upper left', fontsize=8)
fig.tight_layout(); fig.savefig(PLOTS / 'cg_rf_vs_tree.png', dpi=140)
print('wrote', PLOTS / 'cg_rf_vs_tree.png')
