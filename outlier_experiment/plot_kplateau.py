"""k-plateau plot: full distance sweep to k=24 (50 seeds) showing onset, plateau, COLLAPSE.
-> plots/kplateau.png
"""
import os
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_v] = '1'
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, polars as pl
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PLOTS = HERE / 'plots'
d = pl.read_parquet(HERE / 'results_kplateau.parquet')
BASE = {'tree_d3': 0.448, 'tree_d10': 0.440}     # established clean baselines (n_out=0)

fig, ax = plt.subplots(figsize=(9, 5))
for model, col, lab in [('tree_d3', '#e08214', 'tree d3 + DTA'), ('tree_d10', '#1e5eff', 'tree d10 + DTA')]:
    ks = sorted(d['k'].unique().to_list())
    xs, ms, cis = [], [], []
    for k in ks:
        v = d.filter((pl.col('model') == model) & (pl.col('k') == float(k)))['mean_dist'].drop_nans().drop_nulls()
        xs.append(k); ms.append(float(v.mean()) / BASE[model])
        cis.append(1.96 * float(v.std()) / np.sqrt(len(v)) / BASE[model])
    ax.errorbar(xs, ms, yerr=cis, marker='o', lw=2.2, capsize=3, color=col, label=lab)
ax.axhline(1.0, ls=':', color='0.4', lw=1.2, label='baseline (no outlier)')
ax.axvspan(2.0, 5.3, color='#ff7f0e', alpha=0.06)     # versicolor cluster along the ray
ax.axvspan(9.5, 14.5, color='#1f77b4', alpha=0.06)    # setosa territory along the ray
ax.text(3.7, 0.96, 'outlier inside\nversicolor', fontsize=7.5, color='#b06000', ha='center')
ax.text(12, 0.96, 'outlier inside\nsetosa', fontsize=7.5, color='#1f5a9e', ha='center')
ax.text(7.2, 1.33, 'SIGNAL WINDOW\n(empty gap between clusters)', fontsize=8, color='#2ca25f', ha='center')
ax.set_xlabel('k  =  distance to own centroid (x class std)')
ax.set_ylabel('normalised adversarial spread (x baseline)')
ax.set_title('Full distance sweep, 50 seeds: the signal is NON-MONOTONE —\n'
             'fires only while the outlier sits in the GAP between versicolor and setosa,\n'
             'collapses at k=12 when it enters setosa territory (tight CIs, not noise)', fontsize=10)
ax.legend(fontsize=8.5, loc='center right')
ax.set_xlim(0, 25)
fig.tight_layout()
fig.savefig(PLOTS / 'kplateau.png', dpi=140)
print('wrote', PLOTS / 'kplateau.png')
