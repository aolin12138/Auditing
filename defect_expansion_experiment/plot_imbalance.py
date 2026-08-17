"""Phase 0 class-imbalance vs coverage-gap control (tree + DTA, 30 seeds).
  Panel A: normalised spread — spatial (coverage-gap) vs random (imbalance) [H1].
  Panel B: minority-class recall + overall accuracy — the clean discriminator [H2].
-> plots/imbalance/phase0_spread_vs_random.png
"""
import os
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_v] = '1'
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, polars as pl
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotstyle as ps; ps.apply()

HERE = Path(__file__).resolve().parent
PLOTS = HERE / 'plots'; PLOTS.mkdir(exist_ok=True); (PLOTS / 'imbalance').mkdir(exist_ok=True)
d = pl.read_parquet(HERE / 'results_imbalance.parquet')
base = d.filter(pl.col('frac') == 0.0)['mean_dist'].drop_nans().mean()
FRAC = [0.0, 0.25, 0.5, 0.7, 0.85, 0.95]
COL = ps.DEFECT                      # colorblind-safe (Okabe-Ito)
LAB = {'random': 'random deletion (imbalance)', 'spatial': 'spatial deletion (coverage gap)'}


def series(structure, col, norm=False):
    xs, ms, cis = [], [], []
    for f in FRAC:
        sub = d.filter((pl.col('frac') == f) & ((pl.col('structure') == structure) | (pl.col('frac') == 0.0)))
        v = sub[col].drop_nans()
        if len(v):
            xs.append(f); m = float(v.mean()); ms.append(m / base if norm else m)
            cis.append(1.96 * float(v.std()) / np.sqrt(len(v)) / (base if norm else 1))
    return np.array(xs), np.array(ms), np.array(cis)


fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7))

ax = axes[0]
for s in ['random', 'spatial']:
    x, m, ci = series(s, 'mean_dist', norm=True)
    ax.plot(x, m, marker=ps.DEFECT_MK[s], ls=ps.DEFECT_LS[s], color=COL[s], label=LAB[s])
    ax.fill_between(x, m - ci, m + ci, color=COL[s], alpha=0.15)
ax.axhline(1.0, ls=':', color='0.4', lw=1)
ax.set_xlabel('fraction of class removed (matched count both arms)')
ax.set_ylabel('normalised adversarial spread (x baseline)')
ax.set_title('H1: the spatial HOLE adds spread on top of the count effect\n'
             '(both rise; spatial > random, CIs separate at frac>=0.5)', fontsize=9.5)
ax.legend(fontsize=8, loc='upper left')

ax = axes[1]
for s in ['random', 'spatial']:
    x, m, ci = series(s, 'min_recall')
    ax.plot(x, m, marker=ps.DEFECT_MK[s], ls=ps.DEFECT_LS[s], color=COL[s], label=LAB[s] + ' — minority recall')
    ax.fill_between(x, m - ci, m + ci, color=COL[s], alpha=0.15)
    xa, ma, _ = series(s, 'vacc')
    ax.plot(xa, ma, ls='--', marker='.', lw=1.4, color=COL[s], alpha=0.6, label=LAB[s] + ' — overall acc')
ax.set_xlabel('fraction of class removed')
ax.set_ylabel('accuracy / recall')
ax.set_ylim(0, 1.02)
ax.set_title('H2: minority-class RECALL is the clean discriminator\n'
             '(coverage gap craters it to 0.14; imbalance holds ~0.61)', fontsize=9.5)
ax.legend(fontsize=7, loc='lower left')

fig.suptitle('Phase 0 — class imbalance vs coverage gap (overfit tree + white-box DTA, 30 seeds)', fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93])
ps.save(fig, PLOTS / 'imbalance' / 'phase0_spread_vs_random')
print('wrote', PLOTS / 'imbalance/phase0_spread_vs_random')
