"""Threshold figures (reads results_threshold.parquet):
  1. normalised spread vs n_out% for toward/outward/random (shows toward peak-then-collapse).
  2. train AND test accuracy vs n_out% (shows accuracy is blind even at 200% outliers).
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
df = pl.read_parquet(HERE / 'results_threshold.parquet')
base_spread = float(df.filter(pl.col('pct') == 0)['mean_dist'].mean())
DIRS = ['toward', 'outward', 'random']
COL = {'toward': '#c0392b', 'outward': '#8e44ad', 'random': '#16a085'}
PCTS = [5, 10, 25, 50, 75, 100, 150, 200]


def series(direction, col):
    xs, ms, cis = [], [], []
    for pct in PCTS:
        d = df.filter((pl.col('direction') == direction) & (pl.col('pct') == pct))
        v = d[col].to_numpy()
        xs.append(pct); ms.append(np.nanmean(v))
        cis.append(1.96 * np.nanstd(v) / np.sqrt(np.sum(~np.isnan(v))))
    return np.array(xs), np.array(ms), np.array(cis)


fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Panel 1: normalised spread
ax = axes[0]
for d in DIRS:
    x, m, ci = series(d, 'mean_dist')
    m, ci = m / base_spread, ci / base_spread
    ax.plot(x, m, marker='o', color=COL[d], lw=2, label=d)
    ax.fill_between(x, m - ci, m + ci, color=COL[d], alpha=0.15)
ax.axhline(1.0, ls='--', color='0.4', lw=1.2, label='clean baseline (=1.0)')
ax.axvline(100, ls=':', color='0.6', lw=1)
ax.set_xlabel('outliers as % of class size   [k fixed = 8]')
ax.set_ylabel('normalised spread  (x baseline)')
ax.set_title('Spread vs outlier count: TOWARD peaks then collapses at ~100%\n'
             '(outliers become their own blob); random weak; outward null', fontsize=9.5)
ax.legend(fontsize=8)

# Panel 2: train + test accuracy
ax = axes[1]
for d in DIRS:
    x, mt, cit = series(d, 'vacc')
    ax.plot(x, mt, marker='o', color=COL[d], lw=2, label=f'{d} (test)')
    ax.fill_between(x, mt - cit, mt + cit, color=COL[d], alpha=0.12)
xtr, mtr, _ = series('toward', 'tacc')
ax.plot(xtr, mtr, ls='--', color='0.3', lw=2, label='train acc (all dirs =1.0)')
ax.axhline(1/3, ls=':', color='r', lw=1, label='chance (1/3)')
ax.set_xlabel('outliers as % of class size')
ax.set_ylabel('accuracy'); ax.set_ylim(0.25, 1.03)
ax.set_title('Accuracy is blind: train=1.0 (memorised), test ~0.95\n'
             'flat even at 200% outliers (2x the real class)', fontsize=9.5)
ax.legend(fontsize=7, loc='center right')

fig.suptitle('Outlier count threshold sweep (overfit tree + white-box DTA, 10 seeds, 95% CI)', fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(PLOTS / 'phase_threshold.png', dpi=140)
print('wrote', PLOTS / 'phase_threshold.png', flush=True)
