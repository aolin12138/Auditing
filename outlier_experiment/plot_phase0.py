"""Phase 0 figures for the outlier experiment (reads results.parquet).
  1. phase0_dose_response.png  — spread vs k and vs n_out (toward vs outward),
     baseline reference, accuracy twin axis (shows spread rises while accuracy flat).
  2. phase0_interaction_heatmap.png — k x n_out spread (toward direction).
Usage: python plot_phase0.py
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
df = pl.read_parquet(HERE / 'results.parquet')

base = df.filter(pl.col('n_out') == 0)
base_spread = float(base['mean_dist'].mean())
base_vacc = float(base['vacc'].mean())
prod = df.filter(pl.col('n_out') > 0)
COL = {'toward': '#1e5eff', 'outward': '#c0392b'}


def curve(by, direction):
    return (prod.filter(pl.col('direction') == direction)
            .group_by(by).agg(pl.col('mean_dist').mean().alias('spread'),
                              pl.col('mean_dist').std().alias('sd'),
                              pl.col('vacc').mean().alias('vacc')).sort(by))


# ---- Figure 1: dose-response + accuracy twin ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
for ax, by, xlabel in [(axes[0], 'k', 'outlier distance  k  (x class std)'),
                       (axes[1], 'n_out', 'number of outliers  n_out')]:
    for direction in ['toward', 'outward']:
        g = curve(by, direction)
        x = g[by].to_numpy(); y = g['spread'].to_numpy(); sd = g['sd'].to_numpy()
        ax.errorbar(x, y, yerr=sd, marker='o', capsize=3, lw=2, color=COL[direction],
                    label=f'{direction}')
    ax.axhline(base_spread, ls='--', color='0.5', lw=1.2, label=f'clean baseline ({base_spread:.2f})')
    ax.set_xlabel(xlabel); ax.set_ylabel('adversarial spread')
    ax.set_title(f'spread vs {by}')
    ax2 = ax.twinx()                                   # accuracy, to show it stays flat
    gt = curve(by, 'toward')
    ax2.plot(gt[by].to_numpy(), gt['vacc'].to_numpy(), color='green', alpha=0.45, ls=':',
             marker='.', label='test accuracy (toward)')
    ax2.axhline(base_vacc, ls=':', color='green', alpha=0.3)
    ax2.set_ylabel('test accuracy', color='green'); ax2.set_ylim(0.0, 1.03)
    ax2.tick_params(axis='y', colors='green')
axes[0].legend(fontsize=8, loc='center left')
fig.suptitle('Outlier Phase 0: spread rises with severity for TOWARD outliers while accuracy stays flat; '
             'OUTWARD is null (tree is blind to it)', fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(PLOTS / 'phase0_dose_response.png', dpi=140)
plt.close(fig)

# ---- Figure 2: k x n_out interaction heatmap (toward) ----
ks, ns = [2, 4, 6, 8], [1, 3, 5, 10]
piv = (prod.filter(pl.col('direction') == 'toward')
       .group_by(['k', 'n_out']).agg(pl.col('mean_dist').mean().alias('spread')))
M = np.full((len(ks), len(ns)), np.nan)
for r in piv.iter_rows(named=True):
    M[ks.index(int(r['k'])), ns.index(int(r['n_out']))] = r['spread']
fig, ax = plt.subplots(figsize=(5.4, 4.2))
im = ax.imshow(M, cmap='viridis', aspect='auto', origin='lower')
ax.set_xticks(range(len(ns))); ax.set_xticklabels(ns)
ax.set_yticks(range(len(ks))); ax.set_yticklabels(ks)
ax.set_xlabel('number of outliers  n_out'); ax.set_ylabel('distance  k  (x class std)')
mid = np.nanmean(M)
for i in range(len(ks)):
    for j in range(len(ns)):
        if np.isfinite(M[i, j]):
            ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center',
                    color='white' if M[i, j] < mid else 'black', fontsize=9)
fig.colorbar(im, label='adversarial spread (toward)')
ax.set_title(f'Outlier Phase 0: k x n_out interaction (toward)\nclean baseline = {base_spread:.2f}', fontsize=10)
fig.tight_layout()
fig.savefig(PLOTS / 'phase0_interaction_heatmap.png', dpi=140)
plt.close(fig)

print('wrote:')
for p in sorted(PLOTS.glob('phase0_*.png')):
    print(' ', p.name, f'({p.stat().st_size} bytes)')
