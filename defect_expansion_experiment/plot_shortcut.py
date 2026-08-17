"""Phase 0 shortcut (Clever Hans) figures.

Per dataset (iris / wine), 4 rows x 2 cols:
  rows = [test accuracy, first-spurious-split depth, spurious-axis fraction of adv L2
          displacement, mean adv L2 displacement]
  x = corr (shortcut strength), 30 seeds, 95% CI bands.
Answers: as the shortcut strengthens — does test accuracy collapse (Clever Hans cost)? does
the tree lean on the shortcut (split depth -> root)? does the adversarial displacement
concentrate on the spurious axis (H1)? how big are the escapes?

-> plots/shortcut/phase0.png  (PNG + vector PDF)
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
PLOTS = HERE / 'plots'; PLOTS.mkdir(exist_ok=True); (PLOTS / 'shortcut').mkdir(exist_ok=True)
CORR = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
X = np.arange(len(CORR))
ROWS = [('vacc', 'test accuracy\n(Clever Hans cost)', (0.3, 1.0)),
        ('spur_depth', 'depth of first spurious split\n(0 = root; low = reliance)', (0, 3.5)),
        ('spur_frac', 'spurious-axis fraction of adv L2\ndisplacement (H1)', (0, 1.05)),
        ('adv_l2', 'mean adversarial L2 displacement', (0.5, 4.5))]
DS_TITLE = {'iris': 'iris (4-D)', 'wine': 'wine (13-D)'}
COL = ps.OKABE_ITO


def series(df, ds, col):
    xs, ms, cis = [], [], []
    for c in CORR:
        v = df.filter((pl.col('dataset') == ds) & (pl.col('corr') == c))[col].drop_nans()
        m, n = float(v.mean()), len(v)
        se = 1.96 * float(v.std()) / np.sqrt(n) if n > 1 else 0.0
        xs.append(c); ms.append(m); cis.append(se)
    return xs, np.array(ms), np.array(cis)


def main():
    df = pl.read_parquet(HERE / 'results_shortcut.parquet')
    fig, axes = plt.subplots(4, 2, figsize=(12.5, 12.5))
    for i, (col, ylab, ylim) in enumerate(ROWS):
        for j, ds in enumerate(['iris', 'wine']):
            ax = axes[i][j]
            xs, m, ci = series(df, ds, col)
            ax.plot(X, m, marker='o', color=COL['vermillion'], lw=2)
            ax.fill_between(X, m - ci, m + ci, color=COL['vermillion'], alpha=0.15)
            ax.set_xticks(X); ax.set_xticklabels([str(c) for c in CORR])
            if ylim:
                ax.set_ylim(*ylim)
            ps.panel_label(ax, 'abcdefgh'[i * 2 + j])
            if i == 0:
                ax.set_title(DS_TITLE[ds], fontsize=10)
            if j == 0:
                ax.set_ylabel(ylab, fontsize=9)
            if i == 3:
                ax.set_xlabel('shortcut strength corr (train signal / noise sigma)')
    fig.suptitle('Clever Hans / spurious feature — tree + white-box DTA, 30 seeds, train-only injection\n'
                 'as the shortcut strengthens: test accuracy collapses, the first spurious split rises to the\n'
                 'root, and adversarial displacement concentrates on the spurious axis', fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = PLOTS / 'shortcut' / 'phase0'; ps.save(fig, p); plt.close(fig)
    print('wrote', p)


if __name__ == '__main__':
    main()
