"""Phase 0 leakage figure — 3x2 grid, x = n_leak (leaked train points copied into test).

rows = [accuracy: contaminated vacc vs honest vacc_clean (H2),
        memorization locality: margin_leak vs margin_clean (H1),
        global adversarial spread m0 (composition reference)]
cols = iris / wine. 30 seeds, 95% CI. -> plots/leakage/phase0.png
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
PLOTS = HERE / 'plots'; PLOTS.mkdir(exist_ok=True); (PLOTS / 'leakage').mkdir(exist_ok=True)
N_LEAK = [0, 2, 5, 10, 20]
X = np.arange(len(N_LEAK))
COL = ps.OKABE_ITO
DS_TITLE = {'iris': 'iris (4-D)', 'wine': 'wine (13-D)'}
# (column pair, labels, colors, linestyles) per row
ROWS = [(('vacc', 'vacc_clean'), ('contaminated (measured)', 'clean (honest)'),
         'test accuracy', (0.85, 1.0)),
        (('margin_leak', 'margin_clean'), ('leaked points', 'clean points'),
         'adversarial margin\n(perturbation L2)', None),
        (('spread_m0', None), ('global spread', None),
         'global adversarial spread (m0)', None)]


def series(df, ds, col):
    xs, ms, cis = [], [], []
    for nl in N_LEAK:
        v = df.filter((pl.col('dataset') == ds) & (pl.col('n_leak') == nl))[col].drop_nans()
        if len(v):
            m, n = float(v.mean()), len(v)
            se = 1.96 * float(v.std()) / np.sqrt(n) if n > 1 else 0.0
            xs.append(N_LEAK.index(nl)); ms.append(m); cis.append(se)
    return np.array(xs), np.array(ms), np.array(cis)


def main():
    df = pl.read_parquet(HERE / 'results_leakage.parquet')
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 10.5))
    colors = [COL['vermillion'], COL['blue']]
    styles = ['-', '--']
    for i, ((c0, c1), (l0, l1), ylab, ylim) in enumerate(ROWS):
        for j, ds in enumerate(['iris', 'wine']):
            ax = axes[i][j]
            for col, lab, cc, ls in [(c0, l0, colors[0], styles[0]), (c1, l1, colors[1], styles[1])]:
                if col is None:
                    continue
                x, m, ci = series(df, ds, col)
                ax.plot(x, m, marker='o', ls=ls, color=cc, lw=2, label=lab)
                ax.fill_between(x, m - ci, m + ci, color=cc, alpha=0.15)
            ax.set_xticks(X); ax.set_xticklabels([str(n) for n in N_LEAK])
            if ylim:
                ax.set_ylim(*ylim)
            ps.panel_label(ax, 'abcdef'[i * 2 + j])
            if i == 0:
                ax.set_title(DS_TITLE[ds], fontsize=10)
            if j == 0:
                ax.set_ylabel(ylab, fontsize=9)
            if i == 2:
                ax.set_xlabel('n_leak (train points copied into test)')
            if c1 is not None and i == 0 and j == 0:
                ax.legend(fontsize=8, loc='lower right')
            if i == 1 and j == 0:
                ax.legend(fontsize=8, loc='best')
    fig.suptitle('Train–test leakage (mode=leak) — overfit tree + white-box DTA, 30 seeds\n'
                 'H2 CONFIRMED: measured accuracy inflates while honest (clean) accuracy is flat (row a-b).\n'
                 'H1 NULL: leaked points have the same adversarial margin as clean points (row c-d) —\n'
                 'the overfit tree memorizes ALL training data, so leakage leaves no local signature',
                 fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = PLOTS / 'leakage' / 'phase0'; ps.save(fig, p); plt.close(fig)
    print('wrote', p)


if __name__ == '__main__':
    main()
