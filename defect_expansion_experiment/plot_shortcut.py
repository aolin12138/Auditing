"""Phase 0 + Phase 1 shortcut (Clever Hans) figures.

tree + white-box DTA (30 seeds) -> plots/shortcut/phase0.png — 5x2 grid:
  rows = [test accuracy, first-spurious-split depth, spurious-axis fraction of adv L2
          displacement, mean adv L2 displacement, normalised scalar spread (m0+m4)]
svm + black-box HSJ (15 seeds) -> plots/shortcut/svm_hsj.png — 4x2 grid (no tree depth row):
  rows = [test accuracy, spurious-axis fraction of adv L2 displacement, mean adv L2
          displacement, normalised scalar spread (m0+m4)]
x = corr (shortcut strength), 95% CI bands.
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
DS_TITLE = {'iris': 'iris (4-D)', 'wine': 'wine (13-D)'}
COL = ps.OKABE_ITO
SPREAD = [('spread_m0', 'raw OPTICS spread (m0)', '-'), ('spread_m4', 'kNN-local spread (m4)', '--')]
TREE_ROWS = [('vacc', 'test accuracy\n(Clever Hans cost)', (0.3, 1.0)),
             ('spur_depth', 'depth of first spurious split\n(0 = root; low = reliance)', (0, 3.5)),
             ('spur_frac', 'spurious-axis fraction of adv L2\ndisplacement (H1)', (0, 1.05)),
             ('adv_l2', 'mean adversarial L2 displacement', (0.5, 4.5))]
SVM_ROWS = [('vacc', 'test accuracy\n(Clever Hans cost)', (0.3, 1.0)),
            ('spur_frac', 'spurious-axis fraction of adv L2\ndisplacement (H1)', (0, 1.05)),
            ('adv_l2', 'mean adversarial L2 displacement', (0.5, 6.5))]


def series(df, ds, col):
    xs, ms, cis = [], [], []
    for c in CORR:
        v = df.filter((pl.col('dataset') == ds) & (pl.col('corr') == c))[col].drop_nans()
        m, n = float(v.mean()), len(v)
        se = 1.96 * float(v.std()) / np.sqrt(n) if n > 1 else 0.0
        xs.append(c); ms.append(m); cis.append(se)
    return xs, np.array(ms), np.array(cis)


def spread_panel(ax, df, ds):
    for col, lab, ls in SPREAD:
        xs, m, ci = series(df, ds, col)
        base = float(df.filter((pl.col('dataset') == ds) & (pl.col('corr') == 0.0))[col].drop_nans().mean())
        color = COL['vermillion'] if ls == '-' else COL['blue']
        ax.plot(X, m / base, marker='o', ls=ls, color=color, lw=2, label=lab)
        ax.fill_between(X, (m - ci) / base, (m + ci) / base, color=color, alpha=0.12)
    ax.axhline(1.0, ls=':', color='0.4', lw=1)
    ax.set_xticks(X); ax.set_xticklabels([str(c) for c in CORR])
    ax.set_ylim(0.8, 1.7)
    ax.set_xlabel('shortcut strength corr (train signal / noise sigma)')


def main():
    tree = pl.read_parquet(HERE / 'results_shortcut.parquet')
    svm = pl.read_parquet(HERE / 'results_shortcut_svm.parquet')
    # ---- tree + DTA
    rows = TREE_ROWS + [('spread', None, None)]
    fig, axes = plt.subplots(5, 2, figsize=(12.5, 15))
    for i, (col, ylab, ylim) in enumerate(TREE_ROWS):
        for j, ds in enumerate(['iris', 'wine']):
            ax = axes[i][j]
            xs, m, ci = series(tree, ds, col)
            ax.plot(X, m, marker='o', color=COL['vermillion'], lw=2)
            ax.fill_between(X, m - ci, m + ci, color=COL['vermillion'], alpha=0.15)
            ax.set_xticks(X); ax.set_xticklabels([str(c) for c in CORR])
            if ylim:
                ax.set_ylim(*ylim)
            ps.panel_label(ax, 'abcdefghij'[i * 2 + j])
            if i == 0:
                ax.set_title(DS_TITLE[ds], fontsize=10)
            if j == 0:
                ax.set_ylabel(ylab, fontsize=9)
    for j, ds in enumerate(['iris', 'wine']):
        ax = axes[4][j]
        spread_panel(ax, tree, ds)
        ps.panel_label(ax, 'ij'[j])
        if j == 0:
            ax.set_ylabel('normalised adversarial spread\n(× clean baseline)', fontsize=9)
            ax.legend(fontsize=8, loc='lower left')
    fig.suptitle('Clever Hans / spurious feature — tree + white-box DTA, 30 seeds, train-only injection\n'
                 'test accuracy collapses, the first spurious split rises to the root, adversarial displacement\n'
                 'concentrates on the spurious axis (rows a-d); scalar spread (i-j) stays flat on iris and rises\n'
                 'on wine only at extreme doses', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = PLOTS / 'shortcut' / 'phase0'; ps.save(fig, p); plt.close(fig)
    print('wrote', p)
    # ---- svm + HSJ
    fig, axes = plt.subplots(4, 2, figsize=(12.5, 12.5))
    for i, (col, ylab, ylim) in enumerate(SVM_ROWS):
        for j, ds in enumerate(['iris', 'wine']):
            ax = axes[i][j]
            xs, m, ci = series(svm, ds, col)
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
    for j, ds in enumerate(['iris', 'wine']):
        ax = axes[3][j]
        spread_panel(ax, svm, ds)
        ps.panel_label(ax, 'gh'[j])
        if j == 0:
            ax.set_ylabel('normalised adversarial spread\n(× clean baseline)', fontsize=9)
            ax.legend(fontsize=8, loc='lower left')
    fig.suptitle('Clever Hans / spurious feature — svm + black-box HSJ, 15 seeds, train-only injection\n'
                 'the per-axis signal SURVIVES the black-box attack (peaks at corr=4, wine now leads),\n'
                 'dips at corr=8 (survivorship); scalar spread fires only after accuracy collapses',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = PLOTS / 'shortcut' / 'svm_hsj'; ps.save(fig, p); plt.close(fig)
    print('wrote', p)


if __name__ == '__main__':
    main()
