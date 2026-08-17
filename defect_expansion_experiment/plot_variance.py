"""Phase 1 unified + variance figures (tree + white-box DTA).

Per dataset, one 3x2 figure — as the defect grows:
  rows = [test accuracy, minority-class recall, normalised adversarial spread]
  cols = [train-only (clean test), before-split (deletes the band from test too)]
  lines = spatial (coverage gap) vs random (imbalance), 30 seeds, 95% CI.
Answers, for this (model, attack): how acc changes, how spread changes, coverage-gap vs
random, and how train-only vs before-split differs — replicated on iris (4-D) and wine (13-D).
-> plots/variance_<dataset>.png  +  plots/variance_spread_summary.png
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
PLOTS = HERE / 'plots'; PLOTS.mkdir(exist_ok=True)
DFS = {'tree+DTA': pl.read_parquet(HERE / 'results_variance.parquet')}
_svm = HERE / 'results_variance_svm.parquet'
if _svm.exists():
    DFS['svm+HSJ'] = pl.read_parquet(_svm)
D = DFS['tree+DTA']                                    # default for the per-dataset 3x2 figures
FRAC = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
COL = {'random': '#2ca25f', 'spatial': '#c0392b'}
LAB = {'random': 'random (imbalance)', 'spatial': 'spatial (coverage gap)'}
DS_TITLE = {'iris': 'iris (4-D · tc=2 virginica · feat=petal width)',
            'wine': 'wine (13-D · tc=0 · feat=proline)'}


def base_spread(ds, df=None):
    df = D if df is None else df
    return df.filter((pl.col('dataset') == ds) & (pl.col('frac') == 0.0))['mean_dist'].drop_nans().mean()


def series(ds, structure, protocol, col, norm=None, df=None):
    df = D if df is None else df
    xs, ms, cis = [], [], []
    for f in FRAC:
        if f == 0.0:
            sub = df.filter((pl.col('dataset') == ds) & (pl.col('frac') == 0.0))
        else:
            sub = df.filter((pl.col('dataset') == ds) & (pl.col('frac') == f) &
                            (pl.col('structure') == structure) & (pl.col('protocol') == protocol))
        v = sub[col].drop_nans()
        if len(v):
            m = float(v.mean()); s = float(v.std()) if len(v) > 1 else 0.0
            xs.append(f); ms.append(m / norm if norm else m)
            cis.append(1.96 * s / np.sqrt(len(v)) / (norm if norm else 1))
    return np.array(xs), np.array(ms), np.array(cis)


def fig_dataset(ds, df=None, ma='tree+DTA', suffix=''):
    global D
    D = df if df is not None else DFS[ma]
    nb = base_spread(ds, D)
    rows = [('vacc', 'test accuracy', None, (0, 1.03)),
            ('min_recall', 'minority-class recall', None, (0, 1.03)),
            ('mean_dist', 'normalised adversarial spread\n(× clean baseline)', nb, None)]
    cols = ['train_only', 'before_split']
    ctitle = {'train_only': 'train-only injection (clean test)',
              'before_split': 'before-split injection (deletes test band too)'}
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 11), sharex=True)
    for i, (col, ylab, nrm, ylim) in enumerate(rows):
        for j, proto in enumerate(cols):
            ax = axes[i][j]
            for s in ['spatial', 'random']:
                x, m, ci = series(ds, s, proto, col, norm=nrm)
                ax.plot(x, m, marker='o', lw=2.2, color=COL[s], label=LAB[s])
                ax.fill_between(x, m - ci, m + ci, color=COL[s], alpha=0.15)
            if col == 'mean_dist':
                ax.axhline(1.0, ls=':', color='0.4', lw=1)
            if ylim:
                ax.set_ylim(*ylim)
            if i == 0:
                ax.set_title(ctitle[proto], fontsize=10)
            if j == 0:
                ax.set_ylabel(ylab, fontsize=9.5)
            if i == 2:
                ax.set_xlabel('fraction of target class removed (severity)')
            if i == 0 and j == 0:
                ax.legend(fontsize=8, loc='lower left')
    fig.suptitle(f'{DS_TITLE[ds]} — {ma}\n'
                 'coverage gap (spatial) vs imbalance (random) · train-only vs before-split',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = PLOTS / f'variance{suffix}_{ds}.png'; fig.savefig(p, dpi=135); plt.close(fig)
    print('wrote', p)


def fig_summary():
    """Spatial normalised spread, train-only, across dataset x (model,attack) — the fragility map."""
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    col = {'iris': '#b2182b', 'wine': '#2166ac'}
    ls = {'tree+DTA': '-', 'svm+HSJ': '--'}
    for ma, df in DFS.items():
        for ds in ['iris', 'wine']:
            nb = base_spread(ds, df)
            x, m, c = series(ds, 'spatial', 'train_only', 'mean_dist', norm=nb, df=df)
            ax.plot(x, m, marker='o', ls=ls[ma], lw=2.3, color=col[ds], label=f'{ds} · {ma} (spatial)')
            ax.fill_between(x, m - c, m + c, color=col[ds], alpha=0.10)
    ax.axhline(1.0, ls=':', color='0.4', lw=1)
    ax.set_xlabel('fraction of target class removed'); ax.set_ylabel('normalised adversarial spread (spatial)')
    ax.set_title('SPREAD-signal fragility map (train-only, coverage gap)\n'
                 'strong only on iris+tree/DTA (~1.25×); weak under SVM/HSJ and on 13-D wine (~1.0×)\n'
                 '→ the geometry signal is fragile to BOTH dimensionality and attack', fontsize=9.5)
    ax.legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    p = PLOTS / 'variance_spread_summary.png'; fig.savefig(p, dpi=140); plt.close(fig)
    print('wrote', p)


if __name__ == '__main__':
    for ds in ['iris', 'wine']:
        fig_dataset(ds, DFS['tree+DTA'], 'tree+DTA', '')
        if 'svm+HSJ' in DFS:
            fig_dataset(ds, DFS['svm+HSJ'], 'svm+HSJ', '_svm')
    fig_summary()
