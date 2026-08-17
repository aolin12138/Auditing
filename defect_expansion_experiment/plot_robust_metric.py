"""§8 — robust-metric figures.

Per (model, attack): a 5x2 grid — rows = candidate metrics (m0 raw reference, m1 kNN-ratio,
m2b LOF-robust, m3 PCA-spread, m4 kNN-local), cols = iris / wine. Each panel: normalised
metric vs severity, spatial (coverage gap) vs random (imbalance), 30/15-seed 95% CI bands,
train-only protocol, dotted line at 1.0 (= clean baseline). Panel labels a-j.

Plus a concentration figure: std/mean of the per-point dispersion quantity on clean
baseline clouds, per dataset, per metric — with the raw-spread 0.54 (iris) / 0.30 (wine)
reference. A robust metric should equalise these (ratio ~1), like m3.

-> plots/robust_{tree+dta,svm+hsj}.png + plots/robust_concentration.png (PNG + vector PDF)
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
PLOTS = HERE / 'plots'; PLOTS.mkdir(exist_ok=True)
FRAC = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
METRIC_LAB = {
    'm0_raw_spread':  'm0 raw OPTICS spread (reference — fragile on wine)',
    'm1_knn_ratio':   'm1 kNN-distance ratio (relative isolation — null)',
    'm2b_lof_robust': 'm2b LOF local-density ratio, jittered+median (null after fix)',
    'm3_pca_spread':  'm3 PCA-then-spread (OPTICS spread in top-3 PC subspace)',
    'm4_knn_local':   'm4 kNN-graph local spread (mean dist to k nearest neighbours)',
}
ORDER = ['m0_raw_spread', 'm1_knn_ratio', 'm2b_lof_robust', 'm3_pca_spread', 'm4_knn_local']
COL = ps.DEFECT; LAB = ps.DEFECT_LAB; MK = ps.DEFECT_MK; LS = ps.DEFECT_LS


def series(df, ds, metric, structure):
    xs, ms, cis = [], [], []
    for f in FRAC:
        if f == 0.0:
            sub = df.filter((pl.col('dataset') == ds) & (pl.col('metric') == metric) & (pl.col('frac') == 0.0))
        else:
            sub = df.filter((pl.col('dataset') == ds) & (pl.col('metric') == metric) &
                            (pl.col('frac') == f) & (pl.col('structure') == structure))
        v = sub['value_norm'].drop_nans()
        if len(v):
            m = float(v.mean()); se = 1.96 * float(v.std()) / np.sqrt(len(v)) if len(v) > 1 else 0.0
            xs.append(f); ms.append(m); cis.append(se)
    return np.array(xs), np.array(ms), np.array(cis)


def fig_metrics(ma):
    df = pl.read_parquet(HERE / 'results_robust_metric.parquet').filter(pl.col('model_attack') == ma)
    fig, axes = plt.subplots(5, 2, figsize=(12.5, 16), sharex=True)
    for i, metric in enumerate(ORDER):
        for j, ds in enumerate(['iris', 'wine']):
            ax = axes[i][j]
            for st in ['spatial', 'random']:
                x, m, ci = series(df, ds, metric, st)
                ax.plot(x, m, marker=MK[st], ls=LS[st], color=COL[st], lw=2, label=LAB[st])
                ax.fill_between(x, m - ci, m + ci, color=COL[st], alpha=0.15)
            ax.axhline(1.0, ls=':', color='0.4', lw=1)
            ps.panel_label(ax, 'abcdefghij'[i * 2 + j])
            if j == 0:
                ax.set_ylabel('normalised value\n(× clean baseline)', fontsize=9)
            if i == 0:
                ax.set_title(f'{ds} ({4 if ds == "iris" else 13}-D)', fontsize=10)
            if i == 4:
                ax.set_xlabel('fraction of target class removed (severity)')
            if i == 0 and j == 1:
                ax.legend(fontsize=8, loc='upper left')
    fig.suptitle(f'{ma} — candidate spread metrics on the SAME adversarial clouds (train-only)\n'
                 'coverage gap (spatial) vs imbalance (random), 95% CIs · raw OPTICS spread = m0 reference',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    # metric labels down the left edge
    for i, metric in enumerate(ORDER):
        axes[i][0].text(-0.42, 0.5, METRIC_LAB[metric], transform=axes[i][0].transAxes,
                        rotation=90, va='center', ha='center', fontsize=8.5)
    p = PLOTS / f'robust_{ma}'; ps.save(fig, p); plt.close(fig)
    print('wrote', p)


def fig_concentration():
    """std/mean of the per-point dispersion quantity on clean baseline clouds (check c)."""
    s = pl.read_csv(HERE / 'results_robust_summary.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for j, ma in enumerate(['tree+dta', 'svm+hsj']):
        ax = axes[j]
        sub = s.filter((pl.col('model_attack') == ma) & (pl.col('frac') == 0.8))
        mets, iris, wine = [], [], []
        for r in sub.filter(pl.col('dataset') == 'iris').iter_rows(named=True):
            if r['metric'] == 'm2_lof':
                continue                                   # numerically unstable, not meaningful
            mets.append(r['metric']); iris.append(r['conc_iris'])
        for m in mets:
            wine.append(sub.filter((pl.col('metric') == m) & (pl.col('dataset') == 'wine'))['conc_wine'][0])
        x = np.arange(len(mets)); w = 0.38
        ax.bar(x - w / 2, iris, w, label='iris (4-D)', color=ps.OKABE_ITO['blue'])
        ax.bar(x + w / 2, wine, w, label='wine (13-D)', color=ps.OKABE_ITO['vermillion'])
        ax.set_xticks(x); ax.set_xticklabels([m.replace('_', '\n', 1) for m in mets], fontsize=7.5)
        ax.axhline(0.54, ls='--', color=ps.OKABE_ITO['blue'], lw=1, alpha=0.7)
        ax.axhline(0.30, ls='--', color=ps.OKABE_ITO['vermillion'], lw=1, alpha=0.7)
        ax.set_title(f'{ma}\nraw-spread reference: iris 0.54 / wine 0.30 (dashed)', fontsize=9)
        if j == 0:
            ax.set_ylabel('distance-concentration std/mean\n(on clean baseline clouds)')
            ax.legend(fontsize=8)
        ps.panel_label(ax, 'ab'[j])
    fig.suptitle('check (c): does the metric resist distance concentration? (std/mean on baseline clouds)\n'
                 'm3 (PCA) equalises iris≈wine (ratio ~1); m1/m2b/m4 stay imbalanced yet m4 still separates —\n'
                 'its recovery is via locality (bulk pairs excluded), not via equalising concentration',
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    p = PLOTS / 'robust_concentration'; ps.save(fig, p); plt.close(fig)
    print('wrote', p)


if __name__ == '__main__':
    for ma in ['tree+dta', 'svm+hsj']:
        fig_metrics(ma)
    fig_concentration()
