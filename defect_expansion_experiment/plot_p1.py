"""Phase 1 class-imbalance figures (evidence for FINDINGS_imbalance_p1.md).

Reads results_p1.parquet (svm, tree), results_p1_rf.parquet (rf), results_confound.parquet.
Writes to plots/:
  imbalance_p1_models.png     — normalised spread + minority recall, spatial vs random, per model (tc=2) [H1-survive, H2]
  imbalance_p1_asymmetry.png  — tc=0 (separable) vs tc=2 (contested), svm [H-asym]
  imbalance_p1_confound.png   — before-split vs train-only accuracy/recall, tree+DTA [H-confound]
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

d_ft = pl.read_parquet(HERE / 'results_p1.parquet')                 # svm + tree
d_rf = pl.read_parquet(HERE / 'results_p1_rf.parquet')             # rf
D = pl.concat([d_ft, d_rf])
d_cf = pl.read_parquet(HERE / 'results_confound.parquet')

COL = ps.DEFECT                      # colorblind-safe (Okabe-Ito)
LAB = {'random': 'random deletion (imbalance)', 'spatial': 'spatial deletion (coverage gap)'}
MODEL_TITLE = {'svm': 'rbf-SVM + HSJ (black-box)', 'tree': 'overfit tree + HSJ (black-box)',
               'rf': 'RandomForest + HSJ (black-box)'}


def base_spread(model):
    return D.filter((pl.col('model') == model) & (pl.col('frac') == 0.0))['mean_dist'].drop_nans().mean()


def series(model, tc, structure, col, fracs, norm_base=None):
    xs, ms, cis = [], [], []
    for f in fracs:
        if f == 0.0:
            sub = D.filter((pl.col('model') == model) & (pl.col('frac') == 0.0))
        else:
            sub = D.filter((pl.col('model') == model) & (pl.col('frac') == f) &
                           (pl.col('structure') == structure) & (pl.col('tc') == tc))
        v = sub[col].drop_nans()
        if len(v):
            m = float(v.mean()); s = float(v.std()) if len(v) > 1 else 0.0
            xs.append(f)
            ms.append(m / norm_base if norm_base else m)
            cis.append(1.96 * s / np.sqrt(len(v)) / (norm_base if norm_base else 1))
    return np.array(xs), np.array(ms), np.array(cis)


# ─── Figure 1: models × (spread, recall), tc=2 ────────────────────────────────
def fig_models():
    models = ['svm', 'tree', 'rf']
    fracs = {'svm': [0.0, 0.25, 0.5, 0.7, 0.85, 0.95], 'tree': [0.0, 0.25, 0.5, 0.7, 0.85, 0.95],
             'rf': [0.0, 0.5, 0.85, 0.95]}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2), sharex=True)
    for j, model in enumerate(models):
        nb = base_spread(model)
        # top row: normalised spread
        ax = axes[0][j]
        for s in ['random', 'spatial']:
            x, m, ci = series(model, 2, s, 'mean_dist', fracs[model], norm_base=nb)
            ax.plot(x, m, marker=ps.DEFECT_MK[s], ls=ps.DEFECT_LS[s], color=COL[s], label=LAB[s])
            ax.fill_between(x, m - ci, m + ci, color=COL[s], alpha=0.15)
        ax.axhline(1.0, ls=':', color='0.4', lw=1)
        ax.set_title(MODEL_TITLE[model], fontsize=10)
        if j == 0:
            ax.set_ylabel('normalised adversarial spread\n(× clean baseline)')
            ax.legend(fontsize=7.5, loc='upper left')
        # bottom row: minority recall
        ax = axes[1][j]
        for s in ['random', 'spatial']:
            x, m, ci = series(model, 2, s, 'min_recall', fracs[model])
            ax.plot(x, m, marker=ps.DEFECT_MK[s], ls=ps.DEFECT_LS[s], color=COL[s], label=LAB[s] + ' — recall')
            ax.fill_between(x, m - ci, m + ci, color=COL[s], alpha=0.15)
        ax.set_ylim(0, 1.03); ax.set_xlabel('fraction of class 2 removed')
        if j == 0:
            ax.set_ylabel('minority-class recall\n(virginica, on clean test)')
            ax.legend(fontsize=7.5, loc='lower left')
    fig.suptitle('Phase 1 — class imbalance across models (tc=2 virginica, train-only deletion, matched count)\n'
                 'TOP: spatial (coverage gap) > random (imbalance) spread SURVIVES black-box HSJ + ensembling · '
                 'BOTTOM: minority recall is the clean discriminator', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = PLOTS / 'imbalance' / 'p1_models'; ps.save(fig, p); plt.close(fig)
    print('wrote', p)


# ─── Figure 2: tc asymmetry (svm) ─────────────────────────────────────────────
def fig_asymmetry():
    fracs = [0.0, 0.25, 0.5, 0.7, 0.85, 0.95]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    nb = base_spread('svm')
    tc_style = {0: ('setosa (separable)', '-.', '#2166ac'), 2: ('virginica (contested)', '-', '#b2182b')}
    ax = axes[0]
    for tc, (name, ls, c) in tc_style.items():
        x, m, ci = series('svm', tc, 'spatial', 'mean_dist', fracs, norm_base=nb)
        ax.plot(x, m, marker='o', ls=ls, lw=2.2, color=c, label=f'spatial, tc={tc} {name}')
        ax.fill_between(x, m - ci, m + ci, color=c, alpha=0.13)
    ax.axhline(1.0, ls=':', color='0.4', lw=1)
    ax.set_xlabel('fraction of target class removed'); ax.set_ylabel('normalised adversarial spread')
    ax.set_title('H-asym: deleting from a SEPARABLE class is invisible\n'
                 '(setosa flat ~1.0×; only the contested class moves geometry)', fontsize=9.5)
    ax.legend(fontsize=8, loc='upper left')
    ax = axes[1]
    for tc, (name, ls, c) in tc_style.items():
        x, m, ci = series('svm', tc, 'spatial', 'min_recall', fracs)
        ax.plot(x, m, marker='o', ls=ls, lw=2.2, color=c, label=f'tc={tc} {name} — recall')
        ax.fill_between(x, m - ci, m + ci, color=c, alpha=0.13)
    ax.set_ylim(0, 1.03); ax.set_xlabel('fraction of target class removed'); ax.set_ylabel('minority-class recall')
    ax.set_title('setosa recall stays 1.00 (nothing to misclassify);\nvirginica recall craters',
                 fontsize=9.5)
    ax.legend(fontsize=8, loc='lower left')
    fig.suptitle('Phase 1 — class asymmetry (rbf-SVM + HSJ, spatial deletion, train-only)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p = PLOTS / 'imbalance' / 'p1_asymmetry'; ps.save(fig, p); plt.close(fig)
    print('wrote', p)


# ─── Figure 3: confound (before-split vs train-only) ──────────────────────────
def cf_series(protocol, tc, col):
    fracs = [0.0, 0.25, 0.5, 0.7, 0.85, 0.95]
    xs, ms, cis = [], [], []
    for f in fracs:
        if f == 0.0:
            sub = d_cf.filter((pl.col('frac') == 0.0) & (pl.col('tc') == tc))
        else:
            sub = d_cf.filter((pl.col('frac') == f) & (pl.col('protocol') == protocol) & (pl.col('tc') == tc))
        v = sub[col].drop_nans()
        if len(v):
            xs.append(f); ms.append(float(v.mean()))
            cis.append(1.96 * float(v.std()) / np.sqrt(len(v)))
    return np.array(xs), np.array(ms), np.array(cis)


def fig_confound():
    PCOL = {'before_split': '#8856a7', 'train_only': '#e6550d'}
    PLAB = {'before_split': 'before-split (deletes test band too — flagship protocol)',
            'train_only': 'train-only (clean test — correct protocol)'}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for k, (tc, title) in enumerate([(2, 'tc=2 virginica (contested)'), (0, 'tc=0 setosa (separable)')]):
        ax = axes[k]
        for p in ['before_split', 'train_only']:
            x, m, ci = cf_series(p, tc, 'vacc')
            ax.plot(x, m, marker='o', lw=2.2, color=PCOL[p], label=PLAB[p] + ' — test acc')
            ax.fill_between(x, m - ci, m + ci, color=PCOL[p], alpha=0.15)
            x, m, ci = cf_series(p, tc, 'min_recall')
            ax.plot(x, m, marker='.', ls='--', lw=1.5, color=PCOL[p], alpha=0.7,
                    label=PLAB[p] + ' — minority recall')
        ax.set_ylim(0, 1.04); ax.set_xlabel('fraction of class removed (spatial)')
        if k == 0:
            ax.set_ylabel('test accuracy / minority recall')
        ax.set_title(title, fontsize=10); ax.legend(fontsize=6.8, loc='lower left')
    fig.suptitle('Phase 1 — coverage-gap ACCURACY CONFOUND (overfit tree + DTA, 30 seeds)\n'
                 'before-split injection makes accuracy RISE (deletes hard test cases); '
                 'train-only makes it DROP — a pure protocol artifact', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p = PLOTS / 'imbalance' / 'p1_confound'; ps.save(fig, p); plt.close(fig)
    print('wrote', p)


if __name__ == '__main__':
    fig_models(); fig_asymmetry(); fig_confound()
