"""Phase 0.5 figures (reads results_extended.parquet). Spread is NORMALISED to each
model's own clean (n_out=0) baseline: 1.0 = no effect, 1.3 = 30% more scattered.
CIs are 95% (1.96 * sd / sqrt(n)). OFAT slices hold the OTHER factor fixed and say so.
  1. phase05_dose_response.png  — normalised spread vs k (n_out FIXED=10) and vs n_out
     (k FIXED=6), TOWARD, one line per model, 95% CI.
  2. phase05_direction_by_model.png — toward vs outward at k=8,n_out=10 per model.
  3. phase05_tree_heatmaps.png — normalised k x n_out (toward) for tree_d3 and tree_d10.
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
df = pl.read_parquet(HERE / 'results_extended.parquet')

MODELS = ['tree_d3', 'tree_d10', 'svm']
MCOL = {'tree_d3': '#e08214', 'tree_d10': '#1e5eff', 'svm': '#2ca25f'}
MLAB = {'tree_d3': 'tree depth 3 (pruned)', 'tree_d10': 'tree depth 10 (deep)', 'svm': 'SVM (rbf)'}

# normalise mean_dist by each model's clean baseline
base = {m: float(df.filter((pl.col('model') == m) & (pl.col('n_out') == 0))['mean_dist'].mean())
        for m in MODELS}
df = df.with_columns(pl.struct(['model', 'mean_dist'])
                     .map_elements(lambda s: s['mean_dist'] / base[s['model']], return_dtype=pl.Float64)
                     .alias('spread_norm'))


def slice_ci(model, direction, by, fix_col, fix_val):
    d = df.filter((pl.col('model') == model) & (pl.col('direction') == direction)
                  & (pl.col(fix_col) == fix_val))
    g = d.group_by(by).agg(pl.col('spread_norm').mean().alias('m'),
                           pl.col('spread_norm').std().alias('sd'),
                           pl.len().alias('n')).sort(by)
    x = g[by].to_numpy(); m = g['m'].to_numpy()
    ci = 1.96 * g['sd'].to_numpy() / np.sqrt(np.maximum(g['n'].to_numpy(), 1))
    return x, m, ci


# ---- Figure 1: dose-response (toward), normalised, fixed other factor ----
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
for ax, by, fixc, fixv, xlabel in [
        (axes[0], 'k', 'n_out', 10, 'outlier distance  k  (x class std)   [n_out fixed = 10]'),
        (axes[1], 'n_out', 'k', 6, 'number of outliers  n_out   [k fixed = 6]')]:
    for m in MODELS:
        x, mean, ci = slice_ci(m, 'toward', by, fixc, fixv)
        if len(x) == 0:
            continue
        ax.plot(x, mean, marker='o', color=MCOL[m], lw=2, label=MLAB[m])
        ax.fill_between(x, mean - ci, mean + ci, color=MCOL[m], alpha=0.15)
    ax.axhline(1.0, ls='--', color='0.4', lw=1.2, label='clean baseline (=1.0)')
    ax.set_xlabel(xlabel); ax.set_ylabel('normalised spread  (x baseline)')
    ax.set_title(f'TOWARD outliers: spread vs {by}')
axes[0].legend(fontsize=8, loc='upper left')
fig.suptitle('Outlier Phase 0.5 (normalised, 95% CI): trees show a clear TOWARD dose-response; '
             'SVM stays ~1.0 (robust)', fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(PLOTS / 'phase05_dose_response.png', dpi=140); plt.close(fig)

# ---- Figure 2: direction x model at the strong cell (k=8, n_out=10) ----
fig, ax = plt.subplots(figsize=(7, 4.3))
x = np.arange(len(MODELS)); w = 0.36
for i, direction in enumerate(['toward', 'outward']):
    means, cis = [], []
    for m in MODELS:
        d = df.filter((pl.col('model') == m) & (pl.col('direction') == direction)
                      & (pl.col('k') == 8) & (pl.col('n_out') == 10))
        means.append(float(d['spread_norm'].mean()))
        n = d.height
        cis.append(1.96 * float(d['spread_norm'].std() or 0) / np.sqrt(max(n, 1)))
    ax.bar(x + (i - 0.5) * w, means, w, yerr=cis, capsize=4,
           color=['#c0392b', '#8e44ad'][i], alpha=0.85, label=direction)
ax.axhline(1.0, ls='--', color='0.4', lw=1.2, label='clean baseline')
ax.set_xticks(x); ax.set_xticklabels([MLAB[m] for m in MODELS], fontsize=8)
ax.set_ylabel('normalised spread  (x baseline)')
ax.set_title('Outlier signal by model & direction  (k=8, n_out=10, 95% CI)\n'
             'trees: toward >> outward(null);  SVM: robust, faint opposite lean', fontsize=9.5)
ax.legend()
fig.tight_layout(); fig.savefig(PLOTS / 'phase05_direction_by_model.png', dpi=140); plt.close(fig)

# ---- Figure 3: normalised k x n_out heatmaps (toward) for the two trees ----
ks, ns = [2, 3, 4, 5, 6, 8], [1, 3, 5, 10, 15, 20]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, m in zip(axes, ['tree_d3', 'tree_d10']):
    M = np.full((len(ks), len(ns)), np.nan)
    d = df.filter((pl.col('model') == m) & (pl.col('direction') == 'toward'))
    piv = d.group_by(['k', 'n_out']).agg(pl.col('spread_norm').mean().alias('s'))
    for r in piv.iter_rows(named=True):
        M[ks.index(int(r['k'])), ns.index(int(r['n_out']))] = r['s']
    im = ax.imshow(M, cmap='viridis', aspect='auto', origin='lower', vmin=0.9, vmax=1.5)
    ax.set_xticks(range(len(ns))); ax.set_xticklabels(ns)
    ax.set_yticks(range(len(ks))); ax.set_yticklabels(ks)
    ax.set_xlabel('n_out'); ax.set_ylabel('k (x class std)')
    for i in range(len(ks)):
        for j in range(len(ns)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center',
                        color='white' if M[i, j] < 1.2 else 'black', fontsize=8)
    ax.set_title(f'{MLAB[m]}  (toward, normalised)')
fig.colorbar(im, ax=axes.ravel().tolist(), label='normalised spread', shrink=0.85)
fig.suptitle('Outlier Phase 0.5: k x n_out interaction — signal switches on at k>=6 (both depths)', fontsize=10)
fig.savefig(PLOTS / 'phase05_tree_heatmaps.png', dpi=140); plt.close(fig)

# ---- Figure 4: SVM only - toward vs outward vs k (n_out=10), 10 seeds ----
fig, ax = plt.subplots(figsize=(7, 4.4))
for direction, col in [('toward', '#c0392b'), ('outward', '#8e44ad')]:
    x, mean, ci = slice_ci('svm', direction, 'k', 'n_out', 10)
    ax.plot(x, mean, marker='o', lw=2, color=col, label=direction)
    ax.fill_between(x, mean - ci, mean + ci, color=col, alpha=0.15)
ax.axhline(1.0, ls='--', color='0.4', lw=1.2, label='clean baseline (=1.0)')
ax.set_xlabel('outlier distance  k  (x class std)   [n_out fixed = 10]')
ax.set_ylabel('normalised spread  (x baseline)')
ax.set_title('SVM (rbf), 10 seeds, 95% CI: robust to outliers in BOTH directions.\n'
             'outward flat even at k=12 (rbf kernel localises far outliers away)', fontsize=9.5)
ax.legend()
fig.tight_layout(); fig.savefig(PLOTS / 'phase05_svm_directions.png', dpi=140); plt.close(fig)

print('wrote:')
for p in sorted(PLOTS.glob('phase05_*.png')):
    print(' ', p.name, f'({p.stat().st_size} bytes)')
print('\nper-model clean baseline spread:', {k: round(v, 3) for k, v in base.items()})
