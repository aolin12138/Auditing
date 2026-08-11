"""Model PERFORMANCE (train + test accuracy) vs outlier count, toward direction.
All models from both parquets. Dual-y: spread (left) + accuracy (right).
Panel A: tree_d10 (white-box DTA) — full ceiling range.
Panel B: SVM (HSJ) — full ceiling range.
Panel C: all-models accuracy overview at (k=8, n_out=10).
-> plots/model_performance.png
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
ext = pl.read_parquet(HERE / 'results_extended.parquet')
mf = pl.read_parquet(HERE.parent / 'model_family_experiment' / 'results.parquet').filter(pl.col('defect') == 'outlier')
CEIL = 40


def curve(frame, model, direction, k, ns, col):
    """return x, mean, ci for a metric over n_out levels."""
    xs, ms, cis = [], [], []
    for n in ns:
        v = frame.filter((pl.col('model') == model) & (pl.col('direction') == direction)
                         & (pl.col('k') == k) & (pl.col('n_out') == n))[col].drop_nans().drop_nulls()
        if len(v):
            xs.append(n); ms.append(float(v.mean())); cis.append(1.96 * float(v.std()) / np.sqrt(len(v)))
    return np.array(xs), np.array(ms), np.array(cis)


def spread_curve(frame, model, direction, k, ns):
    b = frame.filter((pl.col('model') == model) & (pl.col('n_out') == 0))['mean_dist'].drop_nans().mean()
    xs, ms, cis = curve(frame, model, direction, k, ns, 'mean_dist')
    return xs, ms / b, cis / b


fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# --- Panel A: tree_d10 — toward spread + train/test accuracy ---
ax = axes[0][0]
NS = [0, 1, 3, 5, 10, 15, 20, 30, 40]
xs, m_sp, ci_sp = spread_curve(ext, 'tree_d10', 'toward', 8, NS)
ax.plot(xs, m_sp, marker='o', lw=2.2, color='#1e5eff', label='spread (x baseline)')
ax.fill_between(xs, m_sp - ci_sp, m_sp + ci_sp, color='#1e5eff', alpha=0.12)
ax.axhline(1.0, ls=':', color='0.4', lw=1)
ax.set_ylabel('normalised spread', color='#1e5eff')
ax.tick_params(axis='y', labelcolor='#1e5eff')
ax2 = ax.twinx()
xs_tacc, m_tacc, ci_tacc = curve(ext, 'tree_d10', 'toward', 8, NS, 'tacc')
xs_vacc, m_vacc, ci_vacc = curve(ext, 'tree_d10', 'toward', 8, NS, 'vacc')
ax2.plot(xs_tacc, m_tacc, marker='s', lw=1.5, ls='--', color='#e08214', label='train acc')
ax2.plot(xs_vacc, m_vacc, marker='^', lw=1.5, ls='-.', color='#d73027', label='test acc')
ax2.set_ylabel('accuracy'); ax2.set_ylim(0.7, 1.02)
ax.set_title('tree d10 + DTA (toward): accuracy-BLIND —\nspread collapses at ceiling, acc flat', fontsize=9.5)
lines1, labels1 = ax.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7.5, loc='lower left')

# --- Panel B: SVM — toward spread + train/test accuracy ---
ax = axes[0][1]
NS_SVM = [0, 5, 10, 20, 30, 40]
xs, m_sp, ci_sp = spread_curve(ext, 'svm', 'toward', 8, NS_SVM)
ax.plot(xs, m_sp, marker='o', lw=2.2, color='#2ca25f', label='spread (x baseline)')
ax.fill_between(xs, m_sp - ci_sp, m_sp + ci_sp, color='#2ca25f', alpha=0.12)
ax.axhline(1.0, ls=':', color='0.4', lw=1)
ax.set_ylabel('normalised spread', color='#2ca25f')
ax.tick_params(axis='y', labelcolor='#2ca25f')
ax2 = ax.twinx()
xs_tacc, m_tacc, ci_tacc = curve(ext, 'svm', 'toward', 8, NS_SVM, 'tacc')
xs_vacc, m_vacc, ci_vacc = curve(ext, 'svm', 'toward', 8, NS_SVM, 'vacc')
ax2.plot(xs_tacc, m_tacc, marker='s', lw=1.5, ls='--', color='#e08214', label='train acc')
ax2.plot(xs_vacc, m_vacc, marker='^', lw=1.5, ls='-.', color='#d73027', label='test acc')
ax2.set_ylabel('accuracy'); ax2.set_ylim(0.7, 1.02)
ax.set_title('SVM + HSJ (toward): accuracy DEGRADES —\nas spread slowly rises to ceiling', fontsize=9.5)
lines1, labels1 = ax.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7.5, loc='best')

# --- Panel C: all-models accuracy (train + test) at k=8, n_out=10 ---
ax = axes[1][0]
MODELS = [('tree_d10 + DTA', ext, 'tree_d10', '#1e5eff'),
          ('SVM + HSJ', ext, 'svm', '#2ca25f'),
          ('RF + HSJ', mf, 'rf', '#8e44ad'),
          ('single tree\n+ HSJ', mf, 'tree', '#c0392b')]
mid = len(MODELS) + 1
xticks, xticklabels = [], []
for i, (lab, frame, model, col) in enumerate(MODELS):
    j = i % 2; offset = (j - 0.5) * 0.2
    for metric, marker, mcol in [('tacc', 's', '#e08214'), ('vacc', '^', '#d73027')]:
        v = frame.filter((pl.col('model') == model) & (pl.col('direction') == 'toward')
                         & (pl.col('k') == 8) & (pl.col('n_out') >= 0))[metric].drop_nans().drop_nulls()
        m = float(v.mean()) if len(v) else np.nan
        ax.scatter([i + 1 + offset], [m], marker=marker, s=60, color=mcol, alpha=0.8,
                   zorder=5, label=('train acc' if metric == 'tacc' and i == 0 else
                                    'test acc' if metric == 'vacc' and i == 0 else ''))
    xticks.append(i + 1); xticklabels.append(lab)
ax.set_xticks(xticks); ax.set_xticklabels(xticklabels, fontsize=8)
ax.set_ylabel('accuracy'); ax.set_ylim(0.75, 1.02)
ax.set_title('Train & test accuracy @ toward, k=8\n(tree&RF train=1.0 perfect; SVM degrades)', fontsize=9.5)
ax.legend(fontsize=7.5, loc='lower right')
ax.axhline(1.0, ls=':', color='0.4', lw=0.8, alpha=0.5)
ax.grid(axis='y', alpha=0.3)

# --- Panel D: tree_d3 — toward spread + train/test accuracy (same pattern, title note) ---
ax = axes[1][1]
xs, m_sp, ci_sp = spread_curve(ext, 'tree_d3', 'toward', 8, NS)
ax.plot(xs, m_sp, marker='o', lw=2.2, color='#e08214', label='spread (x baseline)')
ax.fill_between(xs, m_sp - ci_sp, m_sp + ci_sp, color='#e08214', alpha=0.12)
ax.axhline(1.0, ls=':', color='0.4', lw=1)
ax.set_ylabel('normalised spread', color='#e08214')
ax.tick_params(axis='y', labelcolor='#e08214')
ax2 = ax.twinx()
xs_tacc, m_tacc, ci_tacc = curve(ext, 'tree_d3', 'toward', 8, NS, 'tacc')
xs_vacc, m_vacc, ci_vacc = curve(ext, 'tree_d3', 'toward', 8, NS, 'vacc')
ax2.plot(xs_tacc, m_tacc, marker='s', lw=1.5, ls='--', color='#999', label='train acc')
ax2.plot(xs_vacc, m_vacc, marker='^', lw=1.5, ls='-.', color='#d73027', label='test acc')
ax2.set_ylabel('accuracy'); ax2.set_ylim(0.7, 1.02)
ax.set_title('tree d3 + DTA (toward): partial collapse —\npruned tree keeps stratifying outliers', fontsize=9.5)
lines1, labels1 = ax.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7.5, loc='lower left')

fig.suptitle('MODEL PERFORMANCE — spread, train & test accuracy vs outlier count (toward, k=8)', fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(PLOTS / 'model_performance.png', dpi=140)
print('wrote', PLOTS / 'model_performance.png')
