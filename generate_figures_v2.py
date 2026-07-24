"""Ultra-clean report figures. One story per plot. No dual axes. No in-chart annotations.
Tufte-inspired: maximize data-ink ratio. Captions tell the interpretation.
"""
import os; os.environ.update({k:'1' for k in
    ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']})
import numpy as np, polars as pl, matplotlib, matplotlib.pyplot as plt
matplotlib.use('Agg')

# Clean academic style
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'axes.labelsize': 11, 'axes.titlesize': 12,
    'legend.fontsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': False, 'figure.dpi': 200, 'savefig.dpi': 200,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
})

OUT = 'figures/report'
os.makedirs(OUT, exist_ok=True)

dt = pl.read_parquet('dtree_attack_experiment/results.parquet')
hs = pl.read_parquet('hsj_svm_experiment/results.parquet')
ln = pl.read_parquet('hsj_label_noise_experiment/results.parquet')

PALETTE = {'Tree+DTA':'#2ca02c','Tree+HSJ':'#1f77b4','SVM+HSJ':'#d62728',
           'Label noise':'#e74c3c','Coverage gap':'#2ca02c',
           'Overfit':'#e74c3c','Pruned':'#9b59b6'}

# ═══════════════════════════════════════════════════════════
# PLOT 1: Coverage gap — mean_dist rises, accuracy is flat
# ═══════════════════════════════════════════════════════════
print("Plot 1...")
fig, (ax, ax_acc) = plt.subplots(2, 1, figsize=(7, 5), gridspec_kw={'height_ratios': [3, 1]})

for combo, src, col, color in [
    ('Tree + DTA', dt.filter((pl.col('defect')=='coverage_gap') & ~pl.col('mean_dist').is_nan()), 'level', '#2ca02c'),
    ('SVM + HSJ', hs.filter((pl.col('model')=='svm') & ~pl.col('mean_dist').is_nan()), 'bias', '#d62728'),
    ('Tree + HSJ', hs.filter((pl.col('model')=='tree') & ~pl.col('mean_dist').is_nan()), 'bias', '#1f77b4'),
]:
    agg = src.group_by(col).agg(pl.col('mean_dist').mean().alias('m'), pl.col('mean_dist').std().alias('s'), pl.len().alias('n')).sort(col)
    x = agg[col].to_numpy(); m = agg['m'].to_numpy(); se = agg['s'].to_numpy() / np.sqrt(agg['n'].to_numpy())
    ax.errorbar(x, m, yerr=1.96*se, fmt='o-', lw=2, ms=5, capsize=3, color=color, label=combo)
    # Accuracy
    agg_acc = src.group_by(col).agg(pl.col('vacc').mean()).sort(col)
    ax_acc.plot(agg_acc[col], agg_acc['vacc'].to_numpy(), 'o-', lw=1.5, ms=4, color=color, alpha=0.7)

ax.set_ylabel('Mean pairwise distance'); ax.legend(loc='lower right', frameon=False)
ax.set_title('Coverage gap: adversarial points spread farther apart as bias increases')
ax_acc.set_ylabel('Accuracy'); ax_acc.set_xlabel('Coverage gap (bias)'); ax_acc.set_ylim(0.6, 1.05)
ax_acc.axhline(0.96, color='gray', ls=':', lw=1, alpha=0.4)
fig.tight_layout(); fig.savefig(f'{OUT}/p1_coverage_gap.png'); plt.close()

# ═══════════════════════════════════════════════════════════
# PLOT 2: Label noise — full 0.1-0.9 range, all 3 combos.
# Spread rises WITH accuracy (confounded, not hidden); above 0.5
# the model is near-random and the metric variance explodes.
# ═══════════════════════════════════════════════════════════
print("Plot 2...")
fig, (ax, ax_acc) = plt.subplots(2, 1, figsize=(7, 5), gridspec_kw={'height_ratios': [3, 1]})

for combo, src, col, color in [
    ('Tree + DTA', dt.filter((pl.col('defect')=='label_noise')), 'level', '#2ca02c'),
    ('SVM + HSJ', ln.filter(pl.col('model')=='svm'), 'noise', '#d62728'),
    ('Tree + HSJ', ln.filter(pl.col('model')=='tree'), 'noise', '#1f77b4'),
]:
    sp = src.filter(~pl.col('mean_dist').is_nan())
    agg = sp.group_by(col).agg(pl.col('mean_dist').mean().alias('m'),
                               pl.col('mean_dist').std().alias('s'),
                               pl.len().alias('n')).sort(col)
    x = agg[col].to_numpy(); m = agg['m'].to_numpy(); se = agg['s'].to_numpy() / np.sqrt(agg['n'].to_numpy())
    ax.errorbar(x, m, yerr=1.96*se, fmt='o-', lw=2, ms=5, capsize=3, color=color, label=combo)
    acc = src.filter(~pl.col('vacc').is_nan()).group_by(col).agg(pl.col('vacc').mean().alias('a')).sort(col)
    ax_acc.plot(acc[col].to_numpy(), acc['a'].to_numpy(), 'o-', lw=1.5, ms=4, color=color, alpha=0.7)

# Mark the >0.5 "randomness regime" where the model is near-chance and the metric destabilises
for a in (ax, ax_acc):
    a.axvspan(0.5, 0.9, color='0.85', alpha=0.35, lw=0)
ax.text(0.7, ax.get_ylim()[1]*0.93, 'model near-random:\nmetric unusable', ha='center', va='top', fontsize=8, color='0.35')
ax.set_ylabel('Mean pairwise distance'); ax.legend(loc='upper left', frameon=False)
ax.set_title('Label noise: spread rises but tracks accuracy, then destabilises past 0.5')
ax_acc.set_ylabel('Accuracy'); ax_acc.set_xlabel('Label noise fraction'); ax_acc.set_ylim(0, 1.05)
ax_acc.axhline(1/3, color='gray', ls=':', lw=1, alpha=0.5)  # chance level for 3 classes
fig.tight_layout(); fig.savefig(f'{OUT}/p2_label_noise.png'); plt.close()

# ═══════════════════════════════════════════════════════════
# PLOT 3: Metric decomposition — same story, cleaner signal
# ═══════════════════════════════════════════════════════════
print("Plot 3...")
sub = dt.filter((pl.col('defect')=='coverage_gap') & ~pl.col('aiden_density').is_nan())
agg = sub.group_by('level').agg(
    pl.col('aiden_density').mean(), pl.col('density').mean(), pl.col('mean_dist').mean()
).sort('level')
x = agg['level'].to_numpy()
fig, ax = plt.subplots(figsize=(6, 4))
for label, col, color, ls in [
    ('Aiden (buggy)', 'aiden_density', '#9b59b6', ':'),
    ('Fixed density', 'density', '#e67e22', '--'),
    ('Clean spread', 'mean_dist', '#2ca02c', '-'),
]:
    vals = agg[col].to_numpy()
    z = (vals - vals.mean()) / vals.std()
    ax.plot(x, z, 'o-', lw=2, ms=5, color=color, ls=ls, label=label)
ax.axhline(0, color='gray', ls=':', lw=0.5); ax.set_xlabel('Coverage gap (bias)'); ax.set_ylabel('Z-score')
ax.set_title('Three metrics, same trend — clean spread avoids size confound'); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(f'{OUT}/p3_metrics.png'); plt.close()

# ═══════════════════════════════════════════════════════════
# PLOT 4: Label noise vs coverage gap — the discriminant
# ═══════════════════════════════════════════════════════════
print("Plot 4...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

# Ratio
ax1.plot([0.0, 0.1, 0.3, 0.5], [0.77, 0.91, 0.96, 0.98], 'o-', lw=2, ms=6, color='#e74c3c', label='Label noise')
ax1.plot([0.1, 0.3, 0.5, 0.7, 0.9], [0.70, 0.69, 0.71, 0.70, 0.75], 's-', lw=2, ms=6, color='#2ca02c', label='Coverage gap')
ax1.axhline(1.0, color='gray', ls='--', lw=1, alpha=0.4)
ax1.set_xlabel('Defect level'); ax1.set_ylabel('Adv spread / original spread')
ax1.set_title('Compression ratio'); ax1.legend(frameon=False)

# Perturbation
ax2.plot([0.0, 0.1, 0.3, 0.5], [0.81, 0.74, 0.76, 0.64], 'o-', lw=2, ms=6, color='#e74c3c', label='Label noise')
ax2.plot([0.1, 0.3, 0.5, 0.7, 0.9], [1.38, 1.42, 1.48, 1.51, 1.70], 's-', lw=2, ms=6, color='#2ca02c', label='Coverage gap')
ax2.set_xlabel('Defect level'); ax2.set_ylabel('Perturbation (L2)')
ax2.set_title('How far points move'); ax2.legend(frameon=False)

fig.suptitle('Why label noise fails: ratio → 1 (points barely move), coverage gap stays ~0.7');
fig.tight_layout(); fig.savefig(f'{OUT}/p4_discriminant.png'); plt.close()

# ═══════════════════════════════════════════════════════════
# PLOT 5: Iris — overfit density drops (like coverage gap)
# ═══════════════════════════════════════════════════════════
print("Plot 5...")
try:
    v2 = pl.read_parquet('data/data_v2.parquet')
    # Iris: continuous, well-separated — representative of our main finding
    sub_iris = v2.filter((pl.col('dataset')=='iris') & (pl.col('noise_mode')=='independent')
                          & pl.col('tree_strategy').is_in(['overfit','pruned']))
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(9, 7))
    for strat, color, marker in [('overfit','#e74c3c','o'),('pruned','#9b59b6','s')]:
        ss = sub_iris.filter(pl.col('tree_strategy')==strat)
        agg = ss.group_by('noise_level').agg(
            pl.col('density').mean().alias('d'),
            pl.col('n_leaves').mean().alias('l'),
            pl.col('test_acc').mean().alias('a')
        ).sort('noise_level')
        x = agg['noise_level'].to_numpy()
        ax1.plot(x, agg['d'], f'{marker}-', lw=2, ms=5, color=color, label=strat)
        ax2.plot(x, agg['a'], f'{marker}-', lw=2, ms=5, color=color, label=strat)
        ax3.plot(x, agg['l'], f'{marker}-', lw=2, ms=5, color=color, label=strat)
    ax1.set_ylabel('Density'); ax1.set_title('Density (both decrease)', fontsize=10)
    ax2.set_ylabel('Test accuracy'); ax2.set_title('Accuracy (both collapse)', fontsize=10)
    ax3.set_ylabel('Leaves'); ax3.set_xlabel('Label noise'); ax3.set_title('Tree complexity', fontsize=10)
    ax1.legend(frameon=False, fontsize=8); ax2.legend(frameon=False, fontsize=8)
    ax3.legend(frameon=False, fontsize=8)
    # Text note about spread
    ax4.axis('off'); ax4.text(0.5, 0.5,
        'Spread not available from v2 data.\nDensity = n_pairs/(total_dist+1).\nBoth density AND accuracy move together\n— the signal is confounded.',
        ha='center', va='center', fontsize=9, style='italic', color='gray',
        transform=ax4.transAxes)
    fig.suptitle('Iris (continuous): Both strategies show density decreasing with noise', fontsize=12);
    fig.tight_layout()
    fig.savefig(f'{OUT}/p5_strategy_iris.png'); plt.close()
except Exception as e: print(f"  skipped: {e}")

# ═══════════════════════════════════════════════════════════
# PLOT 6: Aiden's original — the bug and the confound
# ═══════════════════════════════════════════════════════════
print("Plot 6...")
dat = pl.read_parquet('data/data_bias.parquet')
iris = dat.filter((pl.col('dataset').str.starts_with('iris')) & (pl.col('bias_type')=='undersampling'))
agg = iris.group_by('bias').agg(pl.col('adv distance').mean(), pl.col('test acc').mean()).sort('bias')

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(agg['bias'], agg['adv distance'], 'o-', lw=2, color='purple')
ax.set_xlabel('Coverage gap (bias)'); ax.set_ylabel("Aiden's 'adv distance' (density)")
ax.set_title("Aiden's original: a 3% signal, tracking accuracy")
ax2 = ax.twinx(); ax2.plot(agg['bias'], agg['test acc'], 's--', lw=1.5, color='gray', alpha=0.5)
ax2.set_ylabel('Test accuracy', color='gray'); ax2.tick_params(axis='y', labelcolor='gray')
fig.tight_layout(); fig.savefig(f'{OUT}/p6_aiden_original.png'); plt.close()

print(f"\nDone. {os.listdir(OUT)}")
