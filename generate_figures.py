"""Generate publication-quality figures for the mid-year technical report.
Reads from committed results.parquet files. Saves to figures/report/.
"""
import os, sys
for _v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[_v] = '1'

import numpy as np
import polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

OUT = 'figures/report'
os.makedirs(OUT, exist_ok=True)

# Style for academic publication
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.labelsize': 12, 'axes.titlesize': 13,
    'legend.fontsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'figure.dpi': 200, 'savefig.dpi': 200,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
})

# ── Load data ──────────────────────────────────────────────
dt = pl.read_parquet('dtree_attack_experiment/results.parquet')
hs = pl.read_parquet('hsj_svm_experiment/results.parquet')
ln = pl.read_parquet('hsj_label_noise_experiment/results.parquet')

def cohen_d(lo, hi):
    lo, hi = lo[~np.isnan(lo)], hi[~np.isnan(hi)]
    if len(lo) < 3 or len(hi) < 3: return np.nan
    return (hi.mean() - lo.mean()) / np.sqrt((lo.var() + hi.var()) / 2)


# ═══════════════════════════════════════════════════════════
# FIGURE 1: Coverage gap clean spread + accuracy, 3 combos
# ═══════════════════════════════════════════════════════════
print("Generating Figure 1: Coverage gap clean spread...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

combos = [
    (0, dt.filter((pl.col('defect') == 'coverage_gap') & ~pl.col('density').is_nan()),
     'level', 'DecisionTree + DTA', '#2ca02c'),
    (1, hs.filter((pl.col('model') == 'tree') & ~pl.col('density').is_nan()),
     'bias', 'DecisionTree + HSJ', '#2ca02c'),
    (2, hs.filter((pl.col('model') == 'svm') & ~pl.col('density').is_nan()),
     'bias', 'SVM (RBF) + HSJ', '#d62728'),
]

for idx, sub, col, title, c in combos:
    agg = sub.group_by(col).agg(
        pl.col('mean_dist').mean().alias('m'), pl.col('mean_dist').std().alias('s'),
        pl.col('vacc').mean().alias('acc'), pl.len().alias('n')
    ).sort(col)
    x = agg[col].to_numpy(); se = agg['s'].to_numpy() / np.sqrt(agg['n'].to_numpy())

    ax1 = axes[idx]
    ax1.errorbar(x, agg['m'], yerr=1.96*se, fmt='o-', lw=2, ms=6, capsize=4, color=c)
    ax1.set_title(title, fontweight='bold')
    ax1.set_xlabel('Coverage gap (bias)'); ax1.set_ylabel('Mean pairwise distance', color=c)
    ax1.tick_params(axis='y', labelcolor=c)

    ax2 = ax1.twinx()
    ax2.plot(x, agg['acc'], 's--', color='#1f77b4', lw=1.5, ms=5, alpha=0.6)
    ax2.set_ylabel('Test accuracy', color='#1f77b4')
    ax2.tick_params(axis='y', labelcolor='#1f77b4')
    ax2.set_ylim(0.5, 1.05)

    # Cohen's d annotation
    lo = sub.filter(pl.col(col) == sub[col].min()); hi = sub.filter(pl.col(col) == sub[col].max())
    d_val = cohen_d(lo['mean_dist'].to_numpy(), hi['mean_dist'].to_numpy())
    d_acc = cohen_d(lo['vacc'].to_numpy(), hi['vacc'].to_numpy())
    axes[idx].text(0.05, 0.05, f"mean_dist d = {d_val:+.2f}\naccuracy d = {d_acc:+.2f}",
                   transform=axes[idx].transAxes, fontsize=8, va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

fig.suptitle('Coverage Gap: Clean Spread Increases Across All Model-Attack Combinations,\nWhile Test Accuracy Stays Flat',
             fontweight='bold')
fig.tight_layout()
fig.savefig(f'{OUT}/fig1_coverage_gap_clean_spread.png', dpi=200)
plt.close()
print("  -> saved fig1_coverage_gap_clean_spread.png")

# ═══════════════════════════════════════════════════════════
# FIGURE 2: Label noise — density + accuracy, dual axes
# ═══════════════════════════════════════════════════════════
print("\nGenerating Figure 2: Label noise density vs accuracy...")
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# Panel A: Tree + DTA
sub = dt.filter((pl.col('defect') == 'label_noise') & ~pl.col('density').is_nan())
agg = sub.group_by('level').agg(
    pl.col('mean_dist').mean().alias('m'), pl.col('mean_dist').std().alias('s'),
    pl.col('vacc').mean().alias('acc'), pl.len().alias('n')
).sort('level')
x = agg['level'].to_numpy(); se = agg['s'].to_numpy() / np.sqrt(agg['n'].to_numpy())

ax = axes[0]
ax.errorbar(x, agg['m'], yerr=1.96*se, fmt='o-', lw=2, ms=6, capsize=4, color='#2ca02c')
ax.set_ylabel('Mean pairwise distance', color='#2ca02c'); ax.tick_params(axis='y', labelcolor='#2ca02c')
ax2 = ax.twinx()
ax2.plot(x, agg['acc'], 's--', color='#d62728', lw=2, ms=6, label='Test Accuracy')
ax2.set_ylabel('Test accuracy', color='#d62728'); ax2.tick_params(axis='y', labelcolor='#d62728')
ax.set_xlabel('Label noise level'); ax.set_title('DecisionTree + DTA', fontweight='bold')

lo = sub.filter(pl.col('level') == 0.1); hi = sub.filter(pl.col('level') == 0.5)
d_md = cohen_d(lo['mean_dist'].to_numpy(), hi['mean_dist'].to_numpy())
d_acc = cohen_d(lo['vacc'].to_numpy(), hi['vacc'].to_numpy())
ax.text(0.95, 0.95, f"mean_dist d = {d_md:+.2f}\naccuracy d = {d_acc:+.2f}",
        transform=ax.transAxes, fontsize=8, va='top', ha='right',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
ax2.legend(loc='lower left', fontsize=8)

# Panel B: SVM + HSJ
s2 = ln.filter((pl.col('model') == 'svm') & ~pl.col('density').is_nan())
agg2 = s2.group_by('noise').agg(
    pl.col('mean_dist').mean().alias('m'), pl.col('mean_dist').std().alias('s'),
    pl.col('vacc').mean().alias('acc'), pl.len().alias('n')
).sort('noise')
x2 = agg2['noise'].to_numpy(); se2 = agg2['s'].to_numpy() / np.sqrt(agg2['n'].to_numpy())

ax = axes[1]
ax.errorbar(x2, agg2['m'], yerr=1.96*se2, fmt='o-', lw=2, ms=6, capsize=4, color='#d62728')
ax.set_ylabel('Mean pairwise distance', color='#d62728'); ax.tick_params(axis='y', labelcolor='#d62728')
ax3 = ax.twinx()
ax3.plot(x2, agg2['acc'], 's--', color='#1f77b4', lw=2, ms=6, label='Test Accuracy')
ax3.set_ylabel('Test accuracy', color='#1f77b4'); ax3.tick_params(axis='y', labelcolor='#1f77b4')
ax.set_xlabel('Label noise level'); ax.set_title('SVM (RBF) + HSJ', fontweight='bold')
ax3.legend(loc='lower left', fontsize=8)

lo = s2.filter(pl.col('noise') == 0.1); hi = s2.filter(pl.col('noise') == 0.5)
d_md = cohen_d(lo['mean_dist'].to_numpy(), hi['mean_dist'].to_numpy())
d_acc = cohen_d(lo['vacc'].to_numpy(), hi['vacc'].to_numpy())
ax.text(0.95, 0.95, f"mean_dist d = {d_md:+.2f}\naccuracy d = {d_acc:+.2f}",
        transform=ax.transAxes, fontsize=8, va='top', ha='right',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

fig.suptitle('Label Noise: Spread Increases Alongside Accuracy Collapse,\nMaking the Geometric Signal Redundant', fontweight='bold')
fig.tight_layout()
fig.savefig(f'{OUT}/fig2_label_noise_accuracy_confound.png', dpi=200)
plt.close()
print("  -> saved fig2_label_noise_accuracy_confound.png")

# ═══════════════════════════════════════════════════════════
# FIGURE 3: Metric decomposition (DT+DTA, coverage gap)
# ═══════════════════════════════════════════════════════════
print("\nGenerating Figure 3: Metric decomposition...")
sub = dt.filter((pl.col('defect') == 'coverage_gap') & ~pl.col('aiden_density').is_nan())
agg3 = sub.group_by('level').agg(
    pl.col('aiden_density').mean().alias('aiden'), pl.col('density').mean().alias('density'),
    pl.col('mean_dist').mean().alias('mean_dist'), pl.col('clust_size').mean().alias('csize'),
).sort('level')

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# Left: all three metrics z-scored
x = agg3['level'].to_numpy()
metrics = {'aiden_density (buggy)': ('aiden', 'purple', 's--'),
           'density (fixed)': ('density', '#e67e22', 'o-'),
           'mean_dist (clean spread)': ('mean_dist', '#2ca02c', 'D-')}
for label, (col, color, fmt) in metrics.items():
    vals = agg3[col].to_numpy()
    z = (vals - vals.mean()) / vals.std()
    axes[0].plot(x, z, fmt, color=color, lw=2, ms=6, label=label)
axes[0].axhline(0, color='gray', ls=':', lw=1)
axes[0].set_xlabel('Coverage gap (bias)'); axes[0].set_ylabel('Z-score')
axes[0].set_title('Metric Comparison (z-scored)', fontweight='bold')
axes[0].legend(fontsize=8)

# Right: density components
c1 = '#e74c3c'; c2 = '#2980b9'
axes[1].bar(x - 0.03, agg3['csize'].to_numpy(), 0.06, color=c1, alpha=0.7, label='Cluster size')
ax2b = axes[1].twinx()
ax2b.bar(x + 0.03, agg3['mean_dist'].to_numpy(), 0.06, color=c2, alpha=0.7, label='Mean distance')
axes[1].set_xlabel('Coverage gap (bias)'); axes[1].set_ylabel('Points per cluster', color=c1)
ax2b.set_ylabel('Mean pairwise distance', color=c2)
axes[1].set_title('Density Components: Both Move Together', fontweight='bold')
h1, l1 = axes[1].get_legend_handles_labels(); h2, l2 = ax2b.get_legend_handles_labels()
axes[1].legend(h1+h2, l1+l2, fontsize=8)

fig.suptitle('Metric Decomposition: Aiden\'s Buggy Metric vs. Fixed vs. Clean Spread', fontweight='bold')
fig.tight_layout()
fig.savefig(f'{OUT}/fig3_metric_decomposition.png', dpi=200)
plt.close()
print("  -> saved fig3_metric_decomposition.png")

# ═══════════════════════════════════════════════════════════
# FIGURE 4: Clean spread Cohen's d comparison table as figure
# ═══════════════════════════════════════════════════════════
print("\nGenerating Figure 4: Cohen's d summary...")
rows_data = []
# Coverage gap
for label, src, col, def_val in [('Tree+DTA', dt, 'level', 'coverage_gap'),
                                   ('Tree+HSJ', hs, 'bias', 'tree'),
                                   ('SVM+HSJ', hs, 'bias', 'svm')]:
    if def_val == 'coverage_gap':
        sub = src.filter((pl.col('defect') == def_val) & ~pl.col('density').is_nan())
    else:
        sub = src.filter((pl.col('model') == def_val) & ~pl.col('density').is_nan())
    lo_level, hi_level = 0.1, 0.9
    lo = sub.filter(pl.col(col) == lo_level); hi = sub.filter(pl.col(col) == hi_level)
    for metric, mcol in [('Aiden density', 'aiden_density'), ('Fixed density', 'density'),
                          ('Clean spread', 'mean_dist'), ('Accuracy', 'vacc')]:
        d_val = cohen_d(lo[mcol].to_numpy(), hi[mcol].to_numpy())
        rows_data.append({'Combination': label, 'Defect': 'Coverage gap',
                          'Metric': metric, "Cohen's d": d_val})

# Label noise
for label, src, col in [('Tree+DTA', dt, 'level'), ('Tree+HSJ', ln, 'noise'), ('SVM+HSJ', ln, 'noise')]:
    if label == 'Tree+DTA':
        sub = src.filter((pl.col('defect') == 'label_noise') & ~pl.col('density').is_nan())
    else:
        m = 'tree' if label.startswith('Tree') else 'svm'
        sub = src.filter((pl.col('model') == m) & ~pl.col('density').is_nan())
    lo_level, hi_level = 0.1, 0.5
    lo = sub.filter(pl.col(col) == lo_level); hi = sub.filter(pl.col(col) == hi_level)
    for metric, mcol in [('Aiden density', 'aiden_density'), ('Fixed density', 'density'),
                          ('Clean spread', 'mean_dist'), ('Accuracy', 'vacc')]:
        d_val = cohen_d(lo[mcol].to_numpy(), hi[mcol].to_numpy())
        rows_data.append({'Combination': label, 'Defect': 'Label noise',
                          'Metric': metric, "Cohen's d": d_val})

df_cd = pl.DataFrame(rows_data)

# Create table figure
fig, ax = plt.subplots(figsize=(12, 5))
ax.axis('off')

# Color coding
def color_d(val):
    if val is None or np.isnan(val): return '#cccccc'
    v = abs(val)
    if v > 1.5: return '#27ae60'  # strong
    elif v > 0.5: return '#f39c12'  # moderate
    else: return '#e74c3c'  # weak

# Build table
table_data = [['Combination', 'Defect', 'Metric', "Cohen's d"]]
cell_colors = [['#ecf0f1']*4]
for r in rows_data:
    d_val = r["Cohen's d"]
    d_str = f'{d_val:+.2f}' if not np.isnan(d_val) else 'N/A'
    table_data.append([r['Combination'], r['Defect'], r['Metric'], d_str])
    cell_colors.append(['white', 'white', 'white', color_d(d_val)])

table = ax.table(cellText=table_data, cellColours=cell_colors,
                  colWidths=[0.2, 0.18, 0.22, 0.15],
                  loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.6)

# Bold header
for j in range(4):
    table[0, j].set_text_props(fontweight='bold', fontsize=11)

ax.set_title("Cohen's d Effect Sizes: Low vs High Defect (0.1→0.9 coverage gap; 0.1→0.5 label noise)",
             fontweight='bold', pad=20)

fig.savefig(f'{OUT}/fig4_cohens_d_table.png', dpi=200)
plt.close()
print("  -> saved fig4_cohens_d_table.png")

# ═══════════════════════════════════════════════════════════
# FIGURE 5: Geometric discriminant — ratio + perturbation
# ═══════════════════════════════════════════════════════════
print("\nGenerating Figure 5: Geometric discriminant...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

# Panel A: Ratio adv_spread/orig_spread
# Label noise values from _probe_move.py output
ln_ratio = [0.77, 0.91, 0.96, 0.99, 0.98]  # noise 0.0, 0.1, 0.3, 0.4, 0.5
ln_noise = [0.0, 0.1, 0.3, 0.4, 0.5]
# Coverage gap values from _probe_cg.py output
cg_ratio = [0.70, 0.69, 0.71, 0.70, 0.75]  # bias 0.1, 0.3, 0.5, 0.7, 0.9
cg_bias = [0.1, 0.3, 0.5, 0.7, 0.9]

ax1.plot(ln_noise, ln_ratio, 'o-', lw=2, ms=8, color='#e74c3c', label='Label noise (Tree+DTA)')
ax1.plot(cg_bias, cg_ratio, 's-', lw=2, ms=8, color='#2ca02c', label='Coverage gap (SVM+HSJ)')
ax1.axhline(1.0, color='gray', ls='--', lw=1, label='Original test-point spread')
ax1.set_xlabel('Defect level'); ax1.set_ylabel('adv_spread / orig_spread')
ax1.set_title('Adversarial Cloud Compression Ratio', fontweight='bold')
ax1.legend(fontsize=8)
ax1.set_ylim(0.6, 1.1)
ax1.annotate('Converges to 1.0\n(artifact)', xy=(0.5, 0.98), fontsize=8, color='#e74c3c',
             ha='center', va='bottom')
ax1.annotate('Stays ~0.70\n(real signal)', xy=(0.5, 0.72), fontsize=8, color='#2ca02c',
             ha='center', va='top')

# Panel B: Perturbation magnitude
ln_perturb = [0.81, 0.74, 0.76, 0.68, 0.64]  # noise 0.0, 0.1, 0.3, 0.4, 0.5
cg_perturb = [1.38, 1.42, 1.48, 1.51, 1.70]  # bias

ax2.plot(ln_noise, ln_perturb, 'o-', lw=2, ms=8, color='#e74c3c', label='Label noise (Tree+DTA)')
ax2.plot(cg_bias, cg_perturb, 's-', lw=2, ms=8, color='#2ca02c', label='Coverage gap (SVM+HSJ)')
ax2.set_xlabel('Defect level'); ax2.set_ylabel('Mean perturbation (L2)')
ax2.set_title('Adversarial Perturbation Magnitude', fontweight='bold')
ax2.legend(fontsize=8)

fig.suptitle('Geometric Discriminant: Label Noise vs Coverage Gap', fontweight='bold')
fig.tight_layout()
fig.savefig(f'{OUT}/fig5_geometric_discriminant.png', dpi=200)
plt.close()
print("  -> saved fig5_geometric_discriminant.png")

# ═══════════════════════════════════════════════════════════
# FIGURE 6: Tree strategy — overfit vs pruned direction flip
# ═══════════════════════════════════════════════════════════
print("\nGenerating Figure 6: Tree strategy comparison...")
try:
    v2 = pl.read_parquet('data/data_v2.parquet')
    if 'density_z' not in v2.columns:
        v2 = v2.with_columns(
            ((pl.col('density') - pl.col('density').mean()) / pl.col('density').std())
            .over(['dataset', 'tree_strategy', 'noise_mode']).alias('density_z')
        )

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Left: overfit vs pruned density z-score (Car Evaluation)
    sub = v2.filter((pl.col('dataset') == 'Car Evaluation') & (pl.col('noise_mode') == 'independent'))

    for strat, color, marker in [('overfit', '#e74c3c', 'o'), ('pruned', '#9b59b6', 's')]:
        ss = sub.filter(pl.col('tree_strategy') == strat)
        agg = ss.group_by('noise_level').agg(
            pl.col('density_z').mean().alias('z'),
            pl.col('density_z').std().alias('zs'),
            pl.len().alias('n')
        ).sort('noise_level')
        x = agg['noise_level'].to_numpy(); se = agg['zs'].to_numpy() / np.sqrt(agg['n'].to_numpy())
        axes[0].errorbar(x, agg['z'], yerr=1.96*se, fmt=f'{marker}-', lw=2, ms=6,
                         capsize=3, color=color, label=strat)
    axes[0].axhline(0, color='gray', ls=':', lw=1)
    axes[0].set_xlabel('Label noise level'); axes[0].set_ylabel('Density (z-score)')
    axes[0].set_title('Car Evaluation: Density vs Label Noise', fontweight='bold')
    axes[0].legend(fontsize=9)

    # Right: n_leaves vs noise
    for strat, color, marker in [('overfit', '#e74c3c', 'o'), ('pruned', '#9b59b6', 's')]:
        ss = sub.filter(pl.col('tree_strategy') == strat)
        agg = ss.group_by('noise_level').agg(
            pl.col('n_leaves').mean().alias('n_leaves'),
            pl.col('n_leaves').std().alias('s'),
            pl.len().alias('n')
        ).sort('noise_level')
        x = agg['noise_level'].to_numpy(); se = agg['s'].to_numpy() / np.sqrt(agg['n'].to_numpy())
        axes[1].errorbar(x, agg['n_leaves'], yerr=1.96*se, fmt=f'{marker}-', lw=2, ms=6,
                         capsize=3, color=color, label=strat)
    axes[1].set_xlabel('Label noise level'); axes[1].set_ylabel('Number of leaves')
    axes[1].set_title('Tree Complexity vs Label Noise', fontweight='bold')
    axes[1].legend(fontsize=9)

    fig.suptitle('Signal Direction Depends on Tree Training Strategy:\nOverfit vs Pruned Show Opposite Trends', fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig6_tree_strategy_flip.png', dpi=200)
    plt.close()
    print("  -> saved fig6_tree_strategy_flip.png")
except Exception as e:
    print(f"  -> SKIPPED: {e}")

print(f"\n{'='*60}")
print(f"All figures saved to {OUT}/")
print(f"Files: {os.listdir(OUT)}")
