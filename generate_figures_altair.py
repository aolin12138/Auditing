"""Generate report figures using Altair (Vega-Lite) — Aiden's original style.
Saves as PNG via altair_saver or PNG export.
"""
import os, sys
import polars as pl
import altair as alt
import numpy as np

OUT = 'figures/report'
os.makedirs(OUT, exist_ok=True)

# ── Load data ──
dt = pl.read_parquet('dtree_attack_experiment/results.parquet')
hs = pl.read_parquet('hsj_svm_experiment/results.parquet')
ln = pl.read_parquet('hsj_label_noise_experiment/results.parquet')

def cohen_d(lo, hi):
    lo, hi = lo[~np.isnan(lo)], hi[~np.isnan(hi)]
    if len(lo) < 3 or len(hi) < 3: return np.nan
    return (hi.mean() - lo.mean()) / np.sqrt((lo.var() + hi.var()) / 2)


# ═══════════════════════════════════════════════════════════
# FIGURE 1: Coverage gap — clean spread (3 combos faceted)
# ═══════════════════════════════════════════════════════════
print("Figure 1: Coverage gap clean spread...")

# Build unified dataframe for faceting
rows = []
# Tree + DTA, coverage gap
sub = dt.filter((pl.col('defect') == 'coverage_gap') & ~pl.col('mean_dist').is_nan())
agg = sub.group_by('level').agg(
    pl.col('mean_dist').mean().alias('mean'), pl.col('mean_dist').std().alias('std'),
    pl.col('vacc').mean().alias('accuracy'), pl.len().alias('n'),
    pl.col('mean_dist').mean().alias('density_component')
).sort('level')
agg = agg.with_columns(pl.lit('DecisionTree + DTA').alias('Combination'))
for r in agg.to_dicts():
    ci = 1.96 * r['std'] / np.sqrt(r['n'])
    rows.append({'Combination': r['Combination'], 'Defect Level': r['level'],
                 'Mean Distance': r['mean'], 'CI': ci, 'Accuracy': r['accuracy']})

# Tree + HSJ
sub2 = hs.filter((pl.col('model') == 'tree') & ~pl.col('mean_dist').is_nan())
agg2 = sub2.group_by('bias').agg(
    pl.col('mean_dist').mean().alias('mean'), pl.col('mean_dist').std().alias('std'),
    pl.col('vacc').mean().alias('accuracy'), pl.len().alias('n')
).sort('bias')
agg2 = agg2.with_columns(pl.lit('DecisionTree + HSJ').alias('Combination'))
for r in agg2.to_dicts():
    ci = 1.96 * r['std'] / np.sqrt(r['n'])
    rows.append({'Combination': r['Combination'], 'Defect Level': r['bias'],
                 'Mean Distance': r['mean'], 'CI': ci, 'Accuracy': r['accuracy']})

# SVM + HSJ
sub3 = hs.filter((pl.col('model') == 'svm') & ~pl.col('mean_dist').is_nan())
agg3 = sub3.group_by('bias').agg(
    pl.col('mean_dist').mean().alias('mean'), pl.col('mean_dist').std().alias('std'),
    pl.col('vacc').mean().alias('accuracy'), pl.len().alias('n')
).sort('bias')
agg3 = agg3.with_columns(pl.lit('SVM (RBF) + HSJ').alias('Combination'))
for r in agg3.to_dicts():
    ci = 1.96 * r['std'] / np.sqrt(r['n'])
    rows.append({'Combination': r['Combination'], 'Defect Level': r['bias'],
                 'Mean Distance': r['mean'], 'CI': ci, 'Accuracy': r['accuracy']})

df_plot = pl.DataFrame(rows)

# Cohen's d annotations
annotations = []
for combo, src, col in [('DecisionTree + DTA', sub, 'level'),
                          ('DecisionTree + HSJ', sub2, 'bias'),
                          ('SVM (RBF) + HSJ', sub3, 'bias')]:
    lo = src.filter(pl.col(col) == src[col].min())
    hi = src.filter(pl.col(col) == src[col].max())
    d_md = cohen_d(lo['mean_dist'].to_numpy(), hi['mean_dist'].to_numpy())
    d_acc = cohen_d(lo['vacc'].to_numpy(), hi['vacc'].to_numpy())
    annotations.append({'Combination': combo, 'label': f'd = {d_md:+.2f} (acc d = {d_acc:+.2f})'})

ann_df = pl.DataFrame(annotations)

# Main chart
base = alt.Chart(df_plot.to_pandas()).encode(
    x=alt.X('Defect Level:Q', title='Coverage Gap (bias)', scale=alt.Scale(domain=[0.05, 0.95]))
)

# Error bars for clean spread
error_bars = base.mark_errorbar(extent='ci').encode(
    y=alt.Y('Mean Distance:Q', title='Mean Pairwise Distance (clean spread)'),
    yError=alt.YError('CI:Q')
)

# Line + points for spread
spread_line = base.mark_line(point=True, strokeWidth=2).encode(
    y='Mean Distance:Q',
    color=alt.Color('Combination:N', legend=alt.Legend(title=None))
)

# Accuracy overlay (dashed)
acc_line = base.mark_line(point=alt.OverlayMarkDef(filled=False, fill='white'),
                           strokeDash=[4, 4], strokeWidth=1.5, opacity=0.5).encode(
    y=alt.Y('Accuracy:Q', title='Test Accuracy', scale=alt.Scale(domain=[0.6, 1.05])),
    color='Combination:N'
)

chart1 = alt.layer(spread_line, acc_line, data=df_plot.to_pandas()).facet(
    facet=alt.Facet('Combination:N', title=None, sort=None),
    columns=3
).resolve_scale(y='independent').properties(
    title=alt.TitleParams(
        'Coverage Gap: Clean Spread Increases While Accuracy Stays Flat',
        subtitle='Mean pairwise distance with 95% CI error bars. Dashed lines show test accuracy.',
        fontSize=14
    )
)

chart1.save(f'{OUT}/fig1_altair_coverage_gap.html')
chart1.save(f'{OUT}/fig1_altair_coverage_gap.png', scale_factor=2)
print("  -> saved fig1_altair_coverage_gap.png")


# ═══════════════════════════════════════════════════════════
# FIGURE 2: Label noise — accuracy confound
# ═══════════════════════════════════════════════════════════
print("Figure 2: Label noise accuracy confound...")

rows2 = []
# Tree + DTA, label noise
s = dt.filter((pl.col('defect') == 'label_noise') & ~pl.col('mean_dist').is_nan())
agg = s.group_by('level').agg(
    pl.col('mean_dist').mean().alias('mean'), pl.col('mean_dist').std().alias('std'),
    pl.col('vacc').mean().alias('accuracy'), pl.len().alias('n')
).sort('level')
agg = agg.with_columns(pl.lit('DecisionTree + DTA').alias('Combination'))
for r in agg.to_dicts():
    ci = 1.96 * r['std'] / np.sqrt(r['n'])
    rows2.append({'Combination': r['Combination'], 'Noise Level': r['level'],
                  'Mean Distance': r['mean'], 'CI': ci, 'Accuracy': r['accuracy']})

# SVM + HSJ
s2 = ln.filter((pl.col('model') == 'svm') & ~pl.col('mean_dist').is_nan())
agg2 = s2.group_by('noise').agg(
    pl.col('mean_dist').mean().alias('mean'), pl.col('mean_dist').std().alias('std'),
    pl.col('vacc').mean().alias('accuracy'), pl.len().alias('n')
).sort('noise')
agg2 = agg2.with_columns(pl.lit('SVM (RBF) + HSJ').alias('Combination'))
for r in agg2.to_dicts():
    ci = 1.96 * r['std'] / np.sqrt(r['n'])
    rows2.append({'Combination': r['Combination'], 'Noise Level': r['noise'],
                  'Mean Distance': r['mean'], 'CI': ci, 'Accuracy': r['accuracy']})

df2 = pl.DataFrame(rows2)

base2 = alt.Chart(df2.to_pandas())

spread2 = base2.mark_line(point=True, strokeWidth=2).encode(
    x=alt.X('Noise Level:Q', title='Label Noise Level', scale=alt.Scale(domain=[0.05, 0.55])),
    y=alt.Y('Mean Distance:Q', title='Mean Pairwise Distance'),
    color='Combination:N'
)

acc2 = base2.mark_line(point=True, strokeDash=[4, 4], strokeWidth=1.5, opacity=0.5).encode(
    x='Noise Level:Q',
    y=alt.Y('Accuracy:Q', title='Test Accuracy', scale=alt.Scale(domain=[0.5, 1.05])),
    color='Combination:N'
)

chart2 = alt.layer(spread2, acc2).facet(
    facet=alt.Facet('Combination:N', title=None),
    columns=2
).resolve_scale(y='independent').properties(
    title=alt.TitleParams(
        'Label Noise: Spread Rises Alongside Accuracy Collapse',
        subtitle='Solid lines = mean pairwise distance (clean spread). Dashed = test accuracy. Both move together.',
        fontSize=14
    )
)

chart2.save(f'{OUT}/fig2_altair_label_noise.html')
chart2.save(f'{OUT}/fig2_altair_label_noise.png', scale_factor=2)
print("  -> saved fig2_altair_label_noise.png")


# ═══════════════════════════════════════════════════════════
# FIGURE 3: Metric decomposition (DT+DTA, coverage gap)
# ═══════════════════════════════════════════════════════════
print("Figure 3: Metric decomposition...")

sub = dt.filter((pl.col('defect') == 'coverage_gap') & ~pl.col('aiden_density').is_nan())
agg_met = sub.group_by('level').agg(
    pl.col('aiden_density').mean().alias('Aiden (buggy)'),
    pl.col('density').mean().alias('Fixed Density'),
    pl.col('mean_dist').mean().alias('Clean Spread'),
)

# Melt to long format for Altair
rows3 = []
for r in agg_met.to_dicts():
    for metric in ['Aiden (buggy)', 'Fixed Density', 'Clean Spread']:
        rows3.append({'Bias': r['level'], 'Metric': metric, 'Value': r[metric]})
df3 = pl.DataFrame(rows3)

# Z-score per metric
df3 = df3.with_columns(
    ((pl.col('Value') - pl.col('Value').mean()) / pl.col('Value').std())
    .over('Metric').alias('Z-Score')
)

chart3 = alt.Chart(df3.to_pandas()).mark_line(point=True, strokeWidth=2).encode(
    x=alt.X('Bias:Q', title='Coverage Gap (bias)', scale=alt.Scale(domain=[0.05, 0.95])),
    y=alt.Y('Z-Score:Q', title='Z-Score'),
    color=alt.Color('Metric:N', scale=alt.Scale(
        domain=['Aiden (buggy)', 'Fixed Density', 'Clean Spread'],
        range=['#9b59b6', '#e67e22', '#2ca02c']
    ))
).properties(
    width=400, height=300,
    title=alt.TitleParams(
        'Metric Comparison (z-scored): Tree + DTA, Coverage Gap',
        subtitle='All three show the same directional trend. Clean spread avoids cluster-size confound.',
        fontSize=14
    )
)

chart3.save(f'{OUT}/fig3_altair_metric_decomp.html')
chart3.save(f'{OUT}/fig3_altair_metric_decomp.png', scale_factor=2)
print("  -> saved fig3_altair_metric_decomp.png")


# ═══════════════════════════════════════════════════════════
# FIGURE 4: Geometric discriminant — ratio + perturbation
# ═══════════════════════════════════════════════════════════
print("Figure 4: Geometric discriminant...")

# Probe results (hardcoded from _probe_move.py and _probe_cg.py output)
probe_rows = [
    # Label noise: Tree+DTA
    {'Defect': 'Label Noise', 'Defect Level': 0.0, 'Metric': 'adv_spread / orig_spread', 'Value': 0.77},
    {'Defect': 'Label Noise', 'Defect Level': 0.1, 'Metric': 'adv_spread / orig_spread', 'Value': 0.91},
    {'Defect': 'Label Noise', 'Defect Level': 0.3, 'Metric': 'adv_spread / orig_spread', 'Value': 0.96},
    {'Defect': 'Label Noise', 'Defect Level': 0.5, 'Metric': 'adv_spread / orig_spread', 'Value': 0.98},
    # Coverage gap: SVM+HSJ
    {'Defect': 'Coverage Gap', 'Defect Level': 0.1, 'Metric': 'adv_spread / orig_spread', 'Value': 0.70},
    {'Defect': 'Coverage Gap', 'Defect Level': 0.3, 'Metric': 'adv_spread / orig_spread', 'Value': 0.69},
    {'Defect': 'Coverage Gap', 'Defect Level': 0.5, 'Metric': 'adv_spread / orig_spread', 'Value': 0.71},
    {'Defect': 'Coverage Gap', 'Defect Level': 0.7, 'Metric': 'adv_spread / orig_spread', 'Value': 0.70},
    {'Defect': 'Coverage Gap', 'Defect Level': 0.9, 'Metric': 'adv_spread / orig_spread', 'Value': 0.75},
    # Perturbation: Label noise
    {'Defect': 'Label Noise', 'Defect Level': 0.0, 'Metric': 'Perturbation (L2)', 'Value': 0.81},
    {'Defect': 'Label Noise', 'Defect Level': 0.1, 'Metric': 'Perturbation (L2)', 'Value': 0.74},
    {'Defect': 'Label Noise', 'Defect Level': 0.3, 'Metric': 'Perturbation (L2)', 'Value': 0.76},
    {'Defect': 'Label Noise', 'Defect Level': 0.5, 'Metric': 'Perturbation (L2)', 'Value': 0.64},
    # Perturbation: Coverage gap
    {'Defect': 'Coverage Gap', 'Defect Level': 0.1, 'Metric': 'Perturbation (L2)', 'Value': 1.38},
    {'Defect': 'Coverage Gap', 'Defect Level': 0.3, 'Metric': 'Perturbation (L2)', 'Value': 1.42},
    {'Defect': 'Coverage Gap', 'Defect Level': 0.5, 'Metric': 'Perturbation (L2)', 'Value': 1.48},
    {'Defect': 'Coverage Gap', 'Defect Level': 0.7, 'Metric': 'Perturbation (L2)', 'Value': 1.51},
    {'Defect': 'Coverage Gap', 'Defect Level': 0.9, 'Metric': 'Perturbation (L2)', 'Value': 1.70},
]
df4 = pl.DataFrame(probe_rows)

chart4 = alt.Chart(df4.to_pandas()).mark_line(point=True, strokeWidth=2).encode(
    x=alt.X('Defect Level:Q', title='Defect Level'),
    y=alt.Y('Value:Q', title=None),
    color=alt.Color('Defect:N', scale=alt.Scale(
        domain=['Label Noise', 'Coverage Gap'],
        range=['#e74c3c', '#2ca02c']
    ))
).facet(
    facet=alt.Facet('Metric:N', title=None),
    columns=2
).resolve_scale(y='independent').properties(
    title=alt.TitleParams(
        'Geometric Discriminant: Label Noise vs Coverage Gap',
        subtitle='Left: Ratio of adversarial spread to original test-point spread (→1 = artifact). Right: Perturbation magnitude (L2).',
        fontSize=14
    )
)

chart4.save(f'{OUT}/fig4_altair_discriminant.html')
chart4.save(f'{OUT}/fig4_altair_discriminant.png', scale_factor=2)
print("  -> saved fig4_altair_discriminant.png")


# ═══════════════════════════════════════════════════════════
# FIGURE 5: Tree strategy flip
# ═══════════════════════════════════════════════════════════
print("Figure 5: Tree strategy flip...")
try:
    v2 = pl.read_parquet('data/data_v2.parquet')
    if 'density_z' not in v2.columns:
        v2 = v2.with_columns(
            ((pl.col('density') - pl.col('density').mean()) / pl.col('density').std())
            .over(['dataset', 'tree_strategy', 'noise_mode']).alias('density_z')
        )

    sub_car = v2.filter((pl.col('dataset') == 'Car Evaluation') & (pl.col('noise_mode') == 'independent'))
    agg_car = sub_car.group_by(['noise_level', 'tree_strategy']).agg(
        pl.col('density_z').mean().alias('Density (z)'),
        pl.col('n_leaves').mean().alias('Leaves'),
    ).filter(pl.col('tree_strategy').is_in(['overfit', 'pruned']))

    # Density panel
    density_chart = alt.Chart(agg_car.to_pandas()).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X('noise_level:Q', title='Label Noise Level'),
        y=alt.Y('Density (z):Q', title='Density (z-score)'),
        color=alt.Color('tree_strategy:N', legend=alt.Legend(title='Strategy'),
                        scale=alt.Scale(domain=['overfit', 'pruned'],
                                        range=['#e74c3c', '#9b59b6']))
    ).properties(title='Density (z-score)', width=250, height=200)

    # Leaves panel
    leaves_chart = alt.Chart(agg_car.to_pandas()).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X('noise_level:Q', title='Label Noise Level'),
        y=alt.Y('Leaves:Q', title='Number of Leaves'),
        color='tree_strategy:N'
    ).properties(title='Tree Complexity', width=250, height=200)

    chart5 = alt.hconcat(density_chart, leaves_chart).resolve_scale(color='shared').properties(
        title=alt.TitleParams(
            'Signal Direction Depends on Tree Strategy: Overfit vs Pruned',
            subtitle='Car Evaluation dataset, independent noise injection. Overfit trees grow; pruned trees shrink.',
            fontSize=14
        )
    )

    chart5.save(f'{OUT}/fig5_altair_tree_strategy.html')
    chart5.save(f'{OUT}/fig5_altair_tree_strategy.png', scale_factor=2)
    print("  -> saved fig5_altair_tree_strategy.png")
except Exception as e:
    print(f"  -> SKIPPED (data_v2.parquet not available): {e}")

print(f"\nAll Altair figures saved to {OUT}/")
