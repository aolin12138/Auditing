"""Replicate the repo's coverage-gap spread analysis on the MLP export.
Repo method (model_family_experiment/plot_cg.py): filter defect==coverage_gap,
group by bias, mean of mean_dist (drop NaN). Repo injection protocol = before_split.
"""
import numpy as np, polars as pl
from pathlib import Path

HERE = Path(__file__).resolve().parent
d = pl.read_csv(HERE / 'export' / 'mlp_results_all.csv')
cg = d.filter(pl.col('defect') == 'coverage_gap')

print("=== coverage_gap grid inventory ===")
print("rows:", cg.height)
for c in ['protocol', 'attack', 'tc', 'feat', 'bias', 'seed']:
    vals = cg[c].unique().sort().to_list()
    print(f"  {c}: {vals}")

BIAS = sorted([b for b in cg['bias'].unique().to_list()])

def curve(sub, col='mean_dist'):
    out = []
    for b in BIAS:
        v = sub.filter(pl.col('bias') == b)[col].drop_nans().drop_nulls()
        m = float(v.mean()) if len(v) else float('nan')
        n = len(v)
        se = 1.96*float(v.std())/np.sqrt(n) if n > 1 else 0.0
        out.append((b, m, n, se))
    return out

def cohens_d(sub, col='mean_dist', lo=0.1, hi=0.9):
    a = sub.filter(pl.col('bias') == lo)[col].drop_nans().drop_nulls().to_numpy()
    b = sub.filter(pl.col('bias') == hi)[col].drop_nans().drop_nulls().to_numpy()
    if len(a) < 2 or len(b) < 2: return float('nan')
    sp = np.sqrt(((len(a)-1)*a.std(ddof=1)**2 + (len(b)-1)*b.std(ddof=1)**2)/(len(a)+len(b)-2))
    return (b.mean()-a.mean())/sp if sp > 0 else float('nan')

for proto in ['before_split', 'train_only']:
    for atk in ['deepfool', 'hsj']:
        sub = cg.filter((pl.col('protocol') == proto) & (pl.col('attack') == atk))
        if sub.height == 0:
            continue
        print(f"\n=== protocol={proto}  attack={atk}  (rows={sub.height}) ===")
        print(f"  {'bias':>5} {'spread(mean_dist)':>18} {'n':>4} {'±95%CI':>8}   {'vacc':>6}")
        for b, m, n, se in curve(sub, 'mean_dist'):
            va = sub.filter(pl.col('bias') == b)['vacc'].drop_nans().mean()
            va = float(va) if va is not None else float('nan')
            print(f"  {b:>5} {m:>18.4f} {n:>4} {se:>8.4f}   {va:>6.3f}")
        d09 = cohens_d(sub)
        # trend: mean_dist at min-nonzero-bias vs max bias
        lo = min([b for b in BIAS if b > 0])
        print(f"  Cohen's d (bias {lo}->0.9) on mean_dist = {d09:+.3f}")

# Also show what happens if you IGNORE protocol (pool both) — a likely peer mistake
print("\n\n=== POOLED over BOTH protocols (likely mis-analysis) ===")
for atk in ['deepfool', 'hsj']:
    sub = cg.filter(pl.col('attack') == atk)
    print(f"\n--- attack={atk} pooled protocols (rows={sub.height}) ---")
    for b, m, n, se in curve(sub, 'mean_dist'):
        print(f"  bias={b}: spread={m:.4f} (n={n})")
    print(f"  Cohen's d 0.1->0.9 = {cohens_d(sub):+.3f}")

# And the buggy/other metric columns for comparison
print("\n\n=== before_split, deepfool: alternative metric columns ===")
sub = cg.filter((pl.col('protocol') == 'before_split') & (pl.col('attack') == 'deepfool'))
for col in ['mean_dist', 'density', 'aiden_density', 'spread_std_space']:
    vals = [f"{m:.3f}" for _, m, _, _ in curve(sub, col)]
    print(f"  {col:>18}: {vals}  (d={cohens_d(sub,col):+.2f})")
