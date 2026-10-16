"""Confirm the density<->spread inversion is the cause of the peer's 'decrease'."""
import numpy as np, polars as pl
from pathlib import Path
HERE = Path(__file__).resolve().parent
d = pl.read_csv(HERE / 'export' / 'mlp_results_all.csv')
cg = d.filter((pl.col('defect') == 'coverage_gap') & (pl.col('protocol') == 'before_split'))

for atk in ['deepfool', 'hsj']:
    s = cg.filter(pl.col('attack') == atk)
    md = s['mean_dist'].to_numpy(); de = s['density'].to_numpy()
    ad = s['aiden_density'].to_numpy(); cs = s['clust_size'].to_numpy()
    print(f"\n=== {atk} (before_split) ===")
    print(f"  corr(mean_dist, density)       = {np.corrcoef(md, de)[0,1]:+.3f}")
    print(f"  corr(mean_dist, aiden_density) = {np.corrcoef(md, ad)[0,1]:+.3f}")
    print(f"  clust_size across bias (mean per bias):")
    for b in [0.1,0.3,0.5,0.7,0.9]:
        row = s.filter(pl.col('bias')==b)
        print(f"    bias={b}: clust_size={row['clust_size'].mean():.2f}  "
              f"mean_dist={row['mean_dist'].mean():.3f}  density={row['density'].mean():.3f}")
    # verify density ~= n/(mean_dist+1): reconstruct implied n
    print(f"  density*(mean_dist+1) [~ n_pairs proxy] mean = {np.mean(de*(md+1)):.2f} "
          f"(std {np.std(de*(md+1)):.2f}) -> ~constant means density is driven by 1/mean_dist")
