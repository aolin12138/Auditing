"""Geometry check for the distance dip at k~4: where along the 'toward' ray does the outlier
enter / sit-in / exit the NEAREST OTHER CLASS cluster? Compare to the spread-vs-k dip.
tc=2 (virginica), toward = nearest other class centroid. Averaged over the 5 folds x 10 seeds
used in the experiment. -> prints table + writes plots/distance_geometry.png
"""
import os
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_v] = '1'
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, polars as pl
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
TC = 2
SEEDS = [42, 58, 125, 7, 13, 21, 99, 123, 200, 777]
KS = [2, 3, 4, 5, 6, 8]
X, y = load_iris(return_X_y=True)


def nearest_other(mu_c, means):
    others = {c: m for c, m in means.items() if c != TC}
    return min(others, key=lambda c: np.linalg.norm(others[c] - mu_c))


# collect geometry per fold (mirrors inject_outliers exactly)
rows = []
for seed in SEEDS:
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        Xt, yt = X[tr], y[tr]
        Xc = Xt[yt == TC]
        mu_c = Xc.mean(0); s = float(Xc.std(0).mean())
        means = {c: Xt[yt == c].mean(0) for c in np.unique(yt)}
        noc = nearest_other(mu_c, means)
        mu_o = means[noc]
        Xo_cls = Xt[yt == noc]                             # the other-class point cloud
        r_noc = float(np.linalg.norm(Xo_cls - mu_o, axis=1).mean())   # its mean radius
        D = float(np.linalg.norm(mu_o - mu_c))            # centroid separation
        dvec = (mu_o - mu_c) / (np.linalg.norm(mu_o - mu_c) + 1e-12)
        for k in KS:
            loc = mu_c + k * s * dvec                     # outlier mean location (no noise)
            d_to_noc_centroid = np.linalg.norm(loc - mu_o)
            d_to_noc_nearest = float(np.linalg.norm(Xo_cls - loc, axis=1).min())
            inside = d_to_noc_centroid < r_noc            # within other cluster's mean radius?
            rows.append(dict(seed=seed, k=k, s=s, D=D, r_noc=r_noc, noc=int(noc),
                             k_at_noc_centroid=D / s, k_enter=(D - r_noc) / s, k_exit=(D + r_noc) / s,
                             d_to_noc_centroid_in_s=d_to_noc_centroid / s,
                             d_to_noc_nearest_in_s=d_to_noc_nearest / s, inside=float(inside)))
d = pl.DataFrame(rows)
noc_id = int(d['noc'].mode()[0])
names = ['setosa', 'versicolor', 'virginica']
print(f"tc={TC} ({names[TC]}); nearest other class = {noc_id} ({names[noc_id]})")
print(f"centroid separation D/s = {d['k_at_noc_centroid'].mean():.2f}  (outlier sits ON the "
      f"other centroid at this k)")
print(f"other-cluster mean radius r_noc/s = {(d['r_noc']/d['s']).mean():.2f}")
print(f"  -> outlier ENTERS other cluster at k~{d['k_enter'].mean():.2f}, "
      f"EXITS at k~{d['k_exit'].mean():.2f}\n")
print("k  | dist(outlier->other centroid)/s | dist->nearest other pt /s | inside other cluster?")
for k in KS:
    c = d.filter(pl.col('k') == k)
    print(f"{k:>2} |  {c['d_to_noc_centroid_in_s'].mean():>6.2f}  "
          f"|  {c['d_to_noc_nearest_in_s'].mean():>6.2f}  |  {c['inside'].mean()*100:>4.0f}% of folds")

# overlay plot: spread-vs-k (from data) + distance-to-other-cluster-vs-k
ext = pl.read_parquet(HERE / 'results_extended.parquet')
base = ext.filter((pl.col('model') == 'tree_d10') & (pl.col('n_out') == 0))['mean_dist'].drop_nans().mean()
sp = [ext.filter((pl.col('model') == 'tree_d10') & (pl.col('direction') == 'toward')
                 & (pl.col('n_out') == 10) & (pl.col('k') == float(k)))['mean_dist'].drop_nans().mean() / base
      for k in KS]
dist_c = [d.filter(pl.col('k') == k)['d_to_noc_centroid_in_s'].mean() for k in KS]
k_center = d['k_at_noc_centroid'].mean()

fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(KS, sp, 'o-', lw=2.4, color='#1e5eff', label='tree d10 spread (toward, n=10)')
ax.axhline(1.0, ls=':', color='0.5')
ax.set_xlabel('k = distance to own centroid (x class std)')
ax.set_ylabel('normalised spread', color='#1e5eff'); ax.tick_params(axis='y', labelcolor='#1e5eff')
ax2 = ax.twinx()
ax2.plot(KS, dist_c, 's--', lw=1.8, color='#c0392b', label='dist(outlier -> other-class centroid) / s')
ax2.axhline(0, ls='-', color='0.7', lw=0.8)
ax2.set_ylabel('distance to OTHER-class centroid (x std)', color='#c0392b'); ax2.tick_params(axis='y', labelcolor='#c0392b')
ax.axvline(k_center, ls='-', color='#c0392b', alpha=0.25, lw=8)
ax.text(k_center, ax.get_ylim()[1], f' outlier ON\n other centroid\n (k={k_center:.1f})',
        va='top', ha='center', fontsize=8, color='#7a1f14')
l1, la1 = ax.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
ax.legend(l1 + l2, la1 + la2, fontsize=8, loc='upper center')
ax.set_title('Distance dip vs geometry: spread is LOW exactly where the outlier sits\n'
             'inside the other-class cluster, and RISES once it exits into empty space', fontsize=9.5)
fig.tight_layout(); fig.savefig(HERE / 'plots' / 'distance_geometry.png', dpi=140)
print('\nwrote', HERE / 'plots' / 'distance_geometry.png')
