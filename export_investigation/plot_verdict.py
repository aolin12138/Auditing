"""Verification figure: repo-method spread (mean_dist) RISES with coverage-gap bias
on the MLP, while the confounded density columns FALL (the peer's likely 'decrease')."""
import numpy as np, polars as pl
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
d = pl.read_csv(HERE / 'export' / 'mlp_results_all.csv')
cg = d.filter((pl.col('defect') == 'coverage_gap') & (pl.col('protocol') == 'before_split'))
BIAS = [0.1, 0.3, 0.5, 0.7, 0.9]

def curve(atk, col):
    ms, cis = [], []
    for b in BIAS:
        v = cg.filter((pl.col('attack')==atk)&(pl.col('bias')==b))[col].drop_nans().drop_nulls()
        ms.append(float(v.mean())); cis.append(1.96*float(v.std())/np.sqrt(len(v)))
    return np.array(ms), np.array(cis)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, atk in zip(axes, ['deepfool', 'hsj']):
    y, ci = curve(atk, 'mean_dist')
    ax.plot(BIAS, y, 'o-', lw=2.4, color='#1e7d34', label='mean_dist = clean SPREAD (repo metric) ▲')
    ax.fill_between(BIAS, y-ci, y+ci, color='#1e7d34', alpha=0.15)
    ax.set_xlabel('coverage-gap bias (fraction of class deleted)')
    ax.set_ylabel('spread (mean pairwise dist)', color='#1e7d34')
    ax.set_title(f'MLP + {atk}  (before_split = repo protocol)', fontsize=10)
    ax2 = ax.twinx()
    yd, cid = curve(atk, 'density')
    ya, cia = curve(atk, 'aiden_density')
    ax2.plot(BIAS, yd, 's--', color='#c0392b', alpha=.85, label="'density'  (confounded) ▼")
    ax2.plot(BIAS, ya, '^:', color='#e67e22', alpha=.85, label="'aiden_density' (buggy) ▼")
    ax2.set_ylabel('density-form metric', color='#c0392b')
    l1,la1 = ax.get_legend_handles_labels(); l2,la2 = ax2.get_legend_handles_labels()
    ax.legend(l1+l2, la1+la2, loc='lower center', fontsize=8)
fig.suptitle("Peer's 'spread decreased' = reading a density-form column. The trusted metric "
             "(mean_dist) RISES with bias; density-form metrics FALL (documented inversion).",
             fontsize=10.5)
fig.tight_layout(rect=[0,0,1,0.96])
out = HERE / 'verdict_cg_spread_vs_density.png'
fig.savefig(out, dpi=140)
print('wrote', out)
