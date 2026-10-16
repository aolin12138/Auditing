"""Shortcut: train accuracy (blind) vs test accuracy (drops) vs the directional metric (tracks reliance).

Makes the point that training accuracy is 1.000 throughout (the overfit tree memorizes regardless of the
shortcut), so it CANNOT see the defect; test accuracy drops; and the adversarial `spur_frac` rises in step.
-> plots/shortcut/train_vs_test_acc.{png,pdf}
"""
import os
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_v] = '1'
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, polars as pl
import matplotlib.pyplot as plt
import plotstyle as ps

HERE = Path(__file__).resolve().parent
PLOTS = HERE / 'plots' / 'shortcut'; PLOTS.mkdir(parents=True, exist_ok=True)
CORR = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
ps.apply()
df = pl.read_parquet(HERE / 'results_shortcut.parquet')


def series(ds, col):
    m, e = [], []
    for c in CORR:
        v = df.filter((pl.col('dataset') == ds) & (pl.col('corr') == c))[col].to_numpy()
        v = v[~np.isnan(v)]
        m.append(v.mean()); e.append(1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
    return np.array(m), np.array(e)


fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
for j, (ds, title) in enumerate([('iris', 'iris (4-D)'), ('wine', 'wine (13-D)')]):
    for col, lab, color, ls, mk in [
        ('tacc', 'train accuracy (blind = 1.0)', ps.OKABE_ITO['black'], ':', 'x'),
        ('vacc', 'test accuracy (Clever Hans cost)', ps.OKABE_ITO['blue'], '--', 's'),
        ('spur_frac', 'spurious-axis displacement frac (our metric)', ps.OKABE_ITO['vermillion'], '-', 'o')]:
        m, e = series(ds, col)
        ax[j].errorbar(CORR, m, yerr=e, color=color, ls=ls, marker=mk, capsize=3, label=lab)
    ax[j].set_xlabel('shortcut strength  corr'); ax[j].set_title(title)
    ax[j].set_ylim(0, 1.05); ps.panel_label(ax[j], 'ab'[j])
ax[0].set_ylabel('accuracy  /  metric value')
ax[0].legend(frameon=False, fontsize=8, loc='center left')
fig.suptitle('Shortcut defect: training accuracy is blind (1.0), test accuracy drops, the directional metric tracks it', y=1.03)
fig.tight_layout()
outs = ps.save(fig, PLOTS / 'train_vs_test_acc')
plt.close(fig)
print('wrote', *[o.name for o in outs])
