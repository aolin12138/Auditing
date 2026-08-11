"""Outlier THRESHOLD sweep (meeting asks): count as a PERCENTAGE of class size (up to 200%),
three directions incl. RANDOM (each outlier a random direction from the centroid, not aimed),
and record BOTH train and test accuracy to see if/when accuracy finally breaks.
Model: overfit tree (max_depth=None) + white-box DTA (fast, no hang). k fixed = 8 (x class std).
Usage: python run_threshold.py   ->  results_threshold.parquet
"""
import os
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_v] = '1'
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
from itertools import product
import numpy as np, polars as pl
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from art.estimators.classification import SklearnClassifier
from run_experiment import inject_outliers, cluster_stats, attack_adv, N_FOLDS

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results_threshold.parquet'
PCTS = [5, 10, 25, 50, 75, 100, 150, 200]          # n_out as % of class size
DIRECTIONS = ['toward', 'outward', 'random']
SEEDS = [42, 58, 125, 7, 13, 21, 99, 123, 200, 777]
TC, K = 2, 8
COLS = ['pct', 'direction', 'k', 'seed', 'tc', 'tacc', 'vacc', 'asucc', 'nadv',
        'mean_dist', 'spread_c2']


def run_cell(X, y, pct, direction, seed):
    yoh = OneHotEncoder(sparse_output=False).fit(y.reshape(-1, 1))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed)
    folds = []
    for tr, te in skf.split(X, y):
        Xt, Xv, yt, yv = X[tr], X[te], y[tr], y[te]
        if pct > 0:
            Xa, ya = inject_outliers(Xt, yt, TC, K, 0, direction=direction,
                                     rng=rng, n_out_frac=pct / 100.0)
        else:
            Xa, ya = Xt, yt
        m = DecisionTreeClassifier(max_depth=None, random_state=42).fit(Xa, ya)
        art = SklearnClassifier(m); art.fit(Xa, yoh.transform(ya.reshape(-1, 1)))
        tacc = float((np.argmax(art.predict(Xa), axis=1) == ya).mean())
        vacc = float((np.argmax(art.predict(Xv), axis=1) == yv).mean())
        adv, succ, labels = attack_adv(art, Xv, yv)
        md = cluster_stats(adv)[2]
        sc = cluster_stats(adv[labels == TC])[2] if len(adv) else np.nan
        folds.append(dict(tacc=tacc, vacc=vacc, asucc=succ, nadv=len(adv), mean_dist=md, spread_c2=sc))
    r = {kk: float(np.nanmean([f[kk] for f in folds])) for kk in folds[0]}
    r.update(pct=pct, direction=direction if pct > 0 else 'none', k=K, seed=seed, tc=TC)
    return r


def main():
    X, y = load_iris(return_X_y=True)
    rows = []
    for seed in SEEDS:
        rows.append(run_cell(X, y, 0, 'none', seed))
        for d, pct in product(DIRECTIONS, PCTS):
            rows.append(run_cell(X, y, pct, d, seed))
    pl.DataFrame([{c: r[c] for c in COLS} for r in rows]).write_parquet(OUT)
    df = pl.read_parquet(OUT)
    base = df.filter(pl.col('pct') == 0)
    print(f'wrote {df.height} rows. baseline: spread={base["mean_dist"].mean():.3f} '
          f'tacc={base["tacc"].mean():.3f} vacc={base["vacc"].mean():.3f}', flush=True)
    for d in DIRECTIONS:
        print(f'\n{d}:  pct -> spread / tacc / vacc', flush=True)
        for pct in PCTS:
            c = df.filter((pl.col('direction') == d) & (pl.col('pct') == pct))
            print(f'  {pct:>4}%: spread={c["mean_dist"].mean():.3f}  '
                  f'tacc={c["tacc"].mean():.3f}  vacc={c["vacc"].mean():.3f}', flush=True)


if __name__ == '__main__':
    main()
