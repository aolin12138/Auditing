"""Plateau check: does spread keep changing as k increases past the onset (k>=6)?
Focused run: tree_d3 + tree_d10, toward, n_out=10, k in {6,8,10,12,14,16,20,24}, 50 seeds.
50 seeds -> CI half-width ~0.03x, enough to distinguish plateau vs drift vs new onset.
-> results_kplateau.parquet, plots/kplateau.png
"""
import os
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_v] = '1'
import sys, warnings, time
from pathlib import Path
import numpy as np, polars as pl
warnings.filterwarnings('ignore')
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import DecisionTreeAttack
import importlib.util

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results_kplateau.parquet'
_spec = importlib.util.spec_from_file_location('ol', HERE / 'run_experiment.py')
_ol = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_ol)
inject_outliers, cluster_stats = _ol.inject_outliers, _ol.cluster_stats

TC, N_OUT, DIR = 2, 10, 'toward'
K_LEVELS = [6, 8, 10, 12, 14, 16, 20, 24]
SEEDS = list(range(0, 50))
MODELS = {'tree_d3': 3, 'tree_d10': 10}
COLS = ['model', 'k', 'seed', 'tacc', 'vacc', 'nadv', 'mean_dist']


def run_cell(model, k, seed):
    X, y = load_iris(return_X_y=True)
    yoh = OneHotEncoder(sparse_output=False).fit(y.reshape(-1, 1))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed)
    folds = []
    for tr, te in skf.split(X, y):
        Xt, Xv, yt, yv = X[tr], X[te], y[tr], y[te]
        Xa, ya = inject_outliers(Xt, yt, TC, k, N_OUT, direction=DIR, rng=rng)
        m = DecisionTreeClassifier(max_depth=MODELS[model], random_state=42).fit(Xa, ya)
        art = SklearnClassifier(m); art.fit(Xa, yoh.transform(ya.reshape(-1, 1)))
        pv = np.argmax(art.predict(Xv), axis=1); c = pv == yv
        if c.sum():
            adv = DecisionTreeAttack(classifier=art).generate(Xv[c])
            ap = np.argmax(art.predict(adv), axis=1)
            adv = adv[ap != yv[c]]
        else:
            adv = np.empty((0, X.shape[1]))
        md = cluster_stats(adv)[2] if len(adv) else np.nan
        folds.append(dict(tacc=m.score(Xa, ya), vacc=float(c.mean()), nadv=len(adv), mean_dist=md))
    r = {kk: float(np.nanmean([f[kk] for f in folds])) for kk in folds[0]}
    r.update(model=model, k=float(k), seed=int(seed))
    return r


def main():
    rows = []; t0 = time.time()
    todo = [(m, k, s) for m in MODELS for k in K_LEVELS for s in SEEDS]
    for i, (m, k, s) in enumerate(todo, 1):
        rows.append(run_cell(m, k, s))
        if i % 200 == 0 or i == len(todo):
            print(f'  [{i}/{len(todo)}] {(time.time()-t0)/60:.1f}m', flush=True)
    pl.DataFrame([{c: r.get(c, np.nan) for c in COLS} for r in rows]).write_parquet(OUT)
    print(f'DONE {len(rows)} rows in {(time.time()-t0)/60:.1f}m', flush=True)


if __name__ == '__main__':
    main()
