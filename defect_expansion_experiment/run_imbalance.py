"""Defect-expansion Phase 0 — CLASS IMBALANCE vs COVERAGE-GAP control (overfit tree + white-box DTA).

Tests H1 (spatial-hole): at MATCHED count removal from class `tc`, does SPATIAL deletion
(coverage gap) produce more adversarial spread than RANDOM deletion (imbalance)? If yes, the
coverage-gap signal is the spatial hole, not merely fewer samples.

Both arms delete from the TRAIN FOLD only (clean test) so the contrast is pure and free of the
coverage-gap test-set confound. Reuses cluster_stats from ../outlier_experiment.
-> results_imbalance.parquet   (one row per frac x structure x seed, 5-fold mean)

Usage: python run_imbalance.py
"""
import os
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_v] = '1'
import warnings, time
from pathlib import Path
from itertools import product
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
OUT = HERE / 'results_imbalance.parquet'
_spec = importlib.util.spec_from_file_location(
    'outlier_lib', HERE.parent / 'outlier_experiment' / 'run_experiment.py')
_ol = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_ol)
cluster_stats = _ol.cluster_stats

FRAC = [0.0, 0.25, 0.5, 0.7, 0.85, 0.95]       # fraction of class tc removed (severity)
STRUCT = ['random', 'spatial']                  # imbalance vs coverage-gap; the key factor
TC = 2                                          # virginica (contested class)
FEAT = 2                                        # petal length (spatial arm sorts by this)
N_FOLDS = 5
SEEDS = [42, 58, 125, 7, 13, 21, 99, 123, 200, 777] + list(range(300, 320))   # 30 seeds (firm up H1)
COLS = ['frac', 'structure', 'tc', 'feat', 'seed', 'tacc', 'vacc', 'min_recall',
        'nadv', 'mean_dist']


def inject_delete(Xt, yt, tc, feat, frac, structure, rng):
    """Remove `frac` of class tc from the train fold. spatial = lowest along feat (mirrors
    model_family.inject_cg); random = uniform (class imbalance). Same count for both arms."""
    if frac <= 0:
        return Xt, yt
    idx = np.where(yt == tc)[0]
    nrem = int(len(idx) * frac)
    if nrem <= 0 or nrem >= len(idx):
        nrem = min(max(nrem, 0), len(idx) - 3)      # keep >=3 of the class
    if nrem <= 0:
        return Xt, yt
    if structure == 'spatial':
        rem = idx[np.argsort(Xt[idx, feat])][:nrem]  # lowest petal-length -> a spatial hole
    else:
        rem = rng.choice(idx, size=nrem, replace=False)
    keep = np.ones(len(yt), dtype=bool); keep[rem] = False
    return Xt[keep], yt[keep]


def run_cell(frac, structure, seed):
    X, y = load_iris(return_X_y=True)
    yoh = OneHotEncoder(sparse_output=False).fit(y.reshape(-1, 1))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed)
    folds = []
    for tr, te in skf.split(X, y):
        Xt, Xv, yt, yv = X[tr], X[te], y[tr], y[te]
        Xa, ya = inject_delete(Xt, yt, TC, FEAT, frac, structure, rng)
        m = DecisionTreeClassifier(max_depth=None, random_state=42).fit(Xa, ya)
        art = SklearnClassifier(m); art.fit(Xa, yoh.transform(ya.reshape(-1, 1)))
        tacc = float((np.argmax(art.predict(Xa), axis=1) == ya).mean())
        pv = np.argmax(art.predict(Xv), axis=1)
        vacc = float((pv == yv).mean())
        mmask = yv == TC
        min_recall = float((pv[mmask] == TC).mean()) if mmask.sum() else np.nan
        c = pv == yv
        if c.sum():
            adv = DecisionTreeAttack(classifier=art).generate(Xv[c])
            ap = np.argmax(art.predict(adv), axis=1)
            adv = adv[ap != yv[c]]
        else:
            adv = np.empty((0, X.shape[1]))
        md = cluster_stats(adv)[2] if len(adv) else np.nan
        folds.append(dict(tacc=tacc, vacc=vacc, min_recall=min_recall, nadv=len(adv), mean_dist=md))
    r = {k: float(np.nanmean([f[k] for f in folds])) for k in folds[0]}
    r.update(frac=float(frac), structure=structure, tc=TC, feat=FEAT, seed=int(seed))
    return r


def main():
    rows, t0 = [], time.time()
    grid = [(f, s, sd) for f, s, sd in product(FRAC, STRUCT, SEEDS)
            if not (f == 0.0 and s == 'spatial')]        # frac=0 identical -> keep only 'random'
    for i, (f, s, sd) in enumerate(grid, 1):
        rows.append(run_cell(f, s, sd))
        if i % 20 == 0 or i == len(grid):
            print(f'  [{i}/{len(grid)}] {(time.time()-t0)/60:.1f}m', flush=True)
    pl.DataFrame([{c: r.get(c, np.nan) for c in COLS} for r in rows]).write_parquet(OUT)
    print(f'DONE: {len(rows)} rows -> {OUT.name} in {(time.time()-t0)/60:.1f}m', flush=True)


if __name__ == '__main__':
    main()
