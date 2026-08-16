"""Phase 1 addendum — QUANTIFY the coverage-gap ACCURACY CONFOUND (before-split vs train-only).

All flagship coverage-gap runs inject the spatial deletion BEFORE the CV split, so the deleted
petal-length band is absent from the TEST set too -> "accuracy stays flat" is partly an artifact
(there are no test points in the hole to misclassify; for the contested class it can even RISE).
The correct protocol deletes from the TRAIN FOLD only, keeping a clean test -> accuracy drops.

This isolates the protocol effect: same model (overfit tree), same attack (white-box DTA), same
class, same spatial deletion — only WHERE the deletion happens (before vs after the split) differs.
Deterministic, no hangs -> simple inline loop.  Reuses inject_delete from ./run_imbalance.py.
-> results_confound.parquet   (one row per protocol x tc x frac x seed, 5-fold mean)

Usage: python run_confound.py
"""
import os
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_v] = '1'
import warnings, time, importlib.util
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

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results_confound.parquet'
_spec = importlib.util.spec_from_file_location('imb_lib', HERE / 'run_imbalance.py')
_imb = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_imb)
inject_delete = _imb.inject_delete
_ol_spec = importlib.util.spec_from_file_location('outlier_lib', HERE.parent / 'outlier_experiment' / 'run_experiment.py')
_ol = importlib.util.module_from_spec(_ol_spec); _ol_spec.loader.exec_module(_ol)
cluster_stats = _ol.cluster_stats

FRAC = [0.0, 0.25, 0.5, 0.7, 0.85, 0.95]
TC = [0, 2]
FEAT = 2
N_FOLDS = 5
SEEDS = [42, 58, 125, 7, 13, 21, 99, 123, 200, 777] + list(range(300, 320))   # 30 seeds
COLS = ['protocol', 'tc', 'feat', 'frac', 'seed', 'tacc', 'vacc', 'min_recall', 'nadv', 'mean_dist']


def delete_before_split(X, y, tc, feat, frac):
    """Spatial deletion applied to the FULL dataset (removes the band from train AND test)."""
    if frac <= 0:
        return X, y
    idx = np.where(y == tc)[0]
    nrem = min(int(len(idx) * frac), len(idx) - 3)
    if nrem <= 0:
        return X, y
    rem = idx[np.argsort(X[idx, feat])][:nrem]        # lowest petal-length band (matches inject_delete spatial)
    keep = np.ones(len(y), dtype=bool); keep[rem] = False
    return X[keep], y[keep]


def run_cell(protocol, tc, frac, seed):
    X, y = load_iris(return_X_y=True)
    yoh = OneHotEncoder(sparse_output=False).fit(y.reshape(-1, 1))
    if protocol == 'before_split' and frac > 0:
        Xg, yg = delete_before_split(X, y, tc, FEAT, frac)
    else:
        Xg, yg = X, y
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed)
    folds = []
    for tr, te in skf.split(Xg, yg):
        Xt, Xv, yt, yv = Xg[tr], Xg[te], yg[tr], yg[te]
        if protocol == 'train_only' and frac > 0:
            Xt, yt = inject_delete(Xt, yt, tc, FEAT, frac, 'spatial', rng)
        m = DecisionTreeClassifier(max_depth=None, random_state=42).fit(Xt, yt)
        art = SklearnClassifier(m); art.fit(Xt, yoh.transform(yt.reshape(-1, 1)))
        tacc = float((np.argmax(art.predict(Xt), axis=1) == yt).mean())
        pv = np.argmax(art.predict(Xv), axis=1)
        vacc = float((pv == yv).mean())
        mmask = yv == tc
        min_recall = float((pv[mmask] == tc).mean()) if mmask.sum() else np.nan
        c = pv == yv
        if c.sum():
            adv = DecisionTreeAttack(classifier=art).generate(Xv[c])
            ap = np.argmax(art.predict(adv), axis=1); adv = adv[ap != yv[c]]
        else:
            adv = np.empty((0, X.shape[1]))
        md = cluster_stats(adv)[2] if len(adv) else np.nan
        folds.append(dict(tacc=tacc, vacc=vacc, min_recall=min_recall, nadv=len(adv), mean_dist=md))
    r = {k: float(np.nanmean([f[k] for f in folds])) for k in folds[0]}
    r.update(protocol=protocol, tc=int(tc), feat=FEAT, frac=float(frac), seed=int(seed))
    return r


def main():
    rows, t0 = [], time.time()
    grid = []
    for tc, seed in product(TC, SEEDS):
        grid.append(('train_only', tc, 0.0, seed))                # frac=0 baseline (protocols identical)
        for proto, frac in product(['before_split', 'train_only'], FRAC[1:]):
            grid.append((proto, tc, frac, seed))
    for i, (p, tc, f, sd) in enumerate(grid, 1):
        rows.append(run_cell(p, tc, f, sd))
        if i % 50 == 0 or i == len(grid):
            print(f'  [{i}/{len(grid)}] {(time.time()-t0)/60:.1f}m', flush=True)
    pl.DataFrame([{c: r.get(c, np.nan) for c in COLS} for r in rows]).write_parquet(OUT)
    print(f'DONE: {len(rows)} rows -> {OUT.name} in {(time.time()-t0)/60:.1f}m', flush=True)


if __name__ == '__main__':
    main()
