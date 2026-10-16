"""PLAN §9 follow-up — what if the problem GENUINELY needs high dimensionality?

User's question: wine kept its accuracy when reduced to k=4 because its top features already carry the
signal (redundant / low intrinsic dimension). But if a problem TRULY needs 7/8/13 features, does
reducing still keep accuracy, and what happens to the geometry probe then?

Test: contrast two synthetic datasets of the same size (10 features, 3 classes), differing ONLY in
intrinsic dimensionality:
  redundant        n_informative=3, n_redundant=7  -> only ~3 dims truly matter (like wine). Reducing to
                   a few top features should KEEP accuracy, and a low-D probe should see the defect.
  all_informative  n_informative=10, n_redundant=0 -> every feature carries independent signal. Reducing
                   should HURT accuracy, and there should be NO k that both keeps accuracy AND is low
                   enough to reveal the geometry signal.

For each: inject the coverage-gap (spatial) vs imbalance (random) defect on the contested class, sweep
the number of top-F features kept `k`, fit tree + DTA, 30 seeds. Report accuracy vs k, the spread
signal (Cohen's d spatial vs random) vs k, and the minority-recall gap vs k (the model-agnostic
full-D detector). The question: is there a k with BOTH good accuracy AND a visible geometry signal?

-> results_highdim_need.parquet
Usage: python run_highdim_need.py
"""
import os
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_v] = '1'
import warnings, time, importlib.util
from pathlib import Path
from itertools import product
import numpy as np, polars as pl
warnings.filterwarnings('ignore')
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import f_classif
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import DecisionTreeAttack

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results_highdim_need.parquet'
_os = importlib.util.spec_from_file_location('ol', HERE.parent / 'outlier_experiment' / 'run_experiment.py')
_o = importlib.util.module_from_spec(_os); _os.loader.exec_module(_o)
cluster_stats = _o.cluster_stats

D = 10
KS = [2, 3, 4, 6, 8, 10]
FRAC = [0.0, 0.5, 0.7]
STRUCT = ['random', 'spatial']
TC = 0
N_FOLDS = 5
SEEDS = [42, 58, 125, 7, 13, 21, 99, 123, 200, 777] + list(range(300, 320))
DATASETS = ['redundant', 'all_informative']


def make_data(name):
    """Fixed dataset (seed 0); CV split varies by seed. Same size/shape, differ only in intrinsic dim."""
    common = dict(n_samples=240, n_features=D, n_classes=3, n_clusters_per_class=1,
                  class_sep=1.3, flip_y=0.0, random_state=0)
    if name == 'redundant':
        X, y = make_classification(n_informative=3, n_redundant=7, **common)
    else:
        X, y = make_classification(n_informative=D, n_redundant=0, **common)
    return StandardScaler().fit_transform(X), y


def rankF(X, y):
    F, _ = f_classif(X, y)
    return np.argsort(np.nan_to_num(F, nan=-np.inf))[::-1]


def inject_delete(Xt, yt, tc, feat, frac, structure, rng):
    if frac <= 0:
        return Xt, yt
    idx = np.where(yt == tc)[0]
    nrem = min(max(int(len(idx) * frac), 0), len(idx) - 3)
    if nrem <= 0:
        return Xt, yt
    if structure == 'spatial':
        rem = idx[np.argsort(Xt[idx, feat])][:nrem]
    else:
        rem = rng.choice(idx, size=nrem, replace=False)
    keep = np.ones(len(yt), dtype=bool); keep[rem] = False
    return Xt[keep], yt[keep]


def run_cell(name, k, structure, frac, seed, Xfull, y, rank):
    cols = rank[:k]
    Xk = Xfull[:, cols]
    feat = 0                                    # top-F feature is local column 0 after selection
    yoh = OneHotEncoder(sparse_output=False).fit(y.reshape(-1, 1))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed)
    folds = []
    for tr, te in skf.split(Xk, y):
        Xt, Xv, yt, yv = Xk[tr], Xk[te], y[tr], y[te]
        Xa, ya = inject_delete(Xt, yt, TC, feat, frac, structure, rng)
        clf = DecisionTreeClassifier(max_depth=None, random_state=42).fit(Xa, ya)
        art = SklearnClassifier(clf); art.fit(Xa, yoh.transform(ya.reshape(-1, 1)))
        pv = np.argmax(art.predict(Xv), axis=1)
        vacc = float((pv == yv).mean())
        mmask = yv == TC
        min_recall = float((pv[mmask] == TC).mean()) if mmask.sum() else np.nan
        c = pv == yv
        if c.sum():
            adv = DecisionTreeAttack(classifier=art).generate(Xv[c])
            ap = np.argmax(art.predict(adv), axis=1); adv = adv[ap != yv[c]]
        else:
            adv = np.empty((0, k))
        md = cluster_stats(adv)[2] if len(adv) else np.nan
        folds.append(dict(vacc=vacc, min_recall=min_recall, nadv=len(adv), mean_dist=md))
    r = {kk: float(np.nanmean([f[kk] for f in folds])) for kk in folds[0]}
    r.update(dataset=name, k=int(k), structure=structure, frac=float(frac), seed=int(seed))
    return r


def cohend(a, b):
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, False
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    d = (a.mean() - b.mean()) / sp if sp > 0 else np.nan
    sa = 1.96 * a.std(ddof=1) / np.sqrt(len(a)); sb = 1.96 * b.std(ddof=1) / np.sqrt(len(b))
    return d, (a.mean() - sa > b.mean() + sb)


def main():
    cols = ['dataset', 'k', 'structure', 'frac', 'seed', 'vacc', 'min_recall', 'nadv', 'mean_dist']
    data = {n: make_data(n) for n in DATASETS}
    ranks = {n: rankF(*data[n]) for n in DATASETS}
    rows, t0 = [], time.time()
    grid = []
    for name, k, seed in product(DATASETS, KS, SEEDS):
        grid.append((name, k, 'random', 0.0, seed))
        for st, fr in product(STRUCT, FRAC[1:]):
            grid.append((name, k, st, fr, seed))
    for name, k, st, fr, sd in grid:
        X, y = data[name]
        rows.append(run_cell(name, k, st, fr, sd, X, y, ranks[name]))
    df = pl.DataFrame([{c: r.get(c, np.nan) for c in cols} for r in rows])
    df.write_parquet(OUT)

    print(f"\n{'='*100}\nDoes reducing keep accuracy when the problem truly needs high-D? (10 features, contested class)\n{'='*100}")
    for name in DATASETS:
        print(f"\n--- {name} ---  (n_informative = {3 if name=='redundant' else D})")
        print(f"{'k':>3} | {'clean vacc':>10} | {'vacc@spatial0.5':>15} | {'spread d@0.5 (sep)':>18} | {'recallΔ@0.5':>11}")
        for k in KS:
            def col(st, fr, c):
                return df.filter((pl.col('dataset') == name) & (pl.col('k') == k) & (pl.col('structure') == st) & (pl.col('frac') == fr))[c].to_numpy()
            v0 = col('random', 0.0, 'vacc').mean()
            vsp = col('spatial', 0.5, 'vacc').mean()
            base = col('random', 0.0, 'mean_dist').mean()
            d, sep = cohend(col('spatial', 0.5, 'mean_dist') / base, col('random', 0.5, 'mean_dist') / base)
            recd = np.nanmean(col('spatial', 0.5, 'min_recall')) - np.nanmean(col('random', 0.5, 'min_recall'))
            print(f"{k:>3} | {v0:>10.3f} | {vsp:>15.3f} | {d:>+13.2f} ({'Y' if sep else 'n'}) | {recd:>+11.2f}")
        # is there a k with BOTH good accuracy (>= 95% of full-k acc) AND a separated signal?
        full = df.filter((pl.col('dataset') == name) & (pl.col('k') == D) & (pl.col('structure') == 'random') & (pl.col('frac') == 0.0))['vacc'].mean()
        sweet = []
        for k in KS:
            v0 = df.filter((pl.col('dataset') == name) & (pl.col('k') == k) & (pl.col('structure') == 'random') & (pl.col('frac') == 0.0))['vacc'].mean()
            base = df.filter((pl.col('dataset') == name) & (pl.col('k') == k) & (pl.col('structure') == 'random') & (pl.col('frac') == 0.0))['mean_dist'].to_numpy().mean()
            sp = df.filter((pl.col('dataset') == name) & (pl.col('k') == k) & (pl.col('structure') == 'spatial') & (pl.col('frac') == 0.5))['mean_dist'].to_numpy() / base
            rr = df.filter((pl.col('dataset') == name) & (pl.col('k') == k) & (pl.col('structure') == 'random') & (pl.col('frac') == 0.5))['mean_dist'].to_numpy() / base
            _, sep = cohend(sp, rr)
            if sep and v0 >= 0.95 * full:
                sweet.append(k)
        print(f"  full-D ({D}) clean vacc = {full:.3f}; k with BOTH >=95% acc AND separated geometry signal: "
              f"{sweet if sweet else 'NONE'}")
    print(f"\nDONE: {len(rows)} rows -> {OUT.name} in {(time.time()-t0)/60:.1f}m")


if __name__ == '__main__':
    main()
