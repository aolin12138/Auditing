"""PLAN §9 frontier — does FITTING a reduced model reveal a HARMFUL defect in genuinely-high-D data?

Direct test of the user's question: when the features REALLY matter (genuinely high-D, irreducible),
does reducing dimensionality (fitting the model ON the top-k features, at the cost of accuracy) reveal
the coverage-gap geometry signal? Finding 10 only tested MEASUREMENT projection (fit full model, project
metric) on the harmful gap; this tests FIT-IN-SUBSPACE (the wine-H4 recovery method) on it.

Workflow (realistic audit): the data already contains a HARMFUL boundary-directed coverage gap injected
in the FULL 15-D space (delete the contested class's boundary-facing points). We then select the top-k
F-score features, FIT an overfit tree on those k, attack, and measure adversarial spread in the k-space.
Sweep k. Question: is there any k where spatial-vs-random spread separates (CI) — and at what accuracy cost?

Dataset = the genuinely-high-D one from run_highdim_harmful (15 informative dims, n_clusters_per_class=3:
accuracy climbs with dimension, top-4 ~81% of full => NOT reducible without accuracy loss).

-> results_highdim_reduce.parquet
Usage: python run_highdim_reduce.py
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
OUT = HERE / 'results_highdim_reduce.parquet'
_os = importlib.util.spec_from_file_location('ol', HERE.parent / 'outlier_experiment' / 'run_experiment.py')
_o = importlib.util.module_from_spec(_os); _os.loader.exec_module(_o)
cluster_stats = _o.cluster_stats

D = 15
KS = [2, 3, 4, 6, 10, 15]
FRAC = [0.0, 0.5, 0.7]
STRUCT = ['random', 'spatial']
TC = 0
N_FOLDS = 5
SEEDS = [42, 58, 125, 7, 13, 21, 99, 123, 200, 777] + list(range(300, 320))


def make_data():
    X, y = make_classification(n_samples=600, n_features=D, n_informative=D, n_redundant=0,
                               n_classes=3, n_clusters_per_class=3, class_sep=1.0, flip_y=0.0,
                               random_state=0)
    return StandardScaler().fit_transform(X), y


def defect_dir(Xt, yt):
    m0 = Xt[yt == TC].mean(0)
    others = [c for c in np.unique(yt) if c != TC]
    nearest = min(others, key=lambda c: np.linalg.norm(Xt[yt == c].mean(0) - m0))
    v = Xt[yt == nearest].mean(0) - m0
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def inject(Xt, yt, frac, structure, rng):
    """Inject the harmful boundary-directed coverage gap in FULL space (train fold only)."""
    if frac <= 0:
        return Xt, yt
    v = defect_dir(Xt, yt)
    idx = np.where(yt == TC)[0]
    nrem = min(max(int(len(idx) * frac), 0), len(idx) - 3)
    if nrem <= 0:
        return Xt, yt
    if structure == 'spatial':
        proj = Xt[idx] @ v
        rem = idx[np.argsort(proj)[::-1][:nrem]]
    else:
        rem = rng.choice(idx, size=nrem, replace=False)
    keep = np.ones(len(yt), dtype=bool); keep[rem] = False
    return Xt[keep], yt[keep]


def run_cell(k, structure, frac, seed, X, y, rank):
    cols = rank[:k]
    yoh = OneHotEncoder(sparse_output=False).fit(y.reshape(-1, 1))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed)
    folds = []
    for tr, te in skf.split(X, y):
        Xt, Xv, yt, yv = X[tr], X[te], y[tr], y[te]
        Xt, yt = inject(Xt, yt, frac, structure, rng)      # gap injected in FULL space, train-only
        # now REDUCE: fit + attack + measure entirely in the top-k subspace
        Xtk, Xvk = Xt[:, cols], Xv[:, cols]
        clf = DecisionTreeClassifier(max_depth=None, random_state=42).fit(Xtk, yt)
        art = SklearnClassifier(clf); art.fit(Xtk, yoh.transform(yt.reshape(-1, 1)))
        pv = np.argmax(art.predict(Xvk), axis=1)
        vacc = float((pv == yv).mean())
        mmask = yv == TC
        recall = float((pv[mmask] == TC).mean()) if mmask.sum() else np.nan
        c = pv == yv
        if c.sum():
            adv = DecisionTreeAttack(classifier=art).generate(Xvk[c])
            ap = np.argmax(art.predict(adv), axis=1); adv = adv[ap != yv[c]]
        else:
            adv = np.empty((0, k))
        md = cluster_stats(adv)[2] if len(adv) else np.nan
        folds.append(dict(vacc=vacc, recall=recall, nadv=len(adv), mean_dist=md))
    r = {kk: float(np.nanmean([f[kk] for f in folds])) for kk in folds[0]}
    r.update(k=int(k), structure=structure, frac=float(frac), seed=int(seed))
    return r


def cohend(a, b):
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, False
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    d = (a.mean() - b.mean()) / sp if sp > 0 else np.nan
    sa = 1.96 * a.std(ddof=1) / np.sqrt(len(a)); sb = 1.96 * b.std(ddof=1) / np.sqrt(len(b))
    return d, (a.mean() - sa > b.mean() + sb) or (b.mean() - sb > a.mean() + sa)


def main():
    X, y = make_data()
    F, _ = f_classif(X, y)
    rank = np.argsort(np.nan_to_num(F, nan=-np.inf))[::-1]
    cols = ['k', 'structure', 'frac', 'seed', 'vacc', 'recall', 'nadv', 'mean_dist']
    rows, t0 = [], time.time()
    grid = []
    for k, seed in product(KS, SEEDS):
        grid.append((k, 'random', 0.0, seed))
        for st, fr in product(STRUCT, FRAC[1:]):
            grid.append((k, st, fr, seed))
    for k, st, fr, sd in grid:
        rows.append(run_cell(k, st, fr, sd, X, y, rank))
    df = pl.DataFrame([{c: r.get(c, np.nan) for c in cols} for r in rows])
    df.write_parquet(OUT)

    print(f"\n{'='*92}\nFIT-IN-SUBSPACE on genuinely-high-D + HARMFUL gap: does reducing reveal the signal?\n{'='*92}")
    print(f"{'k':>3} | {'clean vacc':>10} | {'vacc drop@0.5':>12} | {'spread d@0.5 (sep)':>18} | {'spread d@0.7 (sep)':>18} | {'recallΔ@0.5':>11}")
    for k in KS:
        def col(st, fr, c):
            return df.filter((pl.col('k') == k) & (pl.col('structure') == st) & (pl.col('frac') == fr))[c].to_numpy()
        v0 = col('random', 0.0, 'vacc').mean()
        vdrop = v0 - col('spatial', 0.5, 'vacc').mean()
        out = []
        for fr in [0.5, 0.7]:
            base = col('random', 0.0, 'mean_dist').mean()
            d, sep = cohend(col('spatial', fr, 'mean_dist') / base, col('random', fr, 'mean_dist') / base)
            out.append(f"{d:>+8.2f} ({'Y' if sep else 'n'})")
        recd = np.nanmean(col('spatial', 0.5, 'recall')) - np.nanmean(col('random', 0.5, 'recall'))
        print(f"{k:>3} | {v0:>10.3f} | {vdrop:>+12.3f} | {out[0]:>18} | {out[1]:>18} | {recd:>+11.3f}")
    print(f"\n(full-D k=15 accuracy is the reference; lower k trades accuracy for a simpler boundary)")
    print(f"DONE: {len(rows)} rows -> {OUT.name} in {(time.time()-t0)/60:.1f}m")


if __name__ == '__main__':
    main()
