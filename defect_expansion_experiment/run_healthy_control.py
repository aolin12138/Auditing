"""PLAN §9 negative control — is the low-D spatial-vs-random spread gap the DEFECT, or a low-D artifact?

User's question: when we reduce to low-D and see spatial (coverage-gap) spread > random (imbalance)
spread, is the metric reflecting the defect, or does spread "naturally grow" in low-D regardless? The
clean test: run the IDENTICAL low-D pipeline where there is NO real boundary-distorting defect and check
the pattern is ABSENT. (The random arm already matches deletion COUNT; this adds datasets where the
spatial deletion does not correspond to a harmful coverage gap.)

Four conditions, all at the SAME low dimensionality (3-D), tree+DTA, 30 CV seeds:
  iris_contested   real iris (top-3 F feats), delete class 2 (virginica, ON the boundary) spatially.
                   POSITIVE control — a real coverage gap. Expect spatial >> random.
  iris_separated   SAME data/dim/manipulation, but delete class 0 (setosa, WELL-SEPARATED, wide margin).
                   Deleting a separated class's edge does not move the boundary => expect NULL.
                   The tightest control: only the target class differs, so any spatial>random here is a
                   low-D/metric artifact, not a defect.
  blobs_healthy    well-separated Gaussian blobs (clean structure, big margins, no injected pathology).
                   A coverage gap in wide-margin data should not distort the boundary => expect NULL.
  rand_null        Gaussian features with RANDOM labels (no class structure, no boundary at all).
                   A "coverage gap" cannot reflect a real defect here => expect NULL (strongest null).

Verdict: if iris_contested separates (spatial>random, CI) while the three controls do NOT, the spread
metric reflects the boundary-distorting DEFECT, not low dimensionality per se. If a control ALSO
separates, the low-D gap is (partly) artifactual — a genuine caveat.

-> results_healthy_control.parquet
Usage: python run_healthy_control.py
"""
import os
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_v] = '1'
import warnings, time, importlib.util
from pathlib import Path
from itertools import product
import numpy as np, polars as pl
warnings.filterwarnings('ignore')
from sklearn.datasets import load_iris, make_blobs
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import f_classif
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import DecisionTreeAttack

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results_healthy_control.parquet'
_os = importlib.util.spec_from_file_location('ol', HERE.parent / 'outlier_experiment' / 'run_experiment.py')
_o = importlib.util.module_from_spec(_os); _os.loader.exec_module(_o)
cluster_stats = _o.cluster_stats
_ah = importlib.util.spec_from_file_location('ah', HERE / 'attack_all_helper.py')
_a = importlib.util.module_from_spec(_ah); _ah.loader.exec_module(_a)
attack_all, flips = _a.attack_all, _a.flips

D = 3                                    # fixed low dimensionality for all conditions
FRAC = [0.0, 0.5, 0.7]
STRUCT = ['random', 'spatial']
N_FOLDS = 5
SEEDS = [42, 58, 125, 7, 13, 21, 99, 123, 200, 777] + list(range(300, 320))
CONDS = ['iris_contested', 'iris_separated', 'blobs_healthy', 'rand_null']


def get_data(name):
    """Return (X standardized to D dims, y, tc, feat) — data fixed; CV split varies by seed (as for real sets)."""
    if name.startswith('iris'):
        X, y = load_iris(return_X_y=True)
        X = StandardScaler().fit_transform(X)
        F, _ = f_classif(X, y)
        cols = np.argsort(np.nan_to_num(F, nan=-np.inf))[::-1][:D]
        Xk = X[:, cols]
        tc = 2 if name == 'iris_contested' else 0        # contested vs well-separated class
        return Xk, y, tc, 0                              # spatial deletion along top-F (local col 0)
    if name == 'blobs_healthy':
        X, y = make_blobs(n_samples=150, centers=3, n_features=D, cluster_std=1.0,
                          center_box=(-10, 10), random_state=0)
        X = StandardScaler().fit_transform(X)
        F, _ = f_classif(X, y)
        feat = int(np.argsort(np.nan_to_num(F, nan=-np.inf))[::-1][0])
        return X, y, 0, feat
    # rand_null: gaussian features, RANDOM labels, no structure
    rng = np.random.default_rng(0)
    X = StandardScaler().fit_transform(rng.standard_normal((150, D)))
    y = rng.integers(0, 3, 150)
    return X, y, 0, 0


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


def run_cell(name, structure, frac, seed):
    X, y, tc, feat = get_data(name)
    yoh = OneHotEncoder(sparse_output=False).fit(y.reshape(-1, 1))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed)
    folds = []
    for tr, te in skf.split(X, y):
        Xt, Xv, yt, yv = X[tr], X[te], y[tr], y[te]
        Xa, ya = inject_delete(Xt, yt, tc, feat, frac, structure, rng)
        clf = DecisionTreeClassifier(max_depth=None, random_state=42).fit(Xa, ya)
        art = SklearnClassifier(clf); art.fit(Xa, yoh.transform(ya.reshape(-1, 1)))
        tacc = float((np.argmax(art.predict(Xa), axis=1) == ya).mean())
        pv = np.argmax(art.predict(Xv), axis=1)
        vacc = float((pv == yv).mean())
        mmask = yv == tc
        min_recall = float((pv[mmask] == tc).mean()) if mmask.sum() else np.nan
        c = pv == yv
        adv, _ = attack_all(art, Xv)                     # every test point, model's own prediction
        flip = flips(art, adv, pv)
        adv_all, adv_c = adv[flip], adv[c & flip]        # correct-only subset = old pipeline
        md = cluster_stats(adv_all)[2] if len(adv_all) else np.nan
        md_c = cluster_stats(adv_c)[2] if len(adv_c) else np.nan
        folds.append(dict(tacc=tacc, vacc=vacc, min_recall=min_recall, nadv=len(adv_all),
                          nadv_correct=len(adv_c), mean_dist=md, mean_dist_correct=md_c))
    r = {k: float(np.nanmean([f[k] for f in folds])) for k in folds[0]}
    r.update(dataset=name, tc=int(tc), frac=float(frac), structure=structure, seed=int(seed))
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
    cols = ['dataset', 'tc', 'frac', 'structure', 'seed', 'tacc', 'vacc', 'min_recall', 'nadv', 'mean_dist',
            'nadv_correct', 'mean_dist_correct']
    rows, t0 = [], time.time()
    grid = []
    for name in CONDS:
        for seed in SEEDS:
            grid.append((name, 'random', 0.0, seed))
            for st, fr in product(STRUCT, FRAC[1:]):
                grid.append((name, st, fr, seed))
    for g in grid:
        rows.append(run_cell(*g))
    df = pl.DataFrame([{c: r.get(c, np.nan) for c in cols} for r in rows])
    df.write_parquet(OUT)

    print(f"\n{'='*94}\nNegative control @ D={D}: is spatial-vs-random spread the DEFECT or a low-D artifact?\n{'='*94}")
    print(f"{'condition':>16} | {'clean vacc':>10} | " + " | ".join(f"f{fr}: spat/rand  d(sep)  recallΔ" for fr in FRAC[1:]))
    for name in CONDS:
        v0 = df.filter((pl.col('dataset') == name) & (pl.col('structure') == 'random') & (pl.col('frac') == 0.0))['vacc'].mean()
        cells = []
        for fr in FRAC[1:]:
            def col(st, c):
                return df.filter((pl.col('dataset') == name) & (pl.col('structure') == st) & (pl.col('frac') == fr))[c].to_numpy()
            base = df.filter((pl.col('dataset') == name) & (pl.col('structure') == 'random') & (pl.col('frac') == 0.0))['mean_dist'].to_numpy().mean()
            d, sep = cohend(col('spatial', 'mean_dist') / base, col('random', 'mean_dist') / base)
            recd = np.nanmean(col('spatial', 'min_recall')) - np.nanmean(col('random', 'min_recall'))
            cells.append(f"{col('spatial','mean_dist').mean()/base:.2f}/{col('random','mean_dist').mean()/base:.2f}  {d:+.2f}({'Y' if sep else 'n'})  {recd:+.2f}")
        tag = 'POSITIVE' if name == 'iris_contested' else 'control '
        print(f"{name:>16} | {v0:>10.3f} | " + " | ".join(cells) + f"   [{tag}]")
    print(f"\nDONE: {len(rows)} rows -> {OUT.name} in {(time.time()-t0)/60:.1f}m")


if __name__ == '__main__':
    main()
