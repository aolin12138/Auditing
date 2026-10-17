"""PLAN §9 H4 — wine disambiguation: is the §8 wine failure dimensionality, or defect-not-distorting?

§8 showed the spatial (coverage-gap) vs random (imbalance) spread signal is ~flat on 13-D wine, and no
post-fit reduction recovered it. The padded-iris study (FINDINGS_dimensionality.md) showed that when a
signal genuinely EXISTS but is buried by extra dims, feature selection BEFORE fitting recovers it fully.
So the clean test for wine: remove the dimensionality/concentration confound by restricting wine to its
top-k most class-discriminative features (k small => concentration high, like iris), then run the SAME
defect there. Two outcomes:
  - a spatial-vs-random spread gap APPEARS in low-k wine  -> the signal existed and was buried by
    dimensionality  => RECOVERABLE (wine's §8 failure was dimensionality per se).
  - the gap stays absent even at low k  -> the wine defect does NOT distort the boundary geometry the
    way it distorts iris's  => DEFECT-NOT-DISTORTING (the §8 negative is intrinsic, no reduction helps).

Feature ranking = ANOVA F-score (f_classif) on the full standardized dataset (fixed, reproducible).
spatial coverage-gap deletes class tc along the TOP-ranked feature (most discriminative), mirroring iris.
Model/attack = overfit tree + white-box DTA (deterministic). Contrast the SAME pipeline on iris (its
top-k) as a positive control that low-k selection preserves the known iris signal.

-> results_wine_h4.parquet  (one row per dataset x k x structure x frac x seed, 5-fold mean)
Usage: python run_wine_h4.py
"""
import os, sys
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_v] = '1'
import warnings, time, importlib.util
from pathlib import Path
from itertools import product
import numpy as np, polars as pl
warnings.filterwarnings('ignore')
from sklearn.datasets import load_iris, load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import f_classif
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import DecisionTreeAttack

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results_wine_h4.parquet'
_os = importlib.util.spec_from_file_location('ol', HERE.parent / 'outlier_experiment' / 'run_experiment.py')
_o = importlib.util.module_from_spec(_os); _os.loader.exec_module(_o)
cluster_stats = _o.cluster_stats
_ah = importlib.util.spec_from_file_location('ah', HERE / 'attack_all_helper.py')
_a = importlib.util.module_from_spec(_ah); _ah.loader.exec_module(_a)
attack_all, flips = _a.attack_all, _a.flips

# per-dataset contested target class (from run_variance probe)
CFG = {'wine': dict(load=load_wine, tc=0), 'iris': dict(load=load_iris, tc=2)}
KS = {'wine': [2, 3, 4, 6, 13], 'iris': [2, 3, 4]}
FRAC = [0.0, 0.5, 0.7, 0.85]
STRUCT = ['random', 'spatial']
N_FOLDS = 5
SEEDS = [42, 58, 125, 7, 13, 21, 99, 123, 200, 777] + list(range(300, 320))
COLS = ['dataset', 'k', 'structure', 'frac', 'seed', 'top_feat', 'tacc', 'vacc', 'min_recall',
        'nadv', 'mean_dist', 'nadv_correct', 'mean_dist_correct']


def top_features(X, y, k):
    """Indices of the k most class-discriminative features by ANOVA F-score (fixed ranking)."""
    F, _ = f_classif(X, y)
    F = np.nan_to_num(F, nan=-np.inf)
    return np.argsort(F)[::-1][:k]


def inject_delete(Xt, yt, tc, feat, frac, structure, rng):
    """Remove `frac` of class tc: spatial = lowest along `feat` (a hole); random = uniform. Same count."""
    if frac <= 0:
        return Xt, yt
    idx = np.where(yt == tc)[0]
    nrem = int(len(idx) * frac)
    nrem = min(max(nrem, 0), len(idx) - 3)
    if nrem <= 0:
        return Xt, yt
    if structure == 'spatial':
        rem = idx[np.argsort(Xt[idx, feat])][:nrem]
    else:
        rem = rng.choice(idx, size=nrem, replace=False)
    keep = np.ones(len(yt), dtype=bool); keep[rem] = False
    return Xt[keep], yt[keep]


def run_cell(dataset, k, structure, frac, seed):
    cfg = CFG[dataset]; tc = cfg['tc']
    X, y = cfg['load'](return_X_y=True)
    X = StandardScaler().fit_transform(X)
    cols = top_features(X, y, k)                 # top-k discriminative subspace (fixed, full-data ranking)
    Xk = X[:, cols]
    top_local = 0                                # after slicing, feature 0 = the most discriminative (spatial axis)
    yoh = OneHotEncoder(sparse_output=False).fit(y.reshape(-1, 1))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed)
    folds = []
    for tr, te in skf.split(Xk, y):
        Xt, Xv, yt, yv = Xk[tr], Xk[te], y[tr], y[te]
        Xa, ya = inject_delete(Xt, yt, tc, top_local, frac, structure, rng)
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
    r = {kk: float(np.nanmean([f[kk] for f in folds])) for kk in folds[0]}
    r.update(dataset=dataset, k=int(k), structure=structure, frac=float(frac), seed=int(seed),
             top_feat=int(cols[0]))
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
    rows, t0 = [], time.time()
    grid = []
    for ds in CFG:
        for k, seed in product(KS[ds], SEEDS):
            grid.append((ds, k, 'random', 0.0, seed))
            for st, fr in product(STRUCT, FRAC[1:]):
                grid.append((ds, k, st, fr, seed))
    for i, g in enumerate(grid, 1):
        rows.append(run_cell(*g))
        if i % 150 == 0 or i == len(grid):
            pl.DataFrame([{c: r.get(c, np.nan) for c in COLS} for r in rows]).write_parquet(OUT)
            print(f'  [{i}/{len(grid)}] {(time.time()-t0)/60:.1f}m', flush=True)
    df = pl.DataFrame([{c: r.get(c, np.nan) for c in COLS} for r in rows])
    df.write_parquet(OUT)

    print(f"\n{'='*84}\nH4 wine disambiguation: spatial-vs-random spread Cohen's d in the top-k subspace\n{'='*84}")
    for ds in CFG:
        print(f"\n--- {ds} (tc={CFG[ds]['tc']}, spatial along top-F feature) ---")
        print(f"{'k':>3} | " + " | ".join(f"f{fr} d(sep) recallΔ" for fr in FRAC[1:]))
        for k in KS[ds]:
            line = f"{k:>3} | "
            cells = []
            for fr in FRAC[1:]:
                def col(name, st, f):
                    return df.filter((pl.col('dataset') == ds) & (pl.col('k') == k) &
                                     (pl.col('structure') == st) & (pl.col('frac') == f))[name].to_numpy()
                base = col('mean_dist', 'random', 0.0).mean()
                d, sep = cohend(col('mean_dist', 'spatial', fr) / base, col('mean_dist', 'random', fr) / base)
                rec_sp = np.nanmean(col('min_recall', 'spatial', fr))
                rec_rr = np.nanmean(col('min_recall', 'random', fr))
                cells.append(f"{d:+.2f}({'Y' if sep else 'n'})  {rec_sp - rec_rr:+.2f}")
            print(line + " | ".join(cells))
    print(f"\nDONE: {len(rows)} rows -> {OUT.name} in {(time.time()-t0)/60:.1f}m")


if __name__ == '__main__':
    main()
