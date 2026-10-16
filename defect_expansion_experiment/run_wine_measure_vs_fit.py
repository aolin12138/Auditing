"""PLAN §9 follow-up — can we audit the FULL (high-accuracy) wine model, or must we cripple it?

The user's objection to "select features before training": shrinking to k features to see the signal
either (a) hurts model performance (sometimes the data genuinely needs many features), or (b) if you
keep all the good features you are back in high-D and the signal is buried again.

H4 recovered the wine signal by FITTING the tree on the top-k features (a REDUCED model, lower accuracy
at small k). This script tests the alternative that would dodge the performance cost:

  MEASURE-ONLY: fit the FULL 13-D wine model (its real accuracy), attack in 13-D, then measure the
  adversarial spread in the top-k F-score SUBSPACE of the adversarial cloud (a projection of the
  measurement only — the model is untouched).

If MEASURE-ONLY recovers the spatial-vs-random signal => we can audit the real full-accuracy model and
just project the metric (no performance cost — the objection dissolves). If it does NOT (as the
padded-iris oracle did not), then the signal really is a property of the model's fitted boundary and
you must fit in the subspace (the tension is real, and the honest answer is a probe/deployed split).

Compares three quantities per (structure, frac): full-space spread (=§8 null), measure-only top-k
projection, and — from results_wine_h4.parquet — the fit-in-subspace result (the reduced model).

-> results_wine_measure.parquet
Usage: python run_wine_measure_vs_fit.py
"""
import os
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_v] = '1'
import warnings, time, importlib.util
from pathlib import Path
from itertools import product
import numpy as np, polars as pl
warnings.filterwarnings('ignore')
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import f_classif
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import DecisionTreeAttack

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results_wine_measure.parquet'
_os = importlib.util.spec_from_file_location('ol', HERE.parent / 'outlier_experiment' / 'run_experiment.py')
_o = importlib.util.module_from_spec(_os); _os.loader.exec_module(_o)
cluster_stats = _o.cluster_stats

TC = 0
FRAC = [0.0, 0.5, 0.7]
STRUCT = ['random', 'spatial']
KPROJ = [2, 3, 4]
N_FOLDS = 5
SEEDS = [42, 58, 125, 7, 13, 21, 99, 123, 200, 777] + list(range(300, 320))


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


def run_cell(structure, frac, seed, rankF):
    X, y = load_wine(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    feat = int(rankF[0])                     # spatial deletion along the top-F feature (same axis as H4)
    yoh = OneHotEncoder(sparse_output=False).fit(y.reshape(-1, 1))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed)
    folds = []
    for tr, te in skf.split(X, y):
        Xt, Xv, yt, yv = X[tr], X[te], y[tr], y[te]
        Xa, ya = inject_delete(Xt, yt, TC, feat, frac, structure, rng)   # delete in FULL 13-D
        clf = DecisionTreeClassifier(max_depth=None, random_state=42).fit(Xa, ya)   # FULL 13-D model
        art = SklearnClassifier(clf); art.fit(Xa, yoh.transform(ya.reshape(-1, 1)))
        vacc = float((np.argmax(art.predict(Xv), axis=1) == yv).mean())
        pv = np.argmax(art.predict(Xv), axis=1); c = pv == yv
        if c.sum():
            adv = DecisionTreeAttack(classifier=art).generate(Xv[c])
            ap = np.argmax(art.predict(adv), axis=1); adv = adv[ap != yv[c]]
        else:
            adv = np.empty((0, X.shape[1]))
        rec = dict(vacc=vacc, spread_full=cluster_stats(adv)[2] if len(adv) else np.nan)
        for k in KPROJ:                       # MEASURE-ONLY: project the FULL-model adv cloud onto top-k F axes
            cols = rankF[:k]
            rec[f'spread_proj{k}'] = cluster_stats(adv[:, cols])[2] if len(adv) else np.nan
        folds.append(rec)
    r = {kk: float(np.nanmean([f[kk] for f in folds])) for kk in folds[0]}
    r.update(structure=structure, frac=float(frac), seed=int(seed))
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
    X, y = load_wine(return_X_y=True)
    F, _ = f_classif(StandardScaler().fit_transform(X), y)
    rankF = np.argsort(np.nan_to_num(F, nan=-np.inf))[::-1]
    cols = ['structure', 'frac', 'seed', 'vacc', 'spread_full'] + [f'spread_proj{k}' for k in KPROJ]
    rows, t0 = [], time.time()
    grid = [('random', 0.0, s) for s in SEEDS] + [(st, fr, s) for st, fr, s in product(STRUCT, FRAC[1:], SEEDS)]
    for st, fr, s in grid:
        rows.append(run_cell(st, fr, s, rankF))
    df = pl.DataFrame([{c: r.get(c, np.nan) for c in cols} for r in rows])
    df.write_parquet(OUT)

    h4 = pl.read_parquet(HERE / 'results_wine_h4.parquet') if (HERE / 'results_wine_h4.parquet').exists() else None
    print(f"\n{'='*90}\nWINE: MEASURE-ONLY (full 13-D model, project metric) vs FIT-IN-SUBSPACE (H4), Cohen's d\n{'='*90}")
    print(f"full-model clean vacc = {df.filter((pl.col('structure')=='random')&(pl.col('frac')==0.0))['vacc'].mean():.3f}")
    for fr in FRAC[1:]:
        print(f"\n--- frac={fr} ---")
        print(f"{'metric':>26} | {'spatial/base':>12} | {'random/base':>11} | {'Cohen d (sep)':>14}")
        def dof(metric):
            base = df.filter((pl.col('structure') == 'random') & (pl.col('frac') == 0.0))[metric].to_numpy().mean()
            sp = df.filter((pl.col('structure') == 'spatial') & (pl.col('frac') == fr))[metric].to_numpy() / base
            rr = df.filter((pl.col('structure') == 'random') & (pl.col('frac') == fr))[metric].to_numpy() / base
            d, sep = cohend(sp, rr)
            return sp.mean(), rr.mean(), d, sep
        for metric, lab in [('spread_full', 'full 13-D (=§8 null)')] + [(f'spread_proj{k}', f'measure-only proj k={k}') for k in KPROJ]:
            sm, rm, d, sep = dof(metric)
            print(f"{lab:>26} | {sm:>12.3f} | {rm:>11.3f} | {d:>+9.2f} ({'Y' if sep else 'n'})")
        if h4 is not None:                    # fit-in-subspace reference from H4
            for k in KPROJ:
                base = h4.filter((pl.col('dataset')=='wine')&(pl.col('k')==k)&(pl.col('structure')=='random')&(pl.col('frac')==0.0))['mean_dist'].to_numpy().mean()
                sp = h4.filter((pl.col('dataset')=='wine')&(pl.col('k')==k)&(pl.col('structure')=='spatial')&(pl.col('frac')==fr))['mean_dist'].to_numpy()/base
                rr = h4.filter((pl.col('dataset')=='wine')&(pl.col('k')==k)&(pl.col('structure')=='random')&(pl.col('frac')==fr))['mean_dist'].to_numpy()/base
                d, sep = cohend(sp, rr)
                print(f"{'fit-in-subspace k=%d (H4)'%k:>26} | {sp.mean():>12.3f} | {rr.mean():>11.3f} | {d:>+9.2f} ({'Y' if sep else 'n'})")
    print(f"\nDONE: {len(rows)} rows -> {OUT.name} in {(time.time()-t0)/60:.1f}m")


if __name__ == '__main__':
    main()
