"""
Outlier defect, Phase 0 (signal check) — see PLAN.md.

Defect: inject `n_out` correctly-labeled outliers at distance `k * class_std` from a
target class's centroid, OUTWARD (away from the global centroid), into the TRAINING
fold only. Test fold stays clean (fixes the coverage-gap test-set confound).

Model: DecisionTreeClassifier(max_depth=None)  -> OVERFITS (fits training ~100%), so a
lone far outlier actually deforms the boundary (a depth-3 tree would just ignore it).
Attack: ART DecisionTreeAttack (deterministic, exact, no hangs).
Metric: OPTICS within-cluster mean pairwise distance (spread), overall AND per class.

Phase 0a: go/no-go — strongest cell (k=6, n_out=5) vs clean baseline (n_out=0).
Phase 0b: full factorial k in {2,4,6,8} x n_out in {1,3,5,10} (+ baseline), 3 seeds.
Everything except k and n_out is held fixed. Usage: python run_experiment.py
"""
import os
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_v] = '1'

import warnings
from pathlib import Path
from itertools import product
import numpy as np, polars as pl
warnings.filterwarnings('ignore')

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import OPTICS
from scipy.spatial.distance import pdist
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import DecisionTreeAttack

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results.parquet'

K_LEVELS = [2, 4, 6, 8]
N_OUT_LEVELS = [1, 3, 5, 10]
DIRECTIONS = ['outward', 'toward']
SEEDS = [42, 58, 125]
# tc=2 (virginica) is a CONTESTED class bordering versicolor. The _diag.py sweep showed
# tc=0 (setosa, isolated) + outward = EXACTLY null (Delta leaves 0, Delta spread 0): outward
# outliers land in empty space and the overfit tree never reacts. tc=2 gives the cleanest
# outward(null) vs toward(signal) contrast. Direction is therefore a Phase-0 factor.
TC = 2
N_FOLDS = 5
COLS = ['defect', 'k', 'n_out', 'tc', 'direction', 'ref', 'kind', 'depth', 'model',
        'attack', 'seed', 'outlier_seed', 'feat', 'tacc', 'vacc', 'asucc', 'nadv',
        'density', 'nclust', 'mean_dist', 'clust_size', 'aiden_density',
        'spread_c0', 'spread_c1', 'spread_c2']


# ---------------------------------------------------------------- injection
def _class_means(X, y):
    return {int(c): X[y == c].mean(axis=0) for c in np.unique(y)}


def _nearest_other_class(tc, X, y):
    means = _class_means(X, y)
    mu_c = means[int(tc)]
    others = [(c, m) for c, m in means.items() if c != int(tc)]
    return min(others, key=lambda p: np.linalg.norm(p[1] - mu_c))[0]


def inject_outliers(X_tr, y_tr, tc, k, n_out, direction='outward', ref='class',
                    kind='correct', axis='radial', feat=None, rng=None, n_out_frac=None):
    """Add outliers to class `tc` in the TRAINING data only. Distance k is in units of the
    class's mean per-feature std. `direction`: 'outward' (away from global centroid),
    'toward' (nearest other class), or 'random' (independent random unit vector per outlier,
    i.e. scattered on a sphere of radius k*std around the centroid). If `n_out_frac` is given,
    n_out = round(n_out_frac * current class size) — count as a fraction of the class."""
    if rng is None:
        rng = np.random.default_rng(0)
    n_class = int((y_tr == tc).sum())
    if n_out_frac is not None:
        n_out = max(1, int(round(n_out_frac * n_class)))
    if n_out == 0:
        return X_tr.copy(), y_tr.copy()
    Xc = X_tr[y_tr == tc]
    mu_c = Xc.mean(axis=0)
    sig_c = Xc.std(axis=0)
    s = float(sig_c.mean())                        # isotropic characteristic scale
    g_mu = X_tr.mean(axis=0)                        # global centroid
    base = mu_c if ref == 'class' else g_mu
    means = _class_means(X_tr, y_tr)

    if direction == 'random':                      # each outlier a random direction from centroid
        pts = []
        for _ in range(n_out):
            rd = rng.normal(size=mu_c.shape); rd = rd / (np.linalg.norm(rd) + 1e-12)
            pts.append(base + k * s * rd)
        Xo = np.vstack(pts)
    else:
        if axis == 'feat':                         # axis-aligned displacement
            d = np.zeros_like(mu_c)
            if direction == 'outward':
                d[feat] = np.sign((mu_c - g_mu)[feat]) or 1.0
            else:
                noc = _nearest_other_class(tc, X_tr, y_tr)
                d[feat] = np.sign((means[noc] - mu_c)[feat]) or 1.0
        else:                                      # radial
            if direction == 'outward':
                d = mu_c - g_mu                    # away from data center
            else:
                noc = _nearest_other_class(tc, X_tr, y_tr)
                d = means[noc] - mu_c              # toward nearest other class
            d = d / (np.linalg.norm(d) + 1e-12)
        pts = [base + k * s * d + rng.normal(0, 0.3, size=mu_c.shape) * sig_c
               for _ in range(n_out)]
        Xo = np.vstack(pts)
    label = int(tc) if kind == 'correct' else _nearest_other_class(tc, X_tr, y_tr)
    yo = np.full(n_out, label, dtype=y_tr.dtype)
    return np.vstack([X_tr, Xo]), np.hstack([y_tr, yo])


# ---------------------------------------------------------------- metric (copied from dtree runner)
def cluster_stats(points, ms=3):
    """(fixed_density, n_clusters, mean_dist=spread, cluster_size, aiden_buggy_density)."""
    if len(points) < ms + 1:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    o = OPTICS(min_samples=ms, xi=0.05, min_cluster_size=ms).fit(points)
    dens_fixed, dists_fixed, sizes, dens_aiden = [], [], [], []
    for c in set(o.labels_) - {-1}:
        cp = points[o.labels_ == c]
        if len(cp) < 2:
            continue
        dd = pdist(cp).mean()
        dens_fixed.append(len(cp) / (dd + 1)); dists_fixed.append(dd); sizes.append(len(cp))
        buggy = [np.linalg.norm((cp[i], cp[j])) for i in range(len(cp)) for j in range(i + 1, len(cp))]
        dens_aiden.append(len(cp) / (np.mean(buggy) + 1))
    if not dens_fixed:
        return np.nan, 0, np.nan, np.nan, np.nan
    return (float(np.mean(dens_fixed)), len(dens_fixed), float(np.mean(dists_fixed)),
            float(np.mean(sizes)), float(np.mean(dens_aiden)))


# ---------------------------------------------------------------- attack (per-class labels kept)
def attack_adv(art, Xv, yv):
    p = np.argmax(art.predict(Xv), axis=1)
    c = p == yv
    if c.sum() == 0:
        return np.empty((0, Xv.shape[1])), 0.0, np.empty(0, dtype=yv.dtype)
    adv = DecisionTreeAttack(classifier=art).generate(Xv[c])
    ap = np.argmax(art.predict(adv), axis=1)
    flipped = ap != yv[c]
    return adv[flipped], float(c.mean()), yv[c][flipped]


# ---------------------------------------------------------------- one grid cell
def run_cell(X, y, k, n_out, tc, seed, outlier_seed,
             direction='outward', ref='class', kind='correct', depth=None):
    yoh = OneHotEncoder(sparse_output=False).fit(y.reshape(-1, 1))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    rng = np.random.default_rng(outlier_seed * 1000 + seed)
    classes = [int(c) for c in np.unique(y)]
    folds = []
    for tr, te in skf.split(X, y):
        Xt, Xv, yt, yv = X[tr], X[te], y[tr], y[te]
        Xa, ya = inject_outliers(Xt, yt, tc, k, n_out, direction, ref, kind, rng=rng)
        m = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(Xa, ya)
        art = SklearnClassifier(m)
        art.fit(Xa, yoh.transform(ya.reshape(-1, 1)))          # matches the proven pipeline
        tacc = float((np.argmax(art.predict(Xa), axis=1) == ya).mean())
        vacc = float((np.argmax(art.predict(Xv), axis=1) == yv).mean())
        adv, succ, labels = attack_adv(art, Xv, yv)
        dens, nc, md, cs, aiden = cluster_stats(adv)
        rec = {'tacc': tacc, 'vacc': vacc, 'asucc': succ, 'nadv': len(adv),
               'density': dens, 'nclust': nc, 'mean_dist': md, 'clust_size': cs,
               'aiden_density': aiden}
        for cls in classes:
            sub = adv[labels == cls] if len(adv) else adv
            rec[f'spread_c{cls}'] = cluster_stats(sub)[2]
        folds.append(rec)
    r = {kk: float(np.nanmean([f[kk] for f in folds])) for kk in folds[0]}
    r.update(defect='outlier', k=float(k), n_out=int(n_out), tc=int(tc),
             direction=direction, ref=ref, kind=kind,
             depth=-1 if depth is None else int(depth), model='tree', attack='dta',
             seed=int(seed), outlier_seed=int(outlier_seed), feat=-1)
    return r


# ---------------------------------------------------------------- driver
def main():
    iris = load_iris(); X, y = iris.data, iris.target
    sc = f'spread_c{TC}'

    print('=== Phase 0a: go/no-go (seed=42, tc=%d virginica) ===' % TC, flush=True)
    base = run_cell(X, y, 0, 0, TC, 42, 0)
    tw = run_cell(X, y, 8, 5, TC, 42, 0, direction='toward')
    ow = run_cell(X, y, 8, 5, TC, 42, 0, direction='outward')
    for name, r in [('baseline n_out=0', base), ('toward  k=8 n=5 ', tw), ('outward k=8 n=5 ', ow)]:
        print(f"{name}: tacc={r['tacc']:.3f} vacc={r['vacc']:.3f} "
              f"spread(all)={r['mean_dist']:.3f} {sc}={r[sc]:.3f}", flush=True)
    d_all = tw['mean_dist'] - base['mean_dist']
    d_c = tw[sc] - base[sc]
    d_acc = tw['vacc'] - base['vacc']
    d_ow = ow['mean_dist'] - base['mean_dist']
    print(f"toward:  delta spread(all)={d_all:+.3f}  delta {sc}={d_c:+.3f}  delta vacc={d_acc:+.3f}", flush=True)
    print(f"outward: delta spread(all)={d_ow:+.3f}  (expected ~0 = the null contrast)", flush=True)
    go = np.nan_to_num(d_all) > 0.01 and abs(d_acc) < 0.05
    print('VERDICT:', 'GO (toward signal moved, accuracy ~flat)' if go else 'NO-GO / investigate', flush=True)

    print('\n=== Phase 0b: factorial direction x k x n_out (3 seeds) ===', flush=True)
    rows = []
    for seed in SEEDS:
        rows.append(run_cell(X, y, 0, 0, TC, seed, 0))                     # clean baseline
        for (dirn, k, n_out) in product(DIRECTIONS, K_LEVELS, N_OUT_LEVELS):
            rows.append(run_cell(X, y, k, n_out, TC, seed, 0, direction=dirn))
    df = pl.DataFrame([{c: r[c] for c in COLS} for r in rows])
    df.write_parquet(OUT)
    print(f'wrote {len(rows)} rows -> {OUT}', flush=True)

    agg = [pl.col('mean_dist').mean().round(3).alias('spread'),
           pl.col(sc).mean().round(3).alias(sc),
           pl.col('vacc').mean().round(3).alias('vacc')]
    prod = df.filter(pl.col('n_out') > 0)
    print('\nmain effect of DIRECTION (mean over k, n_out, seeds):', flush=True)
    print(prod.group_by('direction').agg(agg).sort('direction'), flush=True)
    tw_df = prod.filter(pl.col('direction') == 'toward')
    print('\nwithin toward - effect of k (mean over n_out, seeds):', flush=True)
    print(tw_df.group_by('k').agg(agg).sort('k'), flush=True)
    print('\nwithin toward - effect of n_out (mean over k, seeds):', flush=True)
    print(tw_df.group_by('n_out').agg(agg).sort('n_out'), flush=True)
    b = df.filter(pl.col('n_out') == 0)
    print(f"\nclean baseline: spread(all)={b['mean_dist'].mean():.3f} "
          f"{sc}={b[sc].mean():.3f} vacc={b['vacc'].mean():.3f}", flush=True)


if __name__ == '__main__':
    main()
