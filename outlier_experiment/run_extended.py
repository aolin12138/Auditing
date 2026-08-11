"""Outlier experiment, Phase 0.5 extension. Reuses inject_outliers / cluster_stats
from run_experiment.py. Adds:
  - models: tree_d3 (pruned, DTA), tree_d10 (deep ~= unlimited for iris, DTA), svm (rbf, HSJ)
  - extended grid: k in {2,3,4,5,6,8}, n_out in {1,3,5,10,15,20}, direction in {toward,outward}
  - 10 seeds (tree) / 3 seeds (svm, HSJ is slower) to tighten CIs
Normalisation (spread / per-model clean baseline) is done in plot_extended.py.
Resumable: appends to results_extended.parquet, skips done (model,seed,direction,k,n_out).

Usage:
  python run_extended.py --time-svm   # time one SVM+HSJ cell, then exit
  python run_extended.py --tree       # tree_d3 + tree_d10 grids (fast, DTA)
  python run_extended.py --svm        # svm grid (reduced, HSJ)
  python run_extended.py              # everything (tree then svm)
"""
import os
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_v] = '1'
import sys, time, warnings
from pathlib import Path
from itertools import product
import numpy as np, polars as pl
warnings.filterwarnings('ignore')

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import DecisionTreeAttack, HopSkipJump

from run_experiment import inject_outliers, cluster_stats, TC, N_FOLDS, COLS

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results_extended.parquet'

K_LEVELS = [2, 3, 4, 5, 6, 8]
N_OUT_LEVELS = [1, 3, 5, 10, 15, 20, 30, 40]   # 30/40 = 75%/100% of the 40-per-class train fold (ceiling)
DIRECTIONS = ['toward', 'outward', 'random']         # random added (fill the all-models comparison)
SEEDS = [42, 58, 125, 7, 13, 21, 99, 123, 200, 777]        # 10 seeds for the (fast) tree grid
SVM_SEEDS = SEEDS                                            # 10 seeds (was 3) to tighten CIs
SVM_K = [8, 12]                                             # bounded to strong cell (8) + far probe (12) for the ceiling push
SVM_N = [5, 10, 20, 30, 40]                                 # extend SVM to the class-size ceiling
MODELS = {
    'tree_d3':  {'kind': 'tree', 'depth': 3,  'attack': 'dta'},
    'tree_d10': {'kind': 'tree', 'depth': 10, 'attack': 'dta'},
    'svm':      {'kind': 'svm',  'depth': -1, 'attack': 'hsj'},
}


def make_model(spec):
    if spec['kind'] == 'tree':
        return DecisionTreeClassifier(max_depth=spec['depth'], random_state=42)
    return SVC(kernel='rbf', probability=True, random_state=42)


def attack(spec, art, Xv, yv):
    p = np.argmax(art.predict(Xv), axis=1)
    c = p == yv
    if c.sum() == 0:
        return np.empty((0, Xv.shape[1])), 0.0, np.empty(0, dtype=yv.dtype)
    if spec['attack'] == 'dta':
        adv = DecisionTreeAttack(classifier=art).generate(Xv[c])
    else:
        hs = HopSkipJump(classifier=art, norm=2, max_iter=10, max_eval=200,
                         init_eval=50, verbose=False)
        adv = hs.generate(Xv[c])
    ap = np.argmax(art.predict(adv), axis=1)
    fl = ap != yv[c]
    return adv[fl], float(c.mean()), yv[c][fl]


def run_cell(spec_name, X, y, k, n_out, seed, direction):
    spec = MODELS[spec_name]
    yoh = OneHotEncoder(sparse_output=False).fit(y.reshape(-1, 1))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed)
    classes = [int(c) for c in np.unique(y)]
    folds = []
    for tr, te in skf.split(X, y):
        Xt, Xv, yt, yv = X[tr], X[te], y[tr], y[te]
        Xa, ya = inject_outliers(Xt, yt, TC, k, n_out, direction=direction, rng=rng)
        m = make_model(spec).fit(Xa, ya)
        art = SklearnClassifier(m); art.fit(Xa, yoh.transform(ya.reshape(-1, 1)))
        tacc = float((np.argmax(art.predict(Xa), axis=1) == ya).mean())
        vacc = float((np.argmax(art.predict(Xv), axis=1) == yv).mean())
        adv, succ, labels = attack(spec, art, Xv, yv)
        dens, nc, md, cs, aiden = cluster_stats(adv)
        rec = {'tacc': tacc, 'vacc': vacc, 'asucc': succ, 'nadv': len(adv), 'density': dens,
               'nclust': nc, 'mean_dist': md, 'clust_size': cs, 'aiden_density': aiden}
        for cls in classes:
            sub = adv[labels == cls] if len(adv) else adv
            rec[f'spread_c{cls}'] = cluster_stats(sub)[2]
        folds.append(rec)
    r = {kk: float(np.nanmean([f[kk] for f in folds])) for kk in folds[0]}
    r.update(defect='outlier', k=float(k), n_out=int(n_out), tc=int(TC), direction=direction,
             ref='class', kind='correct', depth=int(spec['depth']), model=spec_name,
             attack=spec['attack'], seed=int(seed), outlier_seed=0, feat=-1)
    return r


def jobs_for(model_name, seeds, ks, ns):
    """(model, seed, direction, k, n_out); baseline rows carry direction='none'."""
    out = []
    for seed in seeds:
        out.append((model_name, seed, 'none', 0, 0))                 # clean baseline
        for (d, k, n) in product(DIRECTIONS, ks, ns):
            out.append((model_name, seed, d, k, n))
    return out


def key(r):
    return (r['model'], r['seed'], r['direction'], round(r['k'], 2), r['n_out'])


def run_jobs(jobs, X, y, label):
    rows = pl.read_parquet(OUT).to_dicts() if OUT.exists() else []
    done = {key(r) for r in rows}
    todo = [j for j in jobs if (j[0], j[1], j[2], round(float(j[3]), 2), j[4]) not in done]
    print(f'[{label}] {len(todo)} cells to run ({len(jobs)-len(todo)} already done)', flush=True)
    t0 = time.time()
    for i, (mn, seed, d, k, n) in enumerate(todo, 1):
        r = run_cell(mn, X, y, k, n, seed, d)
        rows.append(r)
        if i % 50 == 0 or i == len(todo):
            pl.DataFrame([{c: r2[c] for c in COLS} for r2 in rows]).write_parquet(OUT)
            el = time.time() - t0
            print(f'  [{label}] {i}/{len(todo)}  {el/60:.1f}m  ({el/max(i,1):.2f}s/cell)', flush=True)
    pl.DataFrame([{c: r2[c] for c in COLS} for r2 in rows]).write_parquet(OUT)


def main():
    iris = load_iris(); X, y = iris.data, iris.target
    arg = sys.argv[1] if len(sys.argv) > 1 else '--all'

    if arg == '--time-svm':
        t0 = time.time()
        r = run_cell('svm', X, y, 8, 10, 42, 'toward')
        print(f'one SVM+HSJ cell (5 folds): {time.time()-t0:.1f}s  '
              f"vacc={r['vacc']:.3f} spread={r['mean_dist']:.3f}", flush=True)
        return

    if arg in ('--tree', '--all'):
        jt = jobs_for('tree_d3', SEEDS, K_LEVELS, N_OUT_LEVELS) + \
             jobs_for('tree_d10', SEEDS, K_LEVELS, N_OUT_LEVELS)
        run_jobs(jt, X, y, 'tree')
    if arg in ('--svm', '--all'):
        run_jobs(jobs_for('svm', SVM_SEEDS, SVM_K, SVM_N), X, y, 'svm')

    df = pl.read_parquet(OUT)
    print(f'\ntotal rows: {df.height}  models: {sorted(df["model"].unique().to_list())}', flush=True)


if __name__ == '__main__':
    main()
