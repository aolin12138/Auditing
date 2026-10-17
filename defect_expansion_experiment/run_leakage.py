"""Phase 0 train-test leakage — PLAN §4.

mode=leak: copy n_leak random TRAIN points into the TEST fold (optionally jittered). Standard
test accuracy is *inflated* because the model has memorized them. The question: does the
adversarial geometry flag the memorization that accuracy actively hides?

Sweep n_leak {0,2,5,10,20}, overfit tree + white-box DTA, 30 seeds, iris (4-D) + wine (13-D).
The test set is contaminated; the train fold is unchanged (leak mode). We tag each test point
leaked/clean and split the metrics.

Per cell (fold-mean) we record:
  vacc            accuracy on the FULL (contaminated) test set  — H2: rises with n_leak
  vacc_clean      accuracy on the non-leaked test points only   — the honest accuracy (flat)
  vacc_leak       accuracy on the leaked points (~1.0, memorized)
  margin_leak     mean adversarial L2 perturbation for LEAKED points  — H1: memorization locality
  margin_clean    same for CLEAN test points (every point attacked; *_correct = old pipeline's
                  correctly-classified-only subset, kept as a sensitivity check)
  locden_leak     mean kNN distance (k=3) in the adv cloud for leaked-origin points (local spread)
  locden_clean    same for clean-origin points
  spread_m0       global OPTICS spread (reference — leakage is LOCAL, so this should be flat)
  nadv, nleak_adv supporting counts

-> results_leakage.parquet   (one row per dataset x n_leak x seed)
Usage: python run_leakage.py [--nseeds N] [--smoke]
"""
import os
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_v] = '1'
import warnings, time, importlib.util, sys
warnings.filterwarnings('ignore')
from pathlib import Path
from itertools import product
import numpy as np, polars as pl
from scipy.spatial.distance import cdist
from sklearn.datasets import load_iris, load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import DecisionTreeAttack

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results_leakage.parquet'
CFG = {'iris': load_iris, 'wine': load_wine}
N_LEAK = [0, 2, 5, 10, 20]
JITTER = 0.0                      # Phase 0: exact copies (cleanest test)
N_FOLDS = 5
SEEDS = [42, 58, 125, 7, 13, 21, 99, 123, 200, 777] + list(range(300, 320))   # 30 seeds
COLS = ['dataset', 'n_leak', 'seed', 'vacc', 'vacc_clean', 'vacc_leak', 'nadv', 'nleak_adv',
        'margin_leak', 'margin_clean', 'locden_leak', 'locden_clean', 'spread_m0',
        'nadv_correct', 'margin_leak_correct', 'margin_clean_correct', 'locden_leak_correct',
        'locden_clean_correct', 'spread_m0_correct']

_os = importlib.util.spec_from_file_location('ol', HERE.parent / 'outlier_experiment' / 'run_experiment.py')
_o = importlib.util.module_from_spec(_os); _os.loader.exec_module(_o)
cluster_stats = _o.cluster_stats
_ah = importlib.util.spec_from_file_location('ah', HERE / 'attack_all_helper.py')
_a = importlib.util.module_from_spec(_ah); _ah.loader.exec_module(_a)
attack_all, flips = _a.attack_all, _a.flips


def _knn_dist(points, k=3):
    """Per-point mean distance to k nearest neighbours within the cloud (local spread)."""
    if len(points) < k + 1:
        return np.full(len(points), np.nan)
    d = cdist(points, points)
    np.fill_diagonal(d, np.inf)
    return np.sort(d, axis=1)[:, :k].mean(axis=1)


def run_cell(dataset, n_leak, seed):
    X, y = CFG[dataset](return_X_y=True)
    X = StandardScaler().fit_transform(X)
    yoh = OneHotEncoder(sparse_output=False).fit(y.reshape(-1, 1))
    rng = np.random.default_rng(seed)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    folds = []
    for tr, te in skf.split(X, y):
        Xt, yt = X[tr], y[tr]
        Xv, yv = X[te], y[te]
        leaked = np.zeros(len(te), dtype=bool)
        if n_leak > 0:
            idx = rng.choice(len(tr), size=min(n_leak, len(tr)), replace=False)
            Xl = Xt[idx] + (rng.normal(0, JITTER, Xt[idx].shape) if JITTER else 0.0)
            Xv = np.vstack([Xv, Xl]); yv = np.concatenate([yv, yt[idx]])
            leaked = np.concatenate([leaked, np.ones(len(idx), dtype=bool)])
        m = DecisionTreeClassifier(max_depth=None, random_state=42).fit(Xt, yt)
        art = SklearnClassifier(m); art.fit(Xt, yoh.transform(yt.reshape(-1, 1)))
        pv = np.argmax(art.predict(Xv), axis=1)
        vacc = float((pv == yv).mean())
        vacc_clean = float((pv[~leaked] == yv[~leaked]).mean()) if (~leaked).any() else np.nan
        vacc_leak = float((pv[leaked] == yv[leaked]).mean()) if leaked.any() else np.nan
        c = pv == yv
        adv, _ = attack_all(art, Xv)                       # every test point, model's own prediction
        flip = flips(art, adv, pv)
        adv_all, x0_all = adv[flip], Xv[flip]
        adv_c, x0_c = adv[c & flip], Xv[c & flip]          # correct-only subset = old pipeline
        leak_all, leak_cf = leaked[flip], leaked[c & flip]
        out = {}
        for tag, a, x, lf in (('', adv_all, x0_all, leak_all), ('_correct', adv_c, x0_c, leak_cf)):
            if len(a):
                margins = np.linalg.norm(a - x, axis=1)
                m_leak = float(margins[lf].mean()) if lf.any() else np.nan
                m_clean = float(margins[~lf].mean()) if (~lf).any() else np.nan
                kd = _knn_dist(a, k=3)
                ld_leak = float(np.nanmean(kd[lf])) if lf.any() else np.nan
                ld_clean = float(np.nanmean(kd[~lf])) if (~lf).any() else np.nan
            else:
                m_leak = m_clean = ld_leak = ld_clean = np.nan
            out['margin_leak' + tag] = m_leak
            out['margin_clean' + tag] = m_clean
            out['locden_leak' + tag] = ld_leak
            out['locden_clean' + tag] = ld_clean
            out['spread_m0' + tag] = cluster_stats(a)[2] if len(a) else np.nan
            out['nadv' + tag] = len(a)
        out['nleak_adv'] = int(leak_all.sum())
        folds.append(dict(vacc=vacc, vacc_clean=vacc_clean, vacc_leak=vacc_leak, **out))
    r = {k: float(np.nanmean([f[k] for f in folds])) for k in folds[0]}
    r.update(dataset=dataset, n_leak=int(n_leak), seed=int(seed))
    return r


def main():
    global SEEDS
    if '--nseeds' in sys.argv:
        SEEDS = SEEDS[:int(sys.argv[sys.argv.index('--nseeds') + 1])]
    if '--smoke' in sys.argv:
        for ds in CFG:
            t0 = time.time(); r = run_cell(ds, 10, 42)
            print(f'  smoke {ds} n_leak=10: {time.time()-t0:.1f}s vacc={r["vacc"]:.3f} '
                  f'vacc_clean={r["vacc_clean"]:.3f} margin_leak={r["margin_leak"]:.3f} '
                  f'margin_clean={r["margin_clean"]:.3f}', flush=True)
        return
    rows, t0 = [], time.time()
    grid = list(product(CFG, N_LEAK, SEEDS))
    for i, (ds, nl, seed) in enumerate(grid, 1):
        rows.append(run_cell(ds, nl, seed))
        if i % 50 == 0 or i == len(grid):
            pl.DataFrame([{c: r.get(c, np.nan) for c in COLS} for r in rows]).write_parquet(OUT)
            print(f'  [{i}/{len(grid)}] {(time.time()-t0)/60:.1f}m (checkpointed {len(rows)} rows)', flush=True)
    print(f'DONE: {len(rows)} rows -> {OUT.name} in {(time.time()-t0)/60:.1f}m', flush=True)


if __name__ == '__main__':
    main()
