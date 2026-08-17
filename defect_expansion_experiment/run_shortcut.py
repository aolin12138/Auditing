"""Phase 0 shortcut / spurious feature (Clever Hans) — PLAN §3.

Appends a 5th feature: label-correlated in TRAIN (corr * (y-1) + N(0,1)) but pure noise in
TEST (N(0,1)). The model can cheat by reading the shortcut axis. Sweep corr {0,.5,1,2,4}
(0 = pure-noise control axis), overfit tree + white-box DecisionTreeAttack, 30 seeds,
iris (4-D) + wine (13-D). Deterministic (no hangs) — same runner pattern as run_variance.py.

Per cell (fold-mean) we record:
  tacc, vacc             train/test accuracy (Clever Hans cost: vacc drops as the model leans)
  nsp_splits             splits in the trained tree on the spurious feature (index 4) —
                         MANIPULATION CHECK: does the tree actually lean on the shortcut?
  nadv, adv_l2           number of successful adversarial examples + their mean L2 displacement
  spur_frac              mean over adv points of dx_spur^2 / ||dx||^2 — the fraction of the
                         adversarial displacement carried by the spurious axis (H1: rises
                         with corr; control ~1/(d+1) by axis symmetry)
  spread_m0, spread_m4   scalar adversarial spread (raw OPTICS + kNN-local from §8)

-> results_shortcut.parquet   (one row per dataset x corr x seed)
Usage: python run_shortcut.py [--smoke]
"""
import os
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_v] = '1'
import warnings, time, importlib.util, sys
warnings.filterwarnings('ignore')
from pathlib import Path
from itertools import product
import numpy as np, polars as pl
from sklearn.datasets import load_iris, load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import DecisionTreeAttack

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results_shortcut.parquet'
CFG = {'iris': dict(load=load_iris, tc=2, title='iris (4-D)'),
       'wine': dict(load=load_wine, tc=0, title='wine (13-D)')}
CORR = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
N_FOLDS = 5
SEEDS = [42, 58, 125, 7, 13, 21, 99, 123, 200, 777] + list(range(300, 320))   # 30 seeds
SPUR_IDX = 4                       # appended feature index (iris: 4; wine: 13 — set per cell)
COLS = ['dataset', 'corr', 'seed', 'tacc', 'vacc', 'nsp_splits', 'spur_depth', 'nadv', 'adv_l2',
        'spur_frac', 'spread_m0', 'spread_m4']

_os = importlib.util.spec_from_file_location('ol', HERE.parent / 'outlier_experiment' / 'run_experiment.py')
_o = importlib.util.module_from_spec(_os); _os.loader.exec_module(_o)
cluster_stats = _o.cluster_stats
_rm = importlib.util.spec_from_file_location('rm', HERE / 'run_robust_metric.py')
_r = importlib.util.module_from_spec(_rm); _rm.loader.exec_module(_r)
m4_knn_local = _r.m4_knn_local


def run_cell(dataset, corr, seed):
    X, y = CFG[dataset]['load'](return_X_y=True)
    X = StandardScaler().fit_transform(X)                 # standardize REAL features only
    d = X.shape[1]; spur = d                             # spurious axis index (last)
    yoh = OneHotEncoder(sparse_output=False).fit(y.reshape(-1, 1))
    rng = np.random.default_rng(seed)
    z = y - 1                                             # {-1, 0, +1} class encoding
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    folds = []
    for tr, te in skf.split(X, y):
        noise_tr = rng.normal(0.0, 1.0, len(tr))
        noise_te = rng.normal(0.0, 1.0, len(te))
        Xt = np.column_stack([X[tr], corr * z[tr] + noise_tr])   # train: signal + noise
        Xv = np.column_stack([X[te], noise_te])                  # test: pure noise
        yt, yv = y[tr], y[te]
        m = DecisionTreeClassifier(max_depth=None, random_state=42).fit(Xt, yt)
        art = SklearnClassifier(m); art.fit(Xt, yoh.transform(yt.reshape(-1, 1)))
        tacc = float((np.argmax(art.predict(Xt), axis=1) == yt).mean())
        pv = np.argmax(art.predict(Xv), axis=1)
        vacc = float((pv == yv).mean())
        nsp = int((m.tree_.feature == spur).sum())        # manipulation check
        spur_depth = np.nan
        if nsp:
            dep = np.zeros(m.tree_.node_count, dtype=int)
            stack = [(0, 0)]
            while stack:
                n, d = stack.pop()
                dep[n] = d
                if m.tree_.children_left[n] != -1:
                    stack.append((m.tree_.children_left[n], d + 1))
                    stack.append((m.tree_.children_right[n], d + 1))
            spur_depth = float(dep[m.tree_.feature == spur].min())   # depth of FIRST spurious split (root=0)
        c = pv == yv
        if c.sum():
            x0 = Xv[c]
            adv = DecisionTreeAttack(classifier=art).generate(x0)
            ap = np.argmax(art.predict(adv), axis=1)
            flip = ap != yv[c]
            adv = adv[flip]; x0 = x0[flip]
        else:
            adv = np.empty((0, d + 1))
        if len(adv):
            dx = adv - x0
            l2 = np.linalg.norm(dx, axis=1)
            nz = l2 > 1e-12
            fracs = (dx[:, spur] ** 2)[nz] / (l2[nz] ** 2)
            adv_l2 = float(np.mean(l2)); spur_frac = float(np.mean(fracs)) if len(fracs) else np.nan
        else:
            adv_l2 = spur_frac = np.nan
        folds.append(dict(tacc=tacc, vacc=vacc, nsp_splits=nsp, spur_depth=spur_depth, nadv=len(adv),
                          adv_l2=adv_l2,
                          spur_frac=spur_frac,
                          spread_m0=cluster_stats(adv)[2] if len(adv) else np.nan,
                          spread_m4=m4_knn_local(adv) if len(adv) else np.nan))
    r = {k: float(np.nanmean([f[k] for f in folds])) for k in folds[0]}
    r.update(dataset=dataset, corr=float(corr), seed=int(seed))
    return r


def main():
    if '--smoke' in sys.argv:
        for ds in CFG:
            t0 = time.time(); r = run_cell(ds, 2.0, 42)
            print(f'  smoke {ds} corr=2: {time.time()-t0:.1f}s vacc={r["vacc"]:.3f} '
                  f'nsp={r["nsp_splits"]:.0f} spur_frac={r["spur_frac"]:.3f} nadv={r["nadv"]:.0f}', flush=True)
        return
    rows, t0 = [], time.time()
    grid = list(product(CFG, CORR, SEEDS))
    for i, (ds, corr, seed) in enumerate(grid, 1):
        rows.append(run_cell(ds, corr, seed))
        if i % 50 == 0 or i == len(grid):
            pl.DataFrame([{c: r.get(c, np.nan) for c in COLS} for r in rows]).write_parquet(OUT)
            print(f'  [{i}/{len(grid)}] {(time.time()-t0)/60:.1f}m (checkpointed {len(rows)} rows)', flush=True)
    print(f'DONE: {len(rows)} rows -> {OUT.name} in {(time.time()-t0)/60:.1f}m', flush=True)


if __name__ == '__main__':
    main()
