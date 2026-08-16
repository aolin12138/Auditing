"""Phase 1 unified sweep + second-dataset variance check.

For a fixed (model, attack) — overfit tree + white-box DecisionTreeAttack, deterministic, no
hangs — sweep defect severity and record BOTH accuracy and adversarial spread, factored by:
  structure ∈ {spatial (coverage gap), random (imbalance)}   — the defect type
  protocol  ∈ {train_only (clean test), before_split (deletes the band from test too)}

Run on TWO datasets (iris 4-D, wine 13-D) to check the findings are not iris-specific and to
gauge cross-dataset variance. Features standardized (required for wine: proline dominates
otherwise). Target class + spatial feature chosen per dataset by measurement (most contested
class = lowest baseline recall; spatial feature = most class-discriminative), documented in CFG.

-> results_variance.parquet   (one row per dataset x structure x protocol x frac x seed, 5-fold mean)
Reuses inject_delete (./run_imbalance.py, train-only) and cluster_stats (../outlier_experiment).
Usage: python run_variance.py
"""
import os
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
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import DecisionTreeAttack

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results_variance.parquet'
_spec = importlib.util.spec_from_file_location('imb_lib', HERE / 'run_imbalance.py')
_imb = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_imb)
inject_delete = _imb.inject_delete
_os = importlib.util.spec_from_file_location('ol', HERE.parent / 'outlier_experiment' / 'run_experiment.py')
_o = importlib.util.module_from_spec(_os); _os.loader.exec_module(_o)
cluster_stats = _o.cluster_stats

# per-dataset config (tc = most contested class; feat = most class-discriminative), from run_variance probe
CFG = {
    'iris': dict(load=load_iris, tc=2, feat=3),   # virginica; petal width
    'wine': dict(load=load_wine, tc=0, feat=12),  # class 0; proline
}
FRAC = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
STRUCT = ['random', 'spatial']
PROTO = ['train_only', 'before_split']
N_FOLDS = 5
SEEDS = [42, 58, 125, 7, 13, 21, 99, 123, 200, 777] + list(range(300, 320))   # 30 seeds (tight CIs)
COLS = ['dataset', 'model', 'attack', 'structure', 'protocol', 'tc', 'feat', 'frac', 'seed',
        'tacc', 'vacc', 'min_recall', 'nadv', 'mean_dist']


def delete_full(X, y, tc, feat, frac, structure, rng):
    """Before-split deletion from the FULL dataset (removes the points from train AND test)."""
    if frac <= 0:
        return X, y
    idx = np.where(y == tc)[0]
    nrem = min(int(len(idx) * frac), len(idx) - 3)
    if nrem <= 0:
        return X, y
    if structure == 'spatial':
        rem = idx[np.argsort(X[idx, feat])][:nrem]        # contiguous low-feature band = a spatial hole
    else:
        rem = rng.choice(idx, size=nrem, replace=False)   # uniform = imbalance
    keep = np.ones(len(y), dtype=bool); keep[rem] = False
    return X[keep], y[keep]


def run_cell(dataset, structure, protocol, frac, seed):
    cfg = CFG[dataset]; tc, feat = cfg['tc'], cfg['feat']
    X, y = cfg['load'](return_X_y=True)
    X = StandardScaler().fit_transform(X)                 # standardize (required for wine)
    yoh = OneHotEncoder(sparse_output=False).fit(y.reshape(-1, 1))
    rng = np.random.default_rng(seed)
    if protocol == 'before_split' and frac > 0:
        X, y = delete_full(X, y, tc, feat, frac, structure, rng)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    folds = []
    for tr, te in skf.split(X, y):
        Xt, Xv, yt, yv = X[tr], X[te], y[tr], y[te]
        if protocol == 'train_only' and frac > 0:
            Xt, yt = inject_delete(Xt, yt, tc, feat, frac, structure, rng)
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
    r.update(dataset=dataset, model='tree', attack='dta', structure=structure, protocol=protocol,
             tc=int(tc), feat=int(feat), frac=float(frac), seed=int(seed))
    return r


def main():
    rows, t0 = [], time.time()
    grid = []
    for ds, seed in product(CFG, SEEDS):
        grid.append((ds, 'random', 'train_only', 0.0, seed))         # frac=0 baseline (shared)
        for struct, proto, frac in product(STRUCT, PROTO, FRAC[1:]):
            grid.append((ds, struct, proto, frac, seed))
    for i, (ds, s, p, f, sd) in enumerate(grid, 1):
        rows.append(run_cell(ds, s, p, f, sd))
        if i % 100 == 0 or i == len(grid):
            print(f'  [{i}/{len(grid)}] {(time.time()-t0)/60:.1f}m', flush=True)
    pl.DataFrame([{c: r.get(c, np.nan) for c in COLS} for r in rows]).write_parquet(OUT)
    print(f'DONE: {len(rows)} rows -> {OUT.name} in {(time.time()-t0)/60:.1f}m', flush=True)


if __name__ == '__main__':
    main()
