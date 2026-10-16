"""Coverage gap vs random deletion with EVERY test point attacked (Aidan Jared's original choice).

Why: the standard pipeline attacks only test points the (defective) model classifies correctly.
Under a coverage gap the model misclassifies the test points in the hole, so they are dropped:
(1) the points the defect damages most never reach the geometry, and (2) the hole and random arms
end up measuring different groups of points (Johnson Zhang's fixed-cohort check, 2026-10-16).
Attacking every test point removes both problems: every condition is measured on the same test
fold, and the damaged points are included.

Per fold, the model's own prediction is the starting label for each point (ART untargeted attack):
correctly classified points are pushed to a wrong class; misclassified points are pushed out of the
class the model wrongly assigns. A candidate is kept if the model's prediction actually changed.
Recorded per cell (5-fold mean):
  spread_all       spread of the cloud from ALL test points             <- the new measurement
  spread_correct   spread of the cloud from correctly classified points  <- the standard pipeline
                   (same run, same adversarial points; must reproduce results_variance for the tree)
  n_all, n_correct, n_wrong   successful adversarial examples from all / correct / misclassified points
  vacc, min_recall test accuracy, recall of the deleted class
Configs: iris (petal width, standardised) and wine (proline, standardised) = Fig 2 / run_variance;
iris_plen (petal length, raw features) = run_imbalance / Johnson's fixed-cohort setting.
Usage: python run_attack_all.py [--model svm] [--smoke] [--nseeds N]
-> results_attack_all.parquet / results_attack_all_svm.parquet
"""
import os, sys
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_v] = '1'
import warnings, time, importlib.util
from pathlib import Path
from itertools import product
from concurrent.futures import ProcessPoolExecutor
import numpy as np, polars as pl
warnings.filterwarnings('ignore')
from sklearn.datasets import load_iris, load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import DecisionTreeAttack, HopSkipJump

HERE = Path(__file__).resolve().parent
MODEL = 'svm' if '--model' in sys.argv and sys.argv[sys.argv.index('--model') + 1] == 'svm' else 'tree'
ATTACK = 'hsj' if MODEL == 'svm' else 'dta'
OUT = HERE / ('results_attack_all_svm.parquet' if MODEL == 'svm' else 'results_attack_all.parquet')
_spec = importlib.util.spec_from_file_location('imb_lib', HERE / 'run_imbalance.py')
_imb = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_imb)
inject_delete = _imb.inject_delete
_os = importlib.util.spec_from_file_location('ol', HERE.parent / 'outlier_experiment' / 'run_experiment.py')
_o = importlib.util.module_from_spec(_os); _os.loader.exec_module(_o)
cluster_stats = _o.cluster_stats

CFG = {   # name: (loader, target class, deletion feature, standardise)
    'iris': (load_iris, 2, 3, True),        # = run_variance (Fig 2)
    'wine': (load_wine, 0, 12, True),       # = run_variance (Fig 2)
    'iris_plen': (load_iris, 2, 2, False),  # = run_imbalance / Johnson's fixed-cohort setting
}
FRAC = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
N_FOLDS = 5
SEEDS = [42, 58, 125, 7, 13, 21, 99, 123, 200, 777] + list(range(300, 320))
COLS = ['dataset', 'model', 'attack', 'structure', 'frac', 'seed', 'vacc', 'min_recall',
        'n_all', 'n_correct', 'n_wrong', 'attack_err', 'spread_all', 'spread_correct']


def _fit(Xt, yt, yoh):
    if MODEL == 'svm':
        return SklearnClassifier(SVC(kernel='rbf', probability=True, random_state=42).fit(Xt, yt))
    m = DecisionTreeClassifier(max_depth=None, random_state=42).fit(Xt, yt)
    art = SklearnClassifier(m); art.fit(Xt, yoh.transform(yt.reshape(-1, 1)))   # as run_variance
    return art


def _attack(art, X):
    """Untargeted attack on every row of X, from the model's own prediction. Returns (adv, n_err)."""
    if MODEL == 'svm':
        return HopSkipJump(classifier=art, norm=2, max_iter=10, max_eval=200, init_eval=50,
                           verbose=False).generate(X), 0
    atk, out, err = DecisionTreeAttack(classifier=art, verbose=False), X.copy(), 0
    for i in range(len(X)):                 # per point: ART can raise IndexError for one point
        try:
            out[i] = atk.generate(X[i:i + 1])[0]
        except IndexError:                  # see analysis/verify_dta_destination.py; point left unmoved
            err += 1
    return out, err


def _spread(cloud):
    return cluster_stats(cloud)[2] if len(cloud) else np.nan


def run_cell(args):
    ds, structure, frac, seed = args
    loader, tc, feat, std = CFG[ds]
    X, y = loader(return_X_y=True)
    if std:
        X = StandardScaler().fit_transform(X)
    yoh = OneHotEncoder(sparse_output=False).fit(y.reshape(-1, 1))
    rng = np.random.default_rng(seed)       # same consumption order as run_variance (train_only)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    folds = []
    for tr, te in skf.split(X, y):
        Xt, Xv, yt, yv = X[tr], X[te], y[tr], y[te]
        if frac > 0:
            Xt, yt = inject_delete(Xt, yt, tc, feat, frac, structure, rng)
        art = _fit(Xt, yt, yoh)
        pv = np.argmax(art.predict(Xv), axis=1)
        adv, err = _attack(art, Xv)
        moved = np.argmax(art.predict(adv), axis=1) != pv         # prediction actually changed
        correct = pv == yv
        mm = yv == tc
        folds.append(dict(vacc=float(correct.mean()),
                          min_recall=float((pv[mm] == tc).mean()) if mm.sum() else np.nan,
                          n_all=int(moved.sum()), n_correct=int((moved & correct).sum()),
                          n_wrong=int((moved & ~correct).sum()), attack_err=err,
                          spread_all=_spread(adv[moved]), spread_correct=_spread(adv[moved & correct])))
    r = {k: float(np.nanmean([f[k] for f in folds])) for k in folds[0]}
    r.update(dataset=ds, model=MODEL, attack=ATTACK, structure=structure, frac=float(frac), seed=int(seed))
    return r


def main():
    global SEEDS
    if '--smoke' in sys.argv:
        for ds in CFG:
            for st, fr in (('random', 0.0), ('spatial', 0.8), ('random', 0.8)):
                t0 = time.time(); r = run_cell((ds, st, fr, 42))
                print(f'  smoke {MODEL} {ds:9s} {st:7s} {fr}: {time.time()-t0:.1f}s vacc={r["vacc"]:.3f} '
                      f'n_all={r["n_all"]:.1f} n_correct={r["n_correct"]:.1f} n_wrong={r["n_wrong"]:.1f} '
                      f'spread_all={r["spread_all"]:.3f} spread_correct={r["spread_correct"]:.3f}', flush=True)
        return
    if '--nseeds' in sys.argv:
        SEEDS = SEEDS[:int(sys.argv[sys.argv.index('--nseeds') + 1])]
    grid = [(ds, 'random', 0.0, s) for ds in CFG for s in SEEDS]
    grid += [(ds, st, f, s) for ds in CFG for st in ('random', 'spatial') for f in FRAC[1:] for s in SEEDS]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(14, os.cpu_count() or 4)) as ex:
        rows = list(ex.map(run_cell, grid, chunksize=4))
    pl.DataFrame([{c: r.get(c, np.nan) for c in COLS} for r in rows]).write_parquet(OUT)
    print(f'DONE: {len(rows)} rows -> {OUT.name} in {(time.time()-t0)/60:.1f}m', flush=True)


if __name__ == '__main__':
    main()
