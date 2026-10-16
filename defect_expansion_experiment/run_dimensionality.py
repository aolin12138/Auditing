"""PLAN §9 Arm 1 — dimensionality sweep on PADDED IRIS (the controlled decay baseline).

The §8 negative (spread collapses on 13-D wine) could not tell apart two causes:
  (1) dimensionality per se — distance concentration numbs the mean pairwise distance, burying a
      real geometry signal, or
  (2) the defect genuinely does not distort the boundary in higher-D (different data, not more dims).
Wine confounds "more dims" with "different dataset". This runner removes that confound.

Controlled design: take the VERIFIED iris defects — coverage-gap (spatial deletion) vs imbalance
(random deletion), where scalar spread works at 4-D — and PAD with `m` pure-noise N(0,1) dimensions
(standardized). The signal lives ENTIRELY in the original 4 iris dims; the `m` padding dims are
ground-truth-uninformative. Sweeping `m` isolates PURE dimensionality from dataset structure.

Determinism guard (the whole point of the control): the noise columns are drawn from a SEPARATE
rng (`pad_rng`), so the injection rng that selects deleted rows is UNTOUCHED by `m`. => across all
`m`, the deleted rows and the 4 real features are byte-identical; only the noise columns are added.
The ONLY thing that changes with `m` is the added noise. Cleanest isolation of dimensionality.

Model+attack: overfit single tree (max_depth=None) + white-box DecisionTreeAttack — deterministic,
no hangs (the trustworthy combo). Metric: raw OPTICS within-cluster mean pairwise spread (=the
fragile §8 metric) in the FULL padded space; prediction = its spatial-vs-random gap decays with `m`.

--save-clouds persists each cell's adversarial cloud (full padded space) so Arm 3 reduction metrics
(run_reduction.py) run offline on the EXACT clouds behind this parquet (same-clouds guarantee, §8 pattern).

-> results_dimensionality.parquet  (one row per m x structure x frac x seed, 5-fold mean)
Reuses inject_delete (./run_imbalance.py) and cluster_stats (../outlier_experiment/run_experiment.py).
Usage: python run_dimensionality.py [--smoke] [--save-clouds] [--nseeds N]
"""
import os, sys
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_v] = '1'
import warnings, time, importlib.util
from pathlib import Path
from itertools import product
import numpy as np, polars as pl
warnings.filterwarnings('ignore')
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import DecisionTreeAttack

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results_dimensionality.parquet'
SAVE_CLOUDS = '--save-clouds' in sys.argv
CLOUDS = HERE / 'clouds_dim'

_spec = importlib.util.spec_from_file_location('imb_lib', HERE / 'run_imbalance.py')
_imb = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_imb)
inject_delete = _imb.inject_delete
_os = importlib.util.spec_from_file_location('ol', HERE.parent / 'outlier_experiment' / 'run_experiment.py')
_o = importlib.util.module_from_spec(_os); _os.loader.exec_module(_o)
cluster_stats = _o.cluster_stats

# --- controlled knobs
M_PAD = [0, 5, 10, 20, 50, 100]                  # number of appended pure-noise N(0,1) dimensions
FRAC = [0.0, 0.25, 0.5, 0.7, 0.85, 0.95]         # fraction of class tc removed (severity)
STRUCT = ['random', 'spatial']                   # imbalance vs coverage-gap; the key factor
TC = 2                                           # virginica (contested class) — Phase-0 imbalance config
FEAT = 2                                         # petal length (spatial arm sorts by this)
N_FOLDS = 5
SEEDS = [42, 58, 125, 7, 13, 21, 99, 123, 200, 777] + list(range(300, 320))   # 30 seeds (tight CIs)
COLS = ['dataset', 'model', 'attack', 'm', 'structure', 'tc', 'feat', 'frac', 'seed',
        'tacc', 'vacc', 'min_recall', 'nadv', 'mean_dist']


def pad_noise(X, m, seed):
    """Append m standardized-N(0,1) noise columns. Separate rng so the injection rng is untouched
    by m (byte-identical deleted rows + real features across all m)."""
    if m <= 0:
        return X
    pad_rng = np.random.default_rng(seed * 1_000 + 7)
    return np.hstack([X, pad_rng.standard_normal((X.shape[0], m))])


def run_cell(m, structure, frac, seed):
    X, y = load_iris(return_X_y=True)
    X = StandardScaler().fit_transform(X)            # standardize the 4 real features to unit scale
    X = pad_noise(X, m, seed)                         # then append m unit-scale noise dims
    yoh = OneHotEncoder(sparse_output=False).fit(y.reshape(-1, 1))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed)                # SAME sequence as Phase-0 imbalance => m=0 matches
    folds, clouds = [], []
    for tr, te in skf.split(X, y):
        Xt, Xv, yt, yv = X[tr], X[te], y[tr], y[te]
        Xa, ya = inject_delete(Xt, yt, TC, FEAT, frac, structure, rng)   # deletes ROWS; feat<4 => real signal
        clf = DecisionTreeClassifier(max_depth=None, random_state=42).fit(Xa, ya)
        art = SklearnClassifier(clf); art.fit(Xa, yoh.transform(ya.reshape(-1, 1)))   # onehot-refit (proven)
        tacc = float((np.argmax(art.predict(Xa), axis=1) == ya).mean())
        pv = np.argmax(art.predict(Xv), axis=1)
        vacc = float((pv == yv).mean())
        mmask = yv == TC
        min_recall = float((pv[mmask] == TC).mean()) if mmask.sum() else np.nan
        c = pv == yv
        if c.sum():
            adv = DecisionTreeAttack(classifier=art).generate(Xv[c])
            ap = np.argmax(art.predict(adv), axis=1)
            flip = ap != yv[c]
            adv, cls = adv[flip], yv[c][flip]
        else:
            adv = np.empty((0, X.shape[1])); cls = np.empty(0, dtype=y.dtype)
        if SAVE_CLOUDS:
            clouds.append((adv, cls))
        md = cluster_stats(adv)[2] if len(adv) else np.nan
        folds.append(dict(tacc=tacc, vacc=vacc, min_recall=min_recall, nadv=len(adv), mean_dist=md))
    if SAVE_CLOUDS:
        d = CLOUDS / f'm{m}' / structure
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / f'frac{frac:.2f}_seed{seed}.npz',
                 **{f'adv{i}': a for i, (a, _) in enumerate(clouds)},
                 **{f'cls{i}': c for i, (_, c) in enumerate(clouds)})
    r = {k: float(np.nanmean([f[k] for f in folds])) for k in folds[0]}
    r.update(dataset='iris_padded', model='tree', attack='dta', m=int(m),
             structure=structure, tc=TC, feat=FEAT, frac=float(frac), seed=int(seed))
    return r


def main():
    global SEEDS
    if '--smoke' in sys.argv:
        for m in (0, 100):
            for st in STRUCT:
                t0 = time.time(); r = run_cell(m, st, 0.85, 42)
                print(f'  smoke m={m:<3} {st:<7}: {time.time()-t0:.2f}s  vacc={r["vacc"]:.3f} '
                      f'recall={r["min_recall"]:.3f} spread={r["mean_dist"]:.3f} nadv={r["nadv"]:.1f}',
                      flush=True)
        return
    if '--nseeds' in sys.argv:
        SEEDS = SEEDS[:int(sys.argv[sys.argv.index('--nseeds') + 1])]
    # grid: frac=0 baseline is shared (structure-independent) -> keep only 'random'
    grid = []
    for m, seed in product(M_PAD, SEEDS):
        grid.append((m, 'random', 0.0, seed))
        for st, fr in product(STRUCT, FRAC[1:]):
            grid.append((m, st, fr, seed))
    rows, t0 = [], time.time()
    for i, (m, st, fr, sd) in enumerate(grid, 1):
        rows.append(run_cell(m, st, fr, sd))
        if i % 100 == 0 or i == len(grid):
            pl.DataFrame([{c: r.get(c, np.nan) for c in COLS} for r in rows]).write_parquet(OUT)
            print(f'  [{i}/{len(grid)}] {(time.time()-t0)/60:.1f}m (checkpointed {len(rows)} rows)', flush=True)
    print(f'DONE: {len(rows)} rows -> {OUT.name} in {(time.time()-t0)/60:.1f}m', flush=True)


if __name__ == '__main__':
    main()
