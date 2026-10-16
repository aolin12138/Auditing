"""Label noise on the report's body configuration (final-report Fig 3b).

The earlier label-noise grids (dtree_attack_experiment/, hsj_label_noise_experiment/) were already
train-only, but used a depth-3 tree on unstandardised iris with 3 split seeds. Every other body
figure uses the overfit tree (max_depth=None) + white-box DecisionTreeAttack, or SVM + HopSkipJump,
on standardised iris and wine with the same 30 seeds. This script puts label noise on that same
configuration so the defects can be compared like for like.

Defect: a fraction `noise` of the TRAINING labels in each fold is flipped to a uniformly chosen
different class (identical definition to dtree_attack_experiment.flip_labels). Test labels stay
clean. Only test points the model classifies correctly (against clean labels) are attacked; only
successful adversarial examples are kept; spread = mean within-cluster pairwise distance over
OPTICS clusters (cluster_stats from ../outlier_experiment, as in every other grid).

Recorded per cell (5-fold mean): tacc (accuracy on the noisy training labels), tacc_clean
(training accuracy against the true labels), vacc (clean test accuracy), worst_recall (lowest
per-class test recall), nadv, mean_dist (spread).

-> results_label_noise.parquet (tree+DTA) / results_label_noise_svm.parquet (svm+HSJ)
Usage: python run_label_noise.py [--model svm] [--smoke] [--nseeds N]
Note: HopSkipJump is not reproducible run-to-run (ART 1.20.1 hop_skip_jump.py:295 draws its
initial point from an unseeded np.random.RandomState()); tree+DTA is fully deterministic.
DecisionTreeAttack is run one point at a time: under heavy noise ART 1.20.1 can raise IndexError
for a single point (its upward search overwrites earlier hits and then runs past the root; see
analysis/verify_dta_destination.py). That point gets no adversarial example -- an attack failure,
like any other -- and is counted in `attack_err`. A batched call would lose the whole fold.
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
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import DecisionTreeAttack, HopSkipJump

HERE = Path(__file__).resolve().parent
MODEL = 'svm' if '--model' in sys.argv and sys.argv[sys.argv.index('--model') + 1] == 'svm' else 'tree'
ATTACK = 'hsj' if MODEL == 'svm' else 'dta'
OUT = HERE / ('results_label_noise_svm.parquet' if MODEL == 'svm' else 'results_label_noise.parquet')
_os = importlib.util.spec_from_file_location('ol', HERE.parent / 'outlier_experiment' / 'run_experiment.py')
_o = importlib.util.module_from_spec(_os); _os.loader.exec_module(_o)
cluster_stats = _o.cluster_stats

DATASETS = {'iris': load_iris, 'wine': load_wine}
NOISE = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]   # same levels as the earlier grids, plus 0
N_FOLDS = 5
SEEDS = [42, 58, 125, 7, 13, 21, 99, 123, 200, 777] + list(range(300, 320))   # the body's 30 seeds
COLS = ['dataset', 'model', 'attack', 'noise', 'seed',
        'tacc', 'tacc_clean', 'vacc', 'worst_recall', 'nadv', 'attack_err', 'mean_dist']


def flip_labels(y_train, noise_level, rng):
    """Flip `noise_level` of the labels to a uniformly chosen different class.
    Same definition as dtree_attack_experiment/run_experiment.py:flip_labels."""
    y_mod = y_train.copy(); n_flip = int(len(y_mod) * noise_level)
    classes = np.unique(y_mod)
    for idx in rng.choice(len(y_mod), size=n_flip, replace=False):
        cur = y_mod[idx]; y_mod[idx] = rng.choice([c for c in classes if c != cur])
    return y_mod


def _fit(Xt, yt, yoh):
    if MODEL == 'svm':
        return SklearnClassifier(SVC(kernel='rbf', probability=True, random_state=42).fit(Xt, yt))
    m = DecisionTreeClassifier(max_depth=None, random_state=42).fit(Xt, yt)
    art = SklearnClassifier(m); art.fit(Xt, yoh.transform(yt.reshape(-1, 1)))   # onehot-refit, as run_variance
    return art


def _attack(art, Xc):
    """Returns (adversarial candidates, number of points ART could not attack)."""
    if MODEL == 'svm':
        return HopSkipJump(classifier=art, norm=2, max_iter=10, max_eval=200, init_eval=50,
                           verbose=False).generate(Xc), 0
    atk, out, err = DecisionTreeAttack(classifier=art), Xc.copy(), 0
    for i in range(len(Xc)):
        try:
            out[i] = atk.generate(Xc[i:i + 1])[0]
        except IndexError:      # ART runs past the root for this point; it stays unperturbed -> filtered out
            err += 1
    return out, err


def run_cell(dataset, noise, seed):
    X, y = DATASETS[dataset](return_X_y=True)
    X = StandardScaler().fit_transform(X)
    yoh = OneHotEncoder(sparse_output=False).fit(y.reshape(-1, 1))
    rng = np.random.default_rng(seed)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    folds = []
    for tr, te in skf.split(X, y):
        Xt, Xv, yt, yv = X[tr], X[te], y[tr], y[te]
        ytn = flip_labels(yt, noise, rng) if noise > 0 else yt
        art = _fit(Xt, ytn, yoh)
        pt = np.argmax(art.predict(Xt), axis=1)
        pv = np.argmax(art.predict(Xv), axis=1)
        recalls = [float((pv[yv == k] == k).mean()) for k in np.unique(yv)]
        c = pv == yv
        err = 0
        if c.sum():
            adv, err = _attack(art, Xv[c])
            adv = adv[np.argmax(art.predict(adv), axis=1) != yv[c]]
        else:
            adv = np.empty((0, X.shape[1]))
        md = cluster_stats(adv)[2] if len(adv) else np.nan
        folds.append(dict(tacc=float((pt == ytn).mean()), tacc_clean=float((pt == yt).mean()),
                          vacc=float(c.mean()), worst_recall=min(recalls), nadv=len(adv), attack_err=err,
                          mean_dist=md))
    r = {k: float(np.nanmean([f[k] for f in folds])) for k in folds[0]}
    r.update(dataset=dataset, model=MODEL, attack=ATTACK, noise=float(noise), seed=int(seed))
    return r


def main():
    global SEEDS
    if '--smoke' in sys.argv:
        for ds, nz in product(DATASETS, [0.0, 0.3]):
            t0 = time.time(); r = run_cell(ds, nz, 42)
            print(f'  smoke {MODEL}+{ATTACK} {ds} noise={nz}: {time.time()-t0:.1f}s tacc={r["tacc"]:.3f} '
                  f'tacc_clean={r["tacc_clean"]:.3f} vacc={r["vacc"]:.3f} worst_recall={r["worst_recall"]:.3f} '
                  f'nadv={r["nadv"]:.1f} spread={r["mean_dist"]:.3f}', flush=True)
        return
    if '--nseeds' in sys.argv:
        SEEDS = SEEDS[:int(sys.argv[sys.argv.index('--nseeds') + 1])]
    grid = list(product(DATASETS, NOISE, SEEDS))
    rows, t0 = [], time.time()
    for i, (ds, nz, sd) in enumerate(grid, 1):
        rows.append(run_cell(ds, nz, sd))
        if i % 50 == 0 or i == len(grid):
            pl.DataFrame([{c: r.get(c, np.nan) for c in COLS} for r in rows]).write_parquet(OUT)  # checkpoint
            print(f'  [{i}/{len(grid)}] {(time.time()-t0)/60:.1f}m (checkpointed {len(rows)} rows)', flush=True)
    print(f'DONE: {len(rows)} rows -> {OUT.name} in {(time.time()-t0)/60:.1f}m', flush=True)


if __name__ == '__main__':
    main()
