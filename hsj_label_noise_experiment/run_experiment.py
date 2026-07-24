"""
HSJ label-noise experiment: Tree vs SVM (iris).

Parallel to hsj_svm_experiment but the defect is LABEL NOISE (flip a fraction of
TRAINING labels to a different class; test labels stay clean) instead of a coverage gap.

Same everything else: HopSkipJump (L2, max_iter=10/max_eval=200/init_eval=50),
within-cluster density = n_points / (mean pairwise dist + 1) per OPTICS cluster.

Same bulletproof driver: each row is a fresh subprocess with a hard timeout, so the
HopSkipJump-on-tree non-convergence hangs are killed and skipped (NaN placeholder).

Grid: 5 noise x 3 split-seeds x 12 noise-seeds x 2 models = 360 runs (n=36 per model/noise cell).

Usage (activated venv terminal):  python run_experiment.py
"""
import os
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_v] = '1'

import sys, time, subprocess
from pathlib import Path
from itertools import product

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results.parquet'
PROGRESS = HERE / 'progress.txt'

NOISE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # extended >0.5 to show the randomness regime
SPLIT_SEEDS  = [42, 58, 125]
NOISE_SEEDS  = list(range(12))       # 0..11  -> 3 x 12 = 36 realizations per cell
MODEL_NAMES  = ['tree', 'svm']
N_FOLDS      = 5
TIMEOUT      = 15                    # s per row before declared hung and skipped

COLS = ['tacc', 'vacc', 'asucc', 'nadv', 'density', 'nclust', 'mean_dist', 'clust_size', 'aiden_density',
        'noise', 'split_seed', 'noise_seed', 'model']


def build_grid():
    return list(product(NOISE_LEVELS, SPLIT_SEEDS, NOISE_SEEDS, MODEL_NAMES))


def done_keys():
    if not OUT.exists():
        return set()
    d = pl.read_parquet(OUT)
    return set(zip(d['noise'].round(2).to_list(), d['split_seed'].to_list(),
                   d['noise_seed'].to_list(), d['model'].to_list()))


def next_undone():
    dk = done_keys()
    for (nz, ss, ns, nm) in build_grid():
        if (round(nz, 2), ss, ns, nm) not in dk:
            return (nz, ss, ns, nm)
    return None


def append_row(r):
    for c in COLS:
        r.setdefault(c, np.nan)
    new = pl.DataFrame([{k: r[k] for k in COLS}])
    merged = pl.concat([pl.read_parquet(OUT), new]) if OUT.exists() else new
    merged.write_parquet(OUT)


def flip_labels(y_train, noise_level, rng):
    """Flip `noise_level` fraction of labels to a different class (exact port of defects.py)."""
    y_mod = y_train.copy()
    n_flip = int(len(y_mod) * noise_level)
    classes = np.unique(y_mod)
    flip_idx = rng.choice(len(y_mod), size=n_flip, replace=False)
    for idx in flip_idx:
        current = y_mod[idx]
        other = [c for c in classes if c != current]
        y_mod[idx] = rng.choice(other)
    return y_mod


# ─────────────────────────── WORKER (one row) ───────────────────────────
def worker():
    import warnings
    warnings.filterwarnings('ignore')
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.cluster import OPTICS
    from scipy.spatial.distance import pdist
    from art.estimators.classification import SklearnClassifier
    from art.attacks.evasion import HopSkipJump

    job = next_undone()
    if job is None:
        print('worker: nothing to do', flush=True)
        return
    noise, split_seed, noise_seed, nm = job

    iris = load_iris(); X, y = iris.data, iris.target
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=split_seed)
    noise_rng = np.random.default_rng(noise_seed * 1000 + split_seed)

    def make_model():
        return (DecisionTreeClassifier(max_depth=3, random_state=42) if nm == 'tree'
                else SVC(kernel='rbf', probability=True, random_state=42))

    folds = []
    for tr, te in skf.split(X, y):
        Xt, Xv, yt, yv = X[tr], X[te], y[tr], y[te]
        yt_noisy = flip_labels(yt, noise, noise_rng)          # corrupt TRAIN labels only
        yoh = OneHotEncoder(sparse_output=False).fit(y.reshape(-1, 1)).transform(yt_noisy.reshape(-1, 1))
        m = make_model().fit(Xt, yt_noisy)
        art = SklearnClassifier(m); art.fit(Xt, yoh)
        p = np.argmax(art.predict(Xv), axis=1); c = p == yv     # correct vs CLEAN test labels
        if c.sum() > 0:
            hs = HopSkipJump(classifier=art, norm=2, max_iter=10, max_eval=200,
                             init_eval=50, verbose=False)
            adv = hs.generate(Xv[c])
            ap = np.argmax(art.predict(adv), axis=1)
            adv = adv[ap != yv[c]]
        else:
            adv = np.array([])
        d, nc, md, cs, ad = np.nan, np.nan, np.nan, np.nan, np.nan
        if len(adv) >= 4:
            o = OPTICS(min_samples=3, xi=0.05, min_cluster_size=3).fit(adv)
            dens_fixed, dists_fixed, sizes, dens_aiden = [], [], [], []
            for cl in set(o.labels_) - {-1}:
                cp = adv[o.labels_ == cl]
                if len(cp) < 2: continue
                dd = pdist(cp).mean()
                dens_fixed.append(len(cp) / (dd + 1)); dists_fixed.append(dd); sizes.append(len(cp))
                buggy = [np.linalg.norm((cp[i], cp[j])) for i in range(len(cp)) for j in range(i+1, len(cp))]
                dens_aiden.append(len(cp) / (np.mean(buggy) + 1))
            if dens_fixed:
                d, nc, md, cs, ad = float(np.mean(dens_fixed)), len(dens_fixed), float(np.mean(dists_fixed)), float(np.mean(sizes)), float(np.mean(dens_aiden))
            else:
                nc = 0
        folds.append({'tacc': m.score(Xt, yt_noisy), 'vacc': m.score(Xv, yv),
                      'asucc': float(c.mean()), 'nadv': len(adv), 'density': d, 'nclust': nc,
                      'mean_dist': md, 'clust_size': cs, 'aiden_density': ad})

    r = {k: float(np.nanmean([f[k] for f in folds])) for k in folds[0]}
    r.update(noise=round(noise, 2), split_seed=split_seed, noise_seed=noise_seed, model=nm)
    append_row(r)
    print(f'worker: done {nm} noise={noise} split={split_seed} nseed={noise_seed}', flush=True)


# ─────────────────────────── DRIVER ───────────────────────────
def driver():
    total = len(build_grid())
    t0 = time.time()
    print(f'Driver: {total} rows, fresh subprocess per row, {TIMEOUT}s timeout\n', flush=True)

    while True:
        have = len(done_keys())
        if have >= total:
            break
        elapsed = time.time() - t0
        pct = have / total
        eta = (elapsed / pct - elapsed) if pct > 0 else 0
        line = f'[{have}/{total}] {pct:.0%}  elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m'
        print(line, flush=True)
        PROGRESS.write_text(line + f'\ntotal_rows={have}/{total}\n')

        job = next_undone()
        p = subprocess.Popen([sys.executable, __file__, '--worker'], cwd=str(HERE),
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            out, err = p.communicate(timeout=TIMEOUT)
            if p.returncode != 0:
                print('  worker error:', (err or '')[-400:], flush=True)
        except subprocess.TimeoutExpired:
            p.kill(); p.communicate()
            if job is not None:
                nz, ss, ns, nm = job
                append_row({'noise': round(nz, 2), 'split_seed': ss, 'noise_seed': ns, 'model': nm})
                print(f'  >>> HUNG, skipped: {nm} noise={nz} split={ss} nseed={ns}', flush=True)

    elapsed = time.time() - t0
    n_skip = int(pl.read_parquet(OUT)['density'].is_nan().sum())
    final = f'DONE: {total} rows in {elapsed/60:.1f} min  ({n_skip} skipped as hung)'
    print('\n' + final, flush=True)
    PROGRESS.write_text(final + f'\ntotal_rows={total}/{total}\n')


if __name__ == '__main__':
    worker() if '--worker' in sys.argv else driver()
