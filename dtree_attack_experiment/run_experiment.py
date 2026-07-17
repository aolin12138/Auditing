"""
DecisionTreeAttack (structural, tree-specific) through TODAY'S exact pipeline.
Fills the missing cube cells: Tree + DecisionTreeAttack for BOTH defects,
measured with the identical within-cluster density metric used for the SVM/HSJ runs.

Metric: within-cluster density = n_points / (mean pairwise dist + 1) per OPTICS cluster.
Model: DecisionTreeClassifier(max_depth=3). Attack: ART DecisionTreeAttack (exact, no hangs).

Two defect grids (tree-only):
  coverage_gap: 5 bias x 3 seeds x 3 classes x 4 features = 180
  label_noise : 5 noise x 3 split-seeds x 12 noise-seeds  = 180
Resumable; checkpoints every 20 rows. Usage: python run_experiment.py
"""
import os
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_v] = '1'

import time, warnings
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
PROGRESS = HERE / 'progress.txt'

BIAS_LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]
NOISE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5]
SEEDS = [42, 58, 125]
CLASSES = [0, 1, 2]
FEATURES = [0, 1, 2, 3]
NOISE_SEEDS = list(range(12))
N_FOLDS = 5
COLS = ['defect', 'level', 'seed', 'noise_seed', 'tc', 'feat', 'tacc', 'vacc', 'asucc', 'nadv', 'density', 'nclust', 'mean_dist', 'clust_size', 'aiden_density']


def inject_bias(X, y, tc, fi, bias):
    if bias == 0: return X.copy(), y.copy()
    mask = y == tc; order = np.argsort(X[mask][:, fi])
    Xs, ys = X[mask][order], y[mask][order]
    nk = max(int(len(Xs) * (1 - bias)), 3)
    return np.vstack([X[~mask], Xs[-nk:]]), np.hstack([y[~mask], ys[-nk:]])


def flip_labels(y_train, noise_level, rng):
    y_mod = y_train.copy(); n_flip = int(len(y_mod) * noise_level)
    classes = np.unique(y_mod)
    for idx in rng.choice(len(y_mod), size=n_flip, replace=False):
        cur = y_mod[idx]; y_mod[idx] = rng.choice([c for c in classes if c != cur])
    return y_mod


def cluster_stats(points, ms=3):
    """Returns (fixed_density, n_clusters, fixed_mean_dist, cluster_size, AIDEN_buggy_density).
    Aiden's bug: np.linalg.norm((p1,p2)) = Frobenius norm of stacked points, not pairwise dist."""
    if len(points) < ms + 1: return np.nan, np.nan, np.nan, np.nan, np.nan
    o = OPTICS(min_samples=ms, xi=0.05, min_cluster_size=ms).fit(points)
    dens_fixed, dists_fixed, sizes, dens_aiden = [], [], [], []
    for c in set(o.labels_) - {-1}:
        cp = points[o.labels_ == c]
        if len(cp) < 2: continue
        dd = pdist(cp).mean()
        dens_fixed.append(len(cp) / (dd + 1)); dists_fixed.append(dd); sizes.append(len(cp))
        buggy = [np.linalg.norm((cp[i], cp[j])) for i in range(len(cp)) for j in range(i+1, len(cp))]
        dens_aiden.append(len(cp) / (np.mean(buggy) + 1))
    if not dens_fixed: return np.nan, 0, np.nan, np.nan, np.nan
    return float(np.mean(dens_fixed)), len(dens_fixed), float(np.mean(dists_fixed)), float(np.mean(sizes)), float(np.mean(dens_aiden))


def attack_adv(art, Xv, yv):
    p = np.argmax(art.predict(Xv), axis=1); c = p == yv
    if c.sum() == 0: return np.array([]), 0.0
    adv = DecisionTreeAttack(classifier=art).generate(Xv[c])
    ap = np.argmax(art.predict(adv), axis=1)
    return adv[ap != yv[c]], float(c.mean())


def run_coverage_gap(X, y, bias, seed, tc, feat):
    np.random.seed(seed)
    Xb, yb = inject_bias(X, y, tc, feat, bias)
    yoh_full = OneHotEncoder(sparse_output=False).fit(y.reshape(-1, 1))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    folds = []
    for tr, te in skf.split(Xb, yb):
        Xt, Xv, yt, yv = Xb[tr], Xb[te], yb[tr], yb[te]
        m = DecisionTreeClassifier(max_depth=3, random_state=42).fit(Xt, yt)
        art = SklearnClassifier(m); art.fit(Xt, yoh_full.transform(yt.reshape(-1, 1)))
        adv, succ = attack_adv(art, Xv, yv)
        d, nc, md, cs, aiden = cluster_stats(adv)
        folds.append({'tacc': m.score(Xt, yt), 'vacc': m.score(Xv, yv), 'asucc': succ,
                      'nadv': len(adv), 'density': d, 'nclust': nc, 'mean_dist': md, 'clust_size': cs, 'aiden_density': aiden})
    r = {k: float(np.nanmean([f[k] for f in folds])) for k in folds[0]}
    r.update(defect='coverage_gap', level=round(bias, 2), seed=seed, noise_seed=-1, tc=tc, feat=feat)
    return r


def run_label_noise(X, y, noise, split_seed, noise_seed):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=split_seed)
    rng = np.random.default_rng(noise_seed * 1000 + split_seed)
    yoh_full = OneHotEncoder(sparse_output=False).fit(y.reshape(-1, 1))
    folds = []
    for tr, te in skf.split(X, y):
        Xt, Xv, yt, yv = X[tr], X[te], y[tr], y[te]
        ytn = flip_labels(yt, noise, rng)
        m = DecisionTreeClassifier(max_depth=3, random_state=42).fit(Xt, ytn)
        art = SklearnClassifier(m); art.fit(Xt, yoh_full.transform(ytn.reshape(-1, 1)))
        adv, succ = attack_adv(art, Xv, yv)
        d, nc, md, cs, aiden = cluster_stats(adv)
        folds.append({'tacc': m.score(Xt, ytn), 'vacc': m.score(Xv, yv), 'asucc': succ,
                      'nadv': len(adv), 'density': d, 'nclust': nc, 'mean_dist': md, 'clust_size': cs, 'aiden_density': aiden})
    r = {k: float(np.nanmean([f[k] for f in folds])) for k in folds[0]}
    r.update(defect='label_noise', level=round(noise, 2), seed=split_seed, noise_seed=noise_seed, tc=-1, feat=-1)
    return r


def done_keys():
    if not OUT.exists(): return set()
    d = pl.read_parquet(OUT)
    return set(zip(d['defect'].to_list(), d['level'].round(2).to_list(), d['seed'].to_list(),
                   d['noise_seed'].to_list(), d['tc'].to_list(), d['feat'].to_list()))


def main():
    iris = load_iris(); X, y = iris.data, iris.target
    jobs = []
    for b, s, c, f in product(BIAS_LEVELS, SEEDS, CLASSES, FEATURES):
        jobs.append(('cg', b, s, c, f))
    for nz, s, ns in product(NOISE_LEVELS, SEEDS, NOISE_SEEDS):
        jobs.append(('ln', nz, s, ns))
    total = len(jobs)

    dk = done_keys()
    rows = pl.read_parquet(OUT).to_dicts() if OUT.exists() else []
    t0 = time.time(); n0 = len(rows)
    for i, job in enumerate(jobs):
        if job[0] == 'cg':
            _, b, s, c, f = job
            if ('coverage_gap', round(b, 2), s, -1, c, f) in dk: continue
            rows.append(run_coverage_gap(X, y, b, s, c, f))
        else:
            _, nz, s, ns = job
            if ('label_noise', round(nz, 2), s, ns, -1, -1) in dk: continue
            rows.append(run_label_noise(X, y, nz, s, ns))
        if len(rows) % 20 == 0 or len(rows) == total:
            el = time.time() - t0; did = len(rows) - n0
            pct = len(rows) / total
            eta = (el / did * (total - len(rows))) if did else 0
            line = f'[{len(rows)}/{total}] {pct:.0%} elapsed={el/60:.1f}m ETA={eta/60:.1f}m'
            print(line, flush=True)
            pl.DataFrame([{k: r[k] for k in COLS} for r in rows]).write_parquet(OUT)
            PROGRESS.write_text(line + '\n')
    pl.DataFrame([{k: r[k] for k in COLS} for r in rows]).write_parquet(OUT)
    print(f'DONE: {len(rows)} rows', flush=True)


if __name__ == '__main__':
    main()
