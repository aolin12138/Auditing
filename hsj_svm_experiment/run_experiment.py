"""
Bulletproof HSJ bias experiment runner.

Root cause of freezes: certain specific CV folds make HopSkipJump's internal
binary search loop forever (e.g. grid row bias=0.3/seed=42/class=2/feat=2/tree,
fold 2). A normal row takes ~4.5s; a pathological one never returns. In a single
long process one bad row freezes everything.

Fix: DRIVER runs each row as its own fresh WORKER subprocess with a hard timeout.
- Worker: does the single next-undone row, appends to results.parquet, exits.
- Driver: if a worker exceeds TIMEOUT, it's killed and that row is written as a
  NaN placeholder (status='skipped') so it's never retried. Everything else flows.
Fully resumable; ART import is only ~1.7s so per-row overhead is tiny.

Usage (from activated venv terminal):
    python run_experiment.py            # run to completion
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

BIAS_LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]
SEEDS       = [42, 58, 125]
CLASSES     = [0, 1, 2]
FEATURES    = [0, 1, 2, 3]
MODEL_NAMES = ['tree', 'svm']
N_FOLDS     = 5
TIMEOUT     = 15          # seconds per row before it's declared hung and skipped

COLS = ['tacc', 'vacc', 'asucc', 'nadv', 'density', 'nclust', 'mean_dist', 'clust_size', 'aiden_density', 'seed', 'bias', 'tc', 'feat', 'model']


def build_grid():
    return list(product(BIAS_LEVELS, SEEDS, CLASSES, FEATURES, MODEL_NAMES))


def done_keys():
    if not OUT.exists():
        return set()
    d = pl.read_parquet(OUT)
    return set(zip(d['bias'].round(2).to_list(), d['seed'].to_list(),
                   d['tc'].to_list(), d['feat'].to_list(), d['model'].to_list()))


def next_undone():
    dk = done_keys()
    for (b, s, c, f, nm) in build_grid():
        if (round(b, 2), s, c, f, nm) not in dk:
            return (b, s, c, f, nm)
    return None


def append_row(r):
    for col in COLS:
        r.setdefault(col, np.nan)
    new = pl.DataFrame([{k: r[k] for k in COLS}])
    merged = pl.concat([pl.read_parquet(OUT), new]) if OUT.exists() else new
    merged.write_parquet(OUT)


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
    bias, seed, tc, feat, nm = job

    iris = load_iris(); X, y = iris.data, iris.target
    np.random.seed(seed)
    mask = y == tc
    order = np.argsort(X[mask][:, feat])
    Xs, ys = X[mask][order], y[mask][order]
    nk = max(int(len(Xs) * (1 - bias)), 3)
    Xb = np.vstack([X[~mask], Xs[-nk:]]); yb = np.hstack([y[~mask], ys[-nk:]])
    yoh = OneHotEncoder(sparse_output=False).fit_transform(yb.reshape(-1, 1))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    def make_model():
        return (DecisionTreeClassifier(max_depth=3, random_state=42) if nm == 'tree'
                else SVC(kernel='rbf', probability=True, random_state=42))

    folds = []
    for tr, te in skf.split(Xb, yb):
        Xt, Xv, yt, yv = Xb[tr], Xb[te], yb[tr], yb[te]
        m = make_model().fit(Xt, yt)
        art = SklearnClassifier(m); art.fit(Xt, yoh[tr])
        p = np.argmax(art.predict(Xv), axis=1); c = p == yv
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
        folds.append({'tacc': m.score(Xt, yt), 'vacc': m.score(Xv, yv),
                      'asucc': float(c.mean()), 'nadv': len(adv), 'density': d, 'nclust': nc,
                      'mean_dist': md, 'clust_size': cs, 'aiden_density': ad})

    r = {k: float(np.nanmean([f[k] for f in folds])) for k in folds[0]}
    r.update(seed=seed, bias=round(bias, 2), tc=tc, feat=feat, model=nm)
    append_row(r)
    print(f'worker: done {nm} bias={bias} seed={seed} tc={tc} feat={feat}', flush=True)


# ─────────────────────────── DRIVER ───────────────────────────
def driver():
    total = len(build_grid())
    t0 = time.time()
    print(f'Driver: {total} rows, one fresh subprocess per row, {TIMEOUT}s timeout\n', flush=True)

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
                print('  worker error:', (err or '')[-500:], flush=True)
        except subprocess.TimeoutExpired:
            p.kill()
            p.communicate()
            if job is not None:
                b, s, c, f, nm = job
                append_row({'seed': s, 'bias': round(b, 2), 'tc': c, 'feat': f, 'model': nm})
                print(f'  >>> HUNG, skipped: {nm} bias={b} seed={s} tc={c} feat={f}', flush=True)

    elapsed = time.time() - t0
    n_skip = int(pl.read_parquet(OUT)['density'].is_nan().sum())
    final = f'DONE: {total} rows in {elapsed/60:.1f} min  ({n_skip} rows skipped as hung)'
    print('\n' + final, flush=True)
    PROGRESS.write_text(final + f'\ntotal_rows={total}/{total}\n')


if __name__ == '__main__':
    worker() if '--worker' in sys.argv else driver()
