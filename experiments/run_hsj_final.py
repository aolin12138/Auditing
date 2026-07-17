"""
Final HSJ experiment: 3 bias levels, iris, tree vs SVM.
Appends to data/hopskipjump_bias.parquet
"""
import numpy as np, polars as pl, time, sys
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import OPTICS
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import HopSkipJump
from itertools import product
import warnings; warnings.filterwarnings('ignore')

BIAS = [0.1, 0.5, 0.9]; SEEDS = [42, 58, 125]
FEAT = [0, 1, 2, 3]; CLASSES = [0, 1, 2]
OUT = "data/hopskipjump_bias.parquet"

def inject_bias(X, y, tc, fi, bias):
    if bias == 0: return X.copy(), y.copy()
    mask = y == tc; order = np.argsort(X[mask][:, fi])
    Xs, ys = X[mask][order], y[mask][order]
    nk = max(int(len(Xs) * (1 - bias)), 3)
    return np.vstack([X[~mask], Xs[-nk:]]), np.hstack([y[~mask], ys[-nk:]])

def opt_density(pts, ms=3):
    if len(pts) < ms + 1: return np.nan, np.nan
    o = OPTICS(min_samples=ms, xi=0.05, min_cluster_size=ms).fit(pts)
    ds = [1/np.mean(o.core_distances_[o.labels_==c]) for c in set(o.labels_)-{-1}
          if len(o.core_distances_[o.labels_==c])>0 and np.mean(o.core_distances_[o.labels_==c])>0]
    return (np.mean(ds), len(ds)) if ds else (np.nan, 0)

def gen_adv(art, X, y):
    p = np.argmax(art.predict(X), axis=1); c = p == y
    Xc, yc = X[c], y[c]
    if len(Xc) == 0: return np.array([]), 0.0
    hs = HopSkipJump(classifier=art, norm=2, max_iter=10, max_eval=200, init_eval=50, verbose=False)
    adv = hs.generate(Xc); ap = np.argmax(art.predict(adv), axis=1)
    return adv[ap != yc], (ap != yc).mean()

def run_one(X, y, m_sk, tc, fi, bias, seed):
    np.random.seed(seed)
    Xb, yb = inject_bias(X, y, tc, fi, bias)
    ohe = OneHotEncoder(sparse_output=False)
    yoh = ohe.fit_transform(yb.reshape(-1, 1))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = []
    for tr, te in skf.split(Xb, yb):
        Xt, Xv, yt, yv = Xb[tr], Xb[te], yb[tr], yb[te]
        m = m_sk.__class__(**m_sk.get_params()).fit(Xt, yt)
        art = SklearnClassifier(m); art.fit(Xt, yoh[tr])
        adv, succ = gen_adv(art, Xv, yv)
        d, nc = opt_density(adv)
        folds.append({'tacc': m.score(Xt, yt), 'vacc': m.score(Xv, yv),
                       'asucc': succ, 'nadv': len(adv), 'density': d, 'nclust': nc})
    r = {k: np.nanmean([f[k] for f in folds]) for k in folds[0]}
    r.update(seed=seed, bias=bias, tc=tc, feat=fi)
    return r

# --- Main ---
iris = load_iris()
models = {'tree': DecisionTreeClassifier(max_depth=3, random_state=42),
           'svm':  SVC(kernel='rbf', probability=True, random_state=42)}
grid = list(product(BIAS, SEEDS, CLASSES, FEAT, models.items()))
TOTAL = len(grid)

# Load existing
if Path(OUT).exists():
    existing = pl.read_parquet(OUT)
    existing_keys = set(zip(existing['bias'], existing['seed'], existing['tc'],
                             existing['feat'], existing['model']))
else:
    existing_keys = set()

# Find what needs running
todo = [(i, b, s, c, f, nm, m) for i, (b, s, c, f, (nm, m)) in enumerate(grid)
        if (b, s, c, f, nm) not in existing_keys]

print(f"Grid: {TOTAL} total, {len(todo)} to run, ~{len(todo)*3:.0f}s ETA", flush=True)

t0 = time.time(); rows = []
for j, (i, b, s, c, f, nm, m) in enumerate(todo):
    try:
        r = run_one(iris.data, iris.target, m, c, f, b, s)
        r['model'] = nm; rows.append(r)
    except Exception as e:
        print(f"  ERROR {i}: {e}", flush=True); continue
    if (j+1) % 30 == 0 or j+1 == len(todo):
        elapsed = time.time() - t0
        pct = (j+1) / len(todo)
        print(f"[{j+1}/{len(todo)}] {pct:.0%} {elapsed:.0f}s ETA={elapsed/pct-elapsed:.0f}s", flush=True)
        if rows:
            new = pl.DataFrame(rows)
            all_data = pl.concat([existing, new]) if Path(OUT).exists() else new
            all_data.write_parquet(OUT)

# Final save
if rows:
    new = pl.DataFrame(rows)
    all_data = pl.concat([existing, new]) if Path(OUT).exists() else new
    all_data.write_parquet(OUT)
    print(f"DONE: {all_data.shape[0]} rows in {time.time()-t0:.0f}s", flush=True)
