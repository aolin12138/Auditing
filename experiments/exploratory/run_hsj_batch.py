"""
Batch runner: runs HSJ bias experiment in chunks, saves incrementally.
Run like: python run_hsj_batch.py [start] [count]
"""
import sys, numpy as np, polars as pl, time, os
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

BIAS_LEVELS = [0.1, 0.5, 0.9]
SEEDS       = [42, 58, 125]
FEATURES    = [0, 1, 2, 3]
CLASSES     = [0, 1, 2]
OUTPUT      = "data/hopskipjump_bias.parquet"

def inject_bias(X, y, target_class, feature_idx, bias):
    if bias == 0: return X.copy(), y.copy()
    mask = y == target_class
    order = np.argsort(X[mask][:, feature_idx])
    Xs, ys = X[mask][order], y[mask][order]
    n_keep = max(int(len(Xs) * (1 - bias)), 3)
    return (np.vstack([X[~mask], Xs[-n_keep:]]), np.hstack([y[~mask], ys[-n_keep:]]))

def opt_density(points, min_samples=3):
    if len(points) < min_samples + 1: return np.nan, np.nan
    o = OPTICS(min_samples=min_samples, xi=0.05, min_cluster_size=min_samples).fit(points)
    densities = [1.0/np.mean(o.core_distances_[o.labels_==c]) for c in set(o.labels_)-{-1}
                 if len(o.core_distances_[o.labels_==c])>0 and np.mean(o.core_distances_[o.labels_==c])>0]
    return (np.mean(densities), len(densities)) if densities else (np.nan, 0)

def gen_adv(art_model, X, y):
    preds = np.argmax(art_model.predict(X), axis=1)
    correct = preds == y; Xc, yc = X[correct], y[correct]
    if len(Xc) == 0: return np.array([]), 0.0
    hs = HopSkipJump(classifier=art_model, norm=2, max_iter=10, max_eval=200, init_eval=50, verbose=False)
    adv = hs.generate(Xc)
    ap = np.argmax(art_model.predict(adv), axis=1)
    return adv[ap != yc], (ap != yc).mean()

def run_one(X, y, model_sklearn, target_class, feature_idx, bias, seed):
    np.random.seed(seed)
    Xb, yb = inject_bias(X, y, target_class, feature_idx, bias)
    ohe = OneHotEncoder(sparse_output=False)
    yb_oh = ohe.fit_transform(yb.reshape(-1, 1))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = []
    for tr, te in skf.split(Xb, yb):
        Xt, Xv, yt, yv = Xb[tr], Xb[te], yb[tr], yb[te]
        m = model_sklearn.__class__(**model_sklearn.get_params()).fit(Xt, yt)
        art = SklearnClassifier(m); art.fit(Xt, yb_oh[tr])
        adv, succ = gen_adv(art, Xv, yv)
        d, nc = opt_density(adv)
        folds.append({'tacc': m.score(Xt,yt), 'vacc': m.score(Xv,yv),
                       'asucc': succ, 'nadv': len(adv), 'density': d, 'nclust': nc})
    r = {k: np.nanmean([f[k] for f in folds]) for k in folds[0]}
    r.update(seed=seed, bias=bias, tc=target_class, feat=feature_idx)
    return r

def main():
    iris = load_iris()
    models = {
        'tree': DecisionTreeClassifier(max_depth=3, random_state=42),
        'svm':  SVC(kernel='rbf', probability=True, random_state=42),
    }
    grid = list(product(BIAS_LEVELS, SEEDS, CLASSES, FEATURES, models.items()))
    total = len(grid)
    
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    count = int(sys.argv[2]) if len(sys.argv) > 2 else total
    end = min(start + count, total)
    
    print(f"Batch: [{start}:{end}] of {total}", flush=True)
    t0 = time.time()
    
    rows = []
    for i in range(start, end):
        bias, seed, tc, feat, (nm, m) = grid[i]
        try:
            r = run_one(iris.data, iris.target, m, tc, feat, bias, seed)
            r['model'] = nm
            rows.append(r)
        except Exception as e:
            print(f"  ERROR row {i} ({nm} bias={bias} seed={seed} tc={tc} feat={feat}): {e}", flush=True)
    
    # Save
    if rows:
        df_new = pl.DataFrame(rows)
        if os.path.exists(OUTPUT):
            df_old = pl.read_parquet(OUTPUT)
            df = pl.concat([df_old, df_new])
        else:
            df = df_new
        df.write_parquet(OUTPUT)
        elapsed = time.time() - t0
        print(f"Saved {df.shape[0]} total rows ({len(rows)} new) in {elapsed:.0f}s", flush=True)
    else:
        print("No new rows.", flush=True)

if __name__ == '__main__':
    main()
