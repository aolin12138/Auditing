"""
Background experiment: HopSkipJump attack on Tree vs SVM (iris, coverage gap bias).
Saves progress to data/hopskipjump_bias.parquet every 60 runs.
"""
import numpy as np, polars as pl, time, warnings, sys, os
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
warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────
BIAS_LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]
SEEDS       = [42, 58, 125]
FEATURES    = [0, 1, 2, 3]
OUTPUT      = "data/hopskipjump_bias.parquet"
# ────────────────────────────────────────────────────────────

# Functions
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
    densities = []
    for c in set(o.labels_) - {-1}:
        cd = o.core_distances_[o.labels_ == c]
        if len(cd) > 0 and np.mean(cd) > 0:
            densities.append(1.0 / np.mean(cd))
    return (np.mean(densities), len(densities)) if densities else (np.nan, 0)

def gen_adv(art_model, X, y):
    preds = np.argmax(art_model.predict(X), axis=1)
    correct = preds == y
    Xc, yc = X[correct], y[correct]
    if len(Xc) == 0: return np.array([]), 0.0
    hs = HopSkipJump(classifier=art_model, norm=2, max_iter=10, max_eval=200,
                     init_eval=50, verbose=False)
    adv = hs.generate(Xc)
    ap = np.argmax(art_model.predict(adv), axis=1)
    success = ap != yc
    return adv[success], success.mean()

def run_one(X, y, model_sklearn, target_class, feature_idx, bias, seed, n_folds=5):
    np.random.seed(seed)
    Xb, yb = inject_bias(X, y, target_class, feature_idx, bias)
    ohe = OneHotEncoder(sparse_output=False)
    yb_oh = ohe.fit_transform(yb.reshape(-1, 1))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for tr, te in skf.split(Xb, yb):
        Xt, Xv, yt, yv = Xb[tr], Xb[te], yb[tr], yb[te]
        m = model_sklearn.__class__(**model_sklearn.get_params()).fit(Xt, yt)
        art = SklearnClassifier(m)
        art.fit(Xt, yb_oh[tr])
        adv, succ = gen_adv(art, Xv, yv)
        d, nc = opt_density(adv)
        folds.append({'tacc': m.score(Xt,yt), 'vacc': m.score(Xv,yv),
                       'asucc': succ, 'nadv': len(adv), 'density': d, 'nclust': nc})
    r = {k: np.nanmean([f[k] for f in folds]) for k in folds[0]}
    r.update(seed=seed, bias=bias, tc=target_class, feat=feature_idx)
    return r

# ── Main ────────────────────────────────────────────────────
iris = load_iris()
models = {
    'tree': DecisionTreeClassifier(max_depth=3, random_state=42),
    'svm':  SVC(kernel='rbf', probability=True, random_state=42),
}
CLASSES = [0, 1, 2]
TOTAL = len(BIAS_LEVELS) * len(SEEDS) * len(CLASSES) * len(FEATURES) * len(models)

print(f"Grid: {len(BIAS_LEVELS)}b x {len(SEEDS)}s x {len(CLASSES)}c x "
      f"{len(FEATURES)}f x {len(models)}m = {TOTAL} runs", flush=True)
print(f"Output: {OUTPUT}", flush=True)

rows = []
t_start = time.time()
for i, (bias, seed, tc, feat, (nm, m)) in enumerate(product(
    BIAS_LEVELS, SEEDS, CLASSES, FEATURES, models.items()
)):
    r = run_one(iris.data, iris.target, m, tc, feat, bias, seed, 5)
    r['model'] = nm
    rows.append(r)
    
    if (i+1) % 60 == 0 or i+1 == TOTAL:
        elapsed = time.time() - t_start
        pct = (i+1) / TOTAL
        eta = elapsed / pct - elapsed
        print(f"[{i+1}/{TOTAL}] {pct:.0%}  "
              f"elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m", flush=True)
        # Save checkpoint
        pl.DataFrame(rows).write_parquet(OUTPUT)

elapsed = time.time() - t_start
df = pl.DataFrame(rows)
df.write_parquet(OUTPUT)
print(f"\nDone in {elapsed/60:.1f}m. {df.shape[0]} rows saved to {OUTPUT}", flush=True)
