"""Diagnostic sweep: for which (target class, direction) does the OVERFIT TREE
actually react to outliers? Leaf-count change = clean mechanistic probe; if leaves
don't move, the tree didn't react at all. Averaged over 5 folds, k=8, n_out=5."""
import os
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_v] = '1'
import warnings; warnings.filterwarnings('ignore')
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from art.estimators.classification import SklearnClassifier
from run_experiment import inject_outliers, attack_adv, cluster_stats, N_FOLDS

X, y = load_iris(return_X_y=True)
yoh = OneHotEncoder(sparse_output=False).fit(y.reshape(-1, 1))
K, N_OUT = 8, 5


def build(Xa, ya, Xv, yv):
    m = DecisionTreeClassifier(max_depth=None, random_state=42).fit(Xa, ya)
    art = SklearnClassifier(m); art.fit(Xa, yoh.transform(ya.reshape(-1, 1)))
    adv, succ, labels = attack_adv(art, Xv, yv)
    return m.get_n_leaves(), adv, labels


print(f"{'tc':>2} {'dir':>8} | {'dleaf':>6} | {'d_spread_all':>12} | {'d_spread_tc':>11}  "
      f"(k={K}, n_out={N_OUT}, overfit tree + DTA, mean over 5 folds)")
for tc in [0, 1, 2]:
    for direction in ['outward', 'toward']:
        dleaf, dspread, dspread_tc = [], [], []
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        for tr, te in skf.split(X, y):
            Xt, Xv, yt, yv = X[tr], X[te], y[tr], y[te]
            rng = np.random.default_rng(0)
            Xa, ya = inject_outliers(Xt, yt, tc, K, N_OUT, direction=direction, rng=rng)
            lc, adv0, lab0 = build(Xt, yt, Xv, yv)
            li, adv1, lab1 = build(Xa, ya, Xv, yv)
            s0, s1 = cluster_stats(adv0)[2], cluster_stats(adv1)[2]
            s0c, s1c = cluster_stats(adv0[lab0 == tc])[2], cluster_stats(adv1[lab1 == tc])[2]
            dleaf.append(li - lc)
            dspread.append((s1 - s0) if np.isfinite(s1 * s0) else np.nan)
            dspread_tc.append((s1c - s0c) if np.isfinite(s1c * s0c) else np.nan)
        print(f"{tc:>2} {direction:>8} | {np.nanmean(dleaf):>+6.1f} | "
              f"{np.nanmean(dspread):>+12.3f} | {np.nanmean(dspread_tc):>+11.3f}")
