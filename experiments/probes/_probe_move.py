"""Probe: does label noise make adversarial points 'barely move' so their
geometry just reflects the original test-point geometry?

Hypothesis (user): more noise -> more tree leaves/boundaries -> tiny perturbation
to flip -> adv points ~ original test points -> adv spread -> original spread.
"""
import os
for v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[v]='1'
import numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial.distance import pdist
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import DecisionTreeAttack

iris = load_iris(); X, y = iris.data, iris.target
NOISE = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
SPLIT_SEEDS = [42, 58, 125]
NOISE_SEEDS = list(range(6))

def flip(yt, nl, rng):
    ym = yt.copy(); nf = int(len(ym)*nl); cs = np.unique(ym)
    for i in rng.choice(len(ym), nf, replace=False):
        ym[i] = rng.choice([c for c in cs if c != ym[i]])
    return ym

for DEPTH in [3, 10]:
    print(f'\n===== TREE max_depth={DEPTH}, DecisionTreeAttack, iris label noise =====')
    print(f'{"noise":>6} {"leaves":>7} {"perturb":>8} {"adv_spread":>11} {"orig_spread":>12} {"ratio":>6} {"test_acc":>9}')
    for nl in NOISE:
        leaves, perturbs, adv_sp, orig_sp, accs = [], [], [], [], []
        for ss in SPLIT_SEEDS:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=ss)
            for ns in NOISE_SEEDS:
                rng = np.random.default_rng(ns*1000+ss)
                for tr, te in skf.split(X, y):
                    Xt, Xv, yt, yv = X[tr], X[te], y[tr], y[te]
                    ytn = flip(yt, nl, rng)
                    m = DecisionTreeClassifier(max_depth=DEPTH, random_state=42).fit(Xt, ytn)
                    yoh = OneHotEncoder(sparse_output=False).fit(y.reshape(-1,1)).transform(ytn.reshape(-1,1))
                    art = SklearnClassifier(m); art.fit(Xt, yoh)
                    p = np.argmax(art.predict(Xv), axis=1); c = p == yv
                    if c.sum() < 4: continue
                    Xc = Xv[c]
                    adv = DecisionTreeAttack(classifier=art, verbose=False).generate(Xc)
                    ap = np.argmax(art.predict(adv), axis=1)
                    ok = ap != yv[c]
                    if ok.sum() < 4: continue
                    adv_ok = adv[ok]; orig_ok = Xc[ok]
                    leaves.append(m.get_n_leaves())
                    perturbs.append(np.linalg.norm(adv_ok - orig_ok, axis=1).mean())
                    adv_sp.append(pdist(adv_ok).mean())
                    orig_sp.append(pdist(orig_ok).mean())
                    accs.append((yv == p).mean())
        if leaves:
            r = np.mean(adv_sp)/np.mean(orig_sp)
            print(f'{nl:>6.1f} {np.mean(leaves):>7.1f} {np.mean(perturbs):>8.4f} '
                  f'{np.mean(adv_sp):>11.4f} {np.mean(orig_sp):>12.4f} {r:>6.3f} {np.mean(accs):>9.3f}')
