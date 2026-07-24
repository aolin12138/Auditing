"""Answers three questions from review.md with evidence:

Q1/Q2 (label noise): Is the adv cloud 'related to the test distribution' even though
      HSJ inits randomly? Why does perturbation DECREASE with more label noise?
      -> pair each adv point to its SOURCE test point, measure per-point perturbation.

Q3 (coverage gap tc=0, bias 0.7 vs 0.9): why does the spread drop only for HSJ?
      -> decompose adv points by SOURCE true class + per-class perturbation, DTA vs HSJ,
         on the SAME tree, plus how many setosa test points even survive to be attacked.
"""
import os
for v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[v]='1'
import numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial.distance import pdist
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import HopSkipJump, DecisionTreeAttack

iris = load_iris(); X, y = iris.data, iris.target
SEEDS = [42, 58, 125]

def inject_bias(X, y, tc, fi, bias):
    mask = y == tc; order = np.argsort(X[mask][:, fi])
    Xs, ys = X[mask][order], y[mask][order]
    nk = max(int(len(Xs)*(1-bias)), 3)
    return np.vstack([X[~mask], Xs[-nk:]]), np.hstack([y[~mask], ys[-nk:]])

def flip_labels(yt, noise, rng):
    y2 = yt.copy(); n = int(len(y2)*noise); cls = np.unique(y2)
    for i in rng.choice(len(y2), n, replace=False):
        y2[i] = rng.choice([c for c in cls if c != y2[i]])
    return y2

# ══════════════════════════════════════════════════════════════════
# Q1/Q2 — LABEL NOISE: adv cloud is TETHERED to test points; perturbation shrinks
# ══════════════════════════════════════════════════════════════════
print('='*70)
print('Q1/Q2  LABEL NOISE (tree + HSJ): is the adv cloud tethered to test pts?')
print('='*70)
print(f'{"noise":>6} {"perturb(per-pt)":>16} {"adv_spread":>11} {"test_spread":>12} {"ratio":>6}')
yoh = OneHotEncoder(sparse_output=False).fit(y.reshape(-1,1))
for noise in [0.1, 0.3, 0.5]:
    P, A, T = [], [], []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        for tr, te in skf.split(X, y):
            Xt, Xv, yt, yv = X[tr], X[te], y[tr], y[te]
            ytn = flip_labels(yt, noise, rng)
            m = DecisionTreeClassifier(max_depth=3, random_state=42).fit(Xt, ytn)
            art = SklearnClassifier(m); art.fit(Xt, yoh.transform(ytn.reshape(-1,1)))
            p = np.argmax(art.predict(Xv), axis=1); c = p == yv
            if c.sum() < 4: continue
            Xc = Xv[c]
            adv = HopSkipJump(classifier=art, norm=2, max_iter=10, max_eval=200,
                              init_eval=50, verbose=False).generate(Xc)
            ap = np.argmax(art.predict(adv), axis=1); ok = ap != yv[c]
            if ok.sum() < 4: continue
            # per-point perturbation: distance from EACH adv point to ITS OWN source test point
            P.append(np.linalg.norm(adv[ok]-Xc[ok], axis=1).mean())
            A.append(pdist(adv[ok]).mean()); T.append(pdist(Xc[ok]).mean())
    print(f'{noise:>6} {np.mean(P):>16.4f} {np.mean(A):>11.4f} {np.mean(T):>12.4f} {np.mean(A)/np.mean(T):>6.3f}')
print('  -> per-point perturbation SHRINKS: each adv point sits closer to ITS OWN test point')
print('  -> so adv cloud = {test points + tiny nudge} -> inherits test spread -> ratio->1')

# ══════════════════════════════════════════════════════════════════
# Q3 — COVERAGE GAP tc=0: decompose adv points by SOURCE true class
# ══════════════════════════════════════════════════════════════════
print()
print('='*70)
print('Q3  COVERAGE GAP, tc=0 (setosa depleted): where do adv points come from?')
print('='*70)
for bias in [0.7, 0.9]:
    print(f'\n--- bias {bias} ---')
    # survival: how many setosa (class 0) TEST points even exist to attack?
    surv_test0, surv_train0 = [], []
    # per source-class perturbation, DTA vs HSJ, on the SAME tree
    agg = {'DTA':{0:[],1:[],2:[]}, 'HSJ':{0:[],1:[],2:[]}}
    spread = {'DTA':[], 'HSJ':[]}
    n_src = {'DTA':{0:0,1:0,2:0}, 'HSJ':{0:0,1:0,2:0}}
    for seed in SEEDS:
        for feat in [0,1,2,3]:
            Xb, yb = inject_bias(X, y, 0, feat, bias)
            skf = StratifiedKFold(5, shuffle=True, random_state=seed)
            for tr, te in skf.split(Xb, yb):
                Xt, Xv, yt, yv = Xb[tr], Xb[te], yb[tr], yb[te]
                surv_test0.append((yv==0).sum()); surv_train0.append((yt==0).sum())
                m = DecisionTreeClassifier(max_depth=3, random_state=42).fit(Xt, yt)
                art = SklearnClassifier(m); art.fit(Xt, yoh.transform(yt.reshape(-1,1)))
                p = np.argmax(art.predict(Xv), axis=1); c = p == yv
                if c.sum() < 4: continue
                Xc, ycls = Xv[c], yv[c]
                # DTA (deterministic) and HSJ (stochastic) on the SAME model/points
                for name, adv in [('DTA', DecisionTreeAttack(classifier=art).generate(Xc)),
                                  ('HSJ', HopSkipJump(classifier=art, norm=2, max_iter=10,
                                       max_eval=200, init_eval=50, verbose=False).generate(Xc))]:
                    ap = np.argmax(art.predict(adv), axis=1); ok = ap != ycls
                    if ok.sum() < 2: continue
                    pert = np.linalg.norm(adv[ok]-Xc[ok], axis=1)
                    spread[name].append(pdist(adv[ok]).mean())
                    for cls in [0,1,2]:
                        sel = ok & (ycls==cls)
                        if sel.sum():
                            agg[name][cls].append(np.linalg.norm(adv[sel]-Xc[sel], axis=1).mean())
                            n_src[name][cls] += int(sel.sum())
    print(f'  setosa points surviving:  train~{np.mean(surv_train0):.1f}   test~{np.mean(surv_test0):.2f}  (of 50 originally, 40 train/10 test)')
    for name in ['DTA','HSJ']:
        tot = sum(n_src[name].values())
        frac = {cls: n_src[name][cls]/tot for cls in [0,1,2]}
        pert = {cls: (np.mean(agg[name][cls]) if agg[name][cls] else float('nan')) for cls in [0,1,2]}
        print(f'  {name}: adv_spread={np.mean(spread[name]):.3f}  |  source-class mix  '
              f'setosa={frac[0]:.0%} versi={frac[1]:.0%} virgi={frac[2]:.0%}  |  '
              f'perturbation by source  s={pert[0]:.2f} v1={pert[1]:.2f} v2={pert[2]:.2f}')
