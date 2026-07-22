"""Does the SAME 'adv cloud -> original test cloud' artifact drive the MAIN
result (SVM + HSJ, coverage gap)? If ratio ~ 1 and rises with bias, the main
signal is also just reflecting original test geometry, not the attack.

Coverage gap changes the test set too, so we track orig_spread explicitly."""
import os
for v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[v]='1'
import numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial.distance import pdist
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import HopSkipJump

iris = load_iris(); X, y = iris.data, iris.target
BIAS = [0.1, 0.3, 0.5, 0.7, 0.9]
SEEDS = [42, 58, 125]

def inject_bias(X, y, tc, fi, bias):
    mask = y == tc; order = np.argsort(X[mask][:, fi])
    Xs, ys = X[mask][order], y[mask][order]
    nk = max(int(len(Xs)*(1-bias)), 3)
    return np.vstack([X[~mask], Xs[-nk:]]), np.hstack([y[~mask], ys[-nk:]])

print('===== SVM + HopSkipJump, coverage gap =====')
print(f'{"bias":>6} {"perturb":>8} {"adv_spread":>11} {"orig_spread":>12} {"ratio":>6} {"test_acc":>9}')
for bias in BIAS:
    perturbs, adv_sp, orig_sp, accs = [], [], [], []
    for seed in SEEDS:
        np.random.seed(seed)
        for tc in [0,1,2]:
            for feat in [0,1,2,3]:
                Xb, yb = inject_bias(X, y, tc, feat, bias)
                yoh = OneHotEncoder(sparse_output=False).fit(y.reshape(-1,1))
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                for tr, te in skf.split(Xb, yb):
                    Xt, Xv, yt, yv = Xb[tr], Xb[te], yb[tr], yb[te]
                    m = SVC(kernel='rbf', probability=True, random_state=42).fit(Xt, yt)
                    art = SklearnClassifier(m); art.fit(Xt, yoh.transform(yt.reshape(-1,1)))
                    p = np.argmax(art.predict(Xv), axis=1); c = p == yv
                    if c.sum() < 4: continue
                    Xc = Xv[c]
                    hs = HopSkipJump(classifier=art, norm=2, max_iter=10, max_eval=200,
                                     init_eval=50, verbose=False)
                    adv = hs.generate(Xc)
                    ap = np.argmax(art.predict(adv), axis=1)
                    ok = ap != yv[c]
                    if ok.sum() < 4: continue
                    adv_ok, orig_ok = adv[ok], Xc[ok]
                    perturbs.append(np.linalg.norm(adv_ok-orig_ok, axis=1).mean())
                    adv_sp.append(pdist(adv_ok).mean())
                    orig_sp.append(pdist(orig_ok).mean())
                    accs.append((yv==p).mean())
    r = np.mean(adv_sp)/np.mean(orig_sp)
    print(f'{bias:>6.1f} {np.mean(perturbs):>8.4f} {np.mean(adv_sp):>11.4f} '
          f'{np.mean(orig_sp):>12.4f} {r:>6.3f} {np.mean(accs):>9.3f}')
