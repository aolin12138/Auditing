import os, sys, time
for v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[v] = '1'
import numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import HopSkipJump

bias, seed, tc, feat, nm = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
iris = load_iris(); X, y = iris.data, iris.target
np.random.seed(seed)
mask = y == tc; order = np.argsort(X[mask][:, feat]); Xs, ys = X[mask][order], y[mask][order]
nk = max(int(len(Xs)*(1-bias)), 3)
Xb = np.vstack([X[~mask], Xs[-nk:]]); yb = np.hstack([y[~mask], ys[-nk:]])
yoh = OneHotEncoder(sparse_output=False).fit_transform(yb.reshape(-1,1))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
model = DecisionTreeClassifier(max_depth=3, random_state=42) if nm=='tree' else SVC(kernel='rbf', probability=True, random_state=42)
t0 = time.time()
for fi,(tr,te) in enumerate(skf.split(Xb,yb)):
    Xt,Xv,yt,yv = Xb[tr],Xb[te],yb[tr],yb[te]
    m = model.__class__(**model.get_params()).fit(Xt,yt)
    art = SklearnClassifier(m); art.fit(Xt,yoh[tr])
    p = np.argmax(art.predict(Xv),axis=1); c = p==yv
    hs = HopSkipJump(classifier=art,norm=2,max_iter=10,max_eval=200,init_eval=50,verbose=False)
    adv = hs.generate(Xv[c]) if c.sum()>0 else np.array([])
    print(f'  fold{fi}: {time.time()-t0:.1f}s cum, {len(adv)} adv, correct={c.sum()}/{len(yv)}', flush=True)
print(f'OK {nm} bias={bias} tc={tc} feat={feat}: {time.time()-t0:.1f}s', flush=True)
