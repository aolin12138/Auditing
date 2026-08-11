"""Probe timing: RF and XGB under HSJ at different n_estimators. Decides grid feasibility."""
import os, time, warnings
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_v] = '1'
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from art.estimators.classification import SklearnClassifier, XGBoostClassifier
from art.attacks.evasion import HopSkipJump

X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


def time_attack(name, art):
    p = np.argmax(art.predict(Xte), axis=1); c = p == yte
    t0 = time.time()
    hs = HopSkipJump(classifier=art, norm=2, max_iter=10, max_eval=200, init_eval=50, verbose=False)
    adv = hs.generate(Xte[c])
    ap = np.argmax(art.predict(adv), axis=1)
    dt = time.time() - t0
    print(f'  {name}: {dt:.1f}s / {c.sum()} pts = {dt/c.sum():.2f}s/pt   '
          f'(=> ~{dt/c.sum()*28*5:.0f}s per 5-fold cell)', flush=True)


for ne in [30, 50]:
    print(f'n_estimators={ne}:', flush=True)
    rf = RandomForestClassifier(n_estimators=ne, random_state=42).fit(Xtr, ytr)
    time_attack(f'RF ne={ne} ', SklearnClassifier(rf))
    try:
        xgb = XGBClassifier(n_estimators=ne, max_depth=3, random_state=42, eval_metric='mlogloss').fit(Xtr, ytr)
        art_xgb = XGBoostClassifier(model=xgb, nb_features=X.shape[1], nb_classes=3)
        time_attack(f'XGB ne={ne}', art_xgb)
    except Exception as e:
        print(f'  XGB ne={ne} ERROR: {type(e).__name__}: {e}', flush=True)
