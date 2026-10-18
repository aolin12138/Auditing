"""Does the petal-length hole stop changing the fitted tree once the overlap band is removed?

Discussion 5.2 drafted the claim that the hole models at higher deletion are the same tree. Test it directly:
for each seed and fold, fit the overfit tree after spatial deletion of virginica along petal length (raw
iris, as results_attack_all 'iris_plen') at two deletion levels, and check whether the two fitted trees
predict identically everywhere their thresholds can matter (same structure: features + thresholds + leaf
classes).
"""
import importlib.util
from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("imb", ROOT / "defect_expansion_experiment" / "run_imbalance.py")
imb = importlib.util.module_from_spec(spec); spec.loader.exec_module(imb)
SEEDS = [42, 58, 125, 7, 13, 21, 99, 123, 200, 777] + list(range(300, 320))


def tree_sig(m):
    t = m.tree_
    return (tuple(t.feature), tuple(np.round(t.threshold, 10)), tuple(np.argmax(t.value[:, 0, :], axis=1)))


X, y = load_iris(return_X_y=True)
for a, b in ((0.5, 0.7), (0.6, 0.8), (0.8, 0.9), (0.4, 0.9)):
    same, total = 0, 0
    for seed in SEEDS:
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            sig = []
            for frac in (a, b):
                Xt, yt = imb.inject_delete(X[tr], y[tr], 2, 2, frac, "spatial", np.random.default_rng(seed))
                sig.append(tree_sig(DecisionTreeClassifier(max_depth=None, random_state=42).fit(Xt, yt)))
            same += sig[0] == sig[1]; total += 1
    print(f"spatial {int(a*100)}% vs {int(b*100)}%: identical fitted tree in {same}/{total} folds")
