"""Which leaf does ART's DecisionTreeAttack actually send a point to?  (final report: Methods + App. D)

ART 1.20.1 `art/attacks/evasion/decision_tree_attack.py` walks up the input's decision path and,
at each ancestor, depth-first searches the sibling subtree for a leaf of a different class. Two
readings of where it stops:
  (A) "nearest": the first ancestor (walking up) whose sibling subtree contains such a leaf
      -- the description in outlier_experiment/MECHANISM.md section 1, step 3;
  (B) "as coded": the loop condition is `while |position| < len(path) - 1 or not found` and the
      result is overwritten at every step, so the search continues up to the ancestor two levels
      below the root and the LAST search wins (continuing to the root only if that one fails).
This script attacks every correctly classified test point with the report's tree configuration
(standardised data, max_depth=None, 5 seeds x 5 folds) and records which reading predicts the
leaf the adversarial example lands in.

Result (2026-10-16): iris 712 points: (A) 0.78, (B) 1.00 ; wine 805 points: (A) 0.77, (B) 1.00.
=> (B). For ~22% of points the attack's destination is not the fewest-traversal leaf.
Side effect of (B): when the near-root searches all fail, the loop indexes past the root and
raises IndexError even though a deeper search had found a valid leaf (seen under heavy label
noise; handled per point in run_label_noise.py and counted as attack failures).

Run: .venv/Scripts/python.exe analysis/verify_dta_destination.py
"""
import warnings

import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import DecisionTreeAttack

warnings.filterwarnings("ignore")


def df_subtree(est, pos, orig):
    """Port of DecisionTreeAttack._df_subtree (untargeted)."""
    if est.get_left_child(pos) == est.get_right_child(pos):
        return [pos] if est.get_classes_at_node(pos) != orig else [-1]
    r = df_subtree(est, est.get_left_child(pos), orig)
    if r[0] != -1:
        return r + [pos]
    r = df_subtree(est, est.get_right_child(pos), orig)
    return r + [pos] if r[0] != -1 else [-1]


def sibling_search(est, path, pos, orig):
    anc, child = path[pos], path[pos + 1]
    sib = est.get_right_child(anc) if child == est.get_left_child(anc) else est.get_left_child(anc)
    return df_subtree(est, sib, orig)


def main():
    for name, load in [("iris", load_iris), ("wine", load_wine)]:
        X, y = load(return_X_y=True)
        X = StandardScaler().fit_transform(X)
        yoh = OneHotEncoder(sparse_output=False).fit(y.reshape(-1, 1))
        hit_a = hit_b = n = 0
        for seed in [42, 58, 125, 7, 13]:
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
                m = DecisionTreeClassifier(max_depth=None, random_state=42).fit(X[tr], y[tr])
                est = SklearnClassifier(m)
                est.fit(X[tr], yoh.transform(y[tr].reshape(-1, 1)))
                Xv, yv = X[te], y[te]
                for x in Xv[np.argmax(est.predict(Xv), 1) == yv]:
                    path = list(est.get_decision_path(x))
                    legit = int(np.argmax(est.predict(x[None])))
                    landed = int(m.apply(DecisionTreeAttack(classifier=est).generate(x[None]))[0])
                    first = next(r[0] for pos in range(-2, -len(path) - 1, -1)
                                 if (r := sibling_search(est, path, pos, legit))[0] != -1)
                    pos, res = -2, [-1]
                    while abs(pos) < len(path) - 1 or res[0] == -1:
                        res = sibling_search(est, path, pos, legit)
                        pos -= 1
                    n += 1
                    hit_a += landed == first
                    hit_b += landed == res[0]
        print(f"{name}: {n} attacked points | (A) nearest: {hit_a / n:.2f} | (B) as coded: {hit_b / n:.2f}")


if __name__ == "__main__":
    main()
