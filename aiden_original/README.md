> **Provenance (verified 2026-10-16):** these files are Aidan Jared's code as of his last commit `f771f9a`
> (24 Mar 2026) with **one change made by us**: `util.py` line 267 `np.linalg.norm((point1, point2))` was
> corrected to `np.linalg.norm(point1 - point2)` in commit `b8a821c` (2026-07-17). For the untouched original
> use `git show f771f9a:util.py`. His split-first, training-only bias injection (lines 187-196) is unchanged.

# Aiden's Original Code (preserved for reference)

This folder contains the original codebase from prior work by Aiden, preserved
as-is for reference and comparison.

## Contents

- `util.py` — core library: `audit_tree()`, `audit_tree_bias()`, `labelencoding()`, synthetic data generation, SVM path (commented out)
- `bias.py` — driver for the original bias (coverage gap) experiment using `audit_tree_bias()`
- `main.py` — driver for the original clean (margin) experiment using `audit_tree()`
- `eda.ipynb` — exploratory analysis of the clean experiment
- `eda_bias.ipynb` — exploratory analysis of the bias experiment (note: one plot has a mislabeled axis)
- `plots/` — original figures

## Key limitations (identified & addressed in our work)

1. **White-box, tree-only attack** (`DecisionTreeAttack`) — cannot be used on SVM or neural networks
2. **Buggy density metric** — Frobenius-norm bug (`np.linalg.norm((p1,p2))` instead of `‖p1−p2‖`), density inversion, mislabeled axis
3. **Only coverage gap** — no testing of other defect types (label noise, feature noise, etc.)
4. **Accuracy confound** — his density signal tracked test accuracy decline (0.934→0.836), not purely geometric change
5. **SVM path never completed** — `audit_svc()` is commented out

## Our extensions

See `../dtree_attack_experiment/`, `../hsj_svm_experiment/`, `../hsj_label_noise_experiment/`,
`../master_analysis.ipynb`, and `../README.md` for the full story.
