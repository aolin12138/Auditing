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
