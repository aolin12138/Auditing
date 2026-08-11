# Model-Family Experiment — Plan

> **Question:** do the established signals (coverage-gap spread ↑, label-noise null)
> generalise beyond the DecisionTree + SVM pair? And can we *visualise* the mechanism?
> Answers the "more model families" item in `.wiki/08-open-questions.md`.

Status: **planned, not yet run.**

---

## 1. The governing constraint (do not break this)

Cross-model comparison is only valid if the **same attack** runs on every model.
That attack is **HopSkipJump (HSJ)** — decision-based, black-box, needs only predicted
labels (`.wiki/05-key-decisions.md`). `DecisionTreeAttack` is single-tree white-box and
**cannot** read an ensemble, so RF/XGBoost use **HSJ only**. Keep Tree+DTA as the
deterministic reference on the tree.

## 2. Models to add

| Model | Library | Attack | Why | Prediction |
|-------|---------|--------|-----|------------|
| **RandomForest** | sklearn (present) | HSJ | bagging → does averaging *wash out* the coverage-gap signal? | signal **weakens** |
| **XGBoost** | `xgboost` (ADD dep) | HSJ | boosting → does hard-point-chasing *sharpen* it? | signal **holds/amplifies** |
| **2-feature viz setup** | any of the above | HSJ | plot the boundary + adversarial landings in 2D | mechanism made *visible* |
| **Small MLP + PGD** *(stretch)* | `torch` (ADD dep) | HSJ (anchor) + PGD | differentiable → gradient attack, the other open-question item | — |

## 3. Design

- **Grids:** re-run the *existing* coverage-gap and label-noise grids with RF and XGBoost
  swapped in for the model, HSJ as the attack. Same iris, 5-fold, 3 seeds, all `tc×feat`.
  Directly comparable to the current Tree/SVM+HSJ results.
- **Metric & schema:** identical to existing HSJ grids (`cluster_stats`, spread reported).
  Add a `model` column (`rf`/`xgb`).
- **Hang handling:** RF and XGBoost are tree ensembles → many flat facets → HSJ hang risk is
  *higher* than single tree. **Reuse the subprocess-timeout runner** (`.wiki/06`); record
  timeouts as NaN.

## 4. The 2-feature visualization setup (highest report value)

Purpose: close the open-question "visualise the actual boundary and adversarial landing
points in 2D to prove the mechanism."

- Restrict iris to **petal length + petal width** (features 2,3 — the most separating pair).
- Train the model; inject a defect (coverage gap *or* an outlier from the sibling thread).
- Plot: decision regions (mesh `predict`), training points, and the **adversarial landing
  points** coloured by cluster. Overlay OPTICS clusters.
- Expected picture: under coverage gap / outlier, adversarial points **scatter into the
  extrapolated region** (empty of data) instead of concentrating at a natural boundary.

Caveats: 2D iris accuracy is slightly lower; run this as its own small grid, not merged with
the 4-feature numbers. This is a *figure-generating* experiment, not a Cohen's-d experiment.

## 5. Dependencies

- RandomForest: already available (scikit-learn).
- **XGBoost:** add `xgboost` to `pyproject.toml` (`uv add xgboost`).
- **MLP + PGD (stretch):** add `torch`; use ART's `PyTorchClassifier` +
  `ProjectedGradientDescent`. Only if we pursue the gradient-attack lens.

## 6. Success criteria (observable)

1. RF+HSJ and XGBoost+HSJ coverage-gap spread trends recorded with Cohen's d vs bias,
   comparable to the existing Tree/SVM+HSJ table.
2. A clear statement of whether the coverage-gap signal **survives** ensembling
   (RF weaker? XGBoost holds?).
3. At least one 2-feature figure showing adversarial points scattering into the
   extrapolated region under a defect vs. concentrating under clean data.

## 7. Relationship to the outlier thread

The outlier thread (`../outlier_experiment/PLAN.md`) *shares* these models — H4 there
(robustness × model) is answered with the RF/XGBoost runners built here. Build the model
adapters once; both threads consume them.
