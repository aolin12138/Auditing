"""
Shared defect injection + measurement pipeline.
Independent of util.py — does not modify original code.

Each defect injector takes (X_train, y_train, severity, seed, **kwargs)
and returns (X_train_modified, y_train_modified).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import polars as pl
from art.attacks.evasion import DecisionTreeAttack
from art.estimators.classification.scikitlearn import ScikitlearnDecisionTreeClassifier
from sklearn.cluster import OPTICS
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from util import eval_model


# ── Defect injectors ──────────────────────────────────────────────

def inject_label_noise(X_train, y_train, noise_level, seed, rng=None):
    """
    Randomly flip noise_level fraction of training labels to a different class.
    Returns X_train unchanged, y_train with flipped labels.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    y_mod = y_train.copy()
    n_samples = len(y_mod)
    n_flip = int(n_samples * noise_level)

    classes = np.unique(y_mod)

    flip_idx = rng.choice(n_samples, size=n_flip, replace=False)
    for idx in flip_idx:
        current = y_mod[idx].item()
        other = [c for c in classes if c != current]
        y_mod[idx] = rng.choice(other)

    actual_noise = (y_mod != y_train).sum() / n_samples
    return X_train, y_mod, actual_noise


# ── Shared measurement pipeline ───────────────────────────────────

def measure_defect(
    X,
    y,
    defect_type,
    defect_level,
    depth=10,
    SEED=42,
    noise_seed=0,
    max_features=2,
    n_splits=10,
    dbscan=True,
):
    """
    Run one complete experiment for a given (dataset, feature, seed, defect_level).

    Parameters
    ----------
    defect_type : str
        'label_noise' or 'coverage_gap'
    defect_level : float
        Severity (0.0 to 1.0). For label_noise: fraction flipped.
    SEED : int
        Seed for train/test split and model init.
    noise_seed : int
        Separate seed for noise injection randomness.
    """
    seed_rng = np.random.default_rng(SEED)
    noise_rng = np.random.default_rng(noise_seed * 1000 + SEED)

    model = DecisionTreeClassifier(
        random_state=SEED,
        max_features=max_features,
        max_depth=depth,
        min_samples_split=2,
        min_samples_leaf=1,
    )
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    train_acces = []
    train_f1s = []
    test_acces = []
    test_f1s = []
    avg_acces = []
    avg_f1s = []
    avg_distances = []
    clusters = []
    actual_levels = []

    for train_idx, test_idx in skf.split(X, y):
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]

        # Inject defect
        if defect_type == "label_noise":
            X_train_d, y_train_d, actual = inject_label_noise(
                X_train, y_train, defect_level, SEED, rng=noise_rng
            )
            actual_levels.append(actual)

        elif defect_type == "coverage_gap":
            # Not implemented here — use util.audit_tree_bias for coverage gap.
            # This stub exists for future consolidation.
            raise NotImplementedError(
                "coverage_gap: use util.audit_tree_bias() directly"
            )

        else:
            raise ValueError(f"Unknown defect_type: {defect_type}")

        # Train
        model.fit(X_train_d, y_train_d)
        classifier = ScikitlearnDecisionTreeClassifier(model)

        # Evaluate
        _, test_acc, test_f1 = eval_model(
            X_test, y_test.reshape(y_test.shape[0], 1), classifier
        )
        _, train_acc, train_f1 = eval_model(
            X_train, y_train.reshape(y_train.shape[0], 1), classifier
        )

        # Attack
        attack = DecisionTreeAttack(classifier=classifier)
        try:
            x_attack_adv = attack.generate(x=X_test)
        except Exception:
            break

        _, acc_adv, f1_adv = eval_model(
            x_attack_adv, y_test.reshape(y_test.shape[0], 1), classifier
        )

        # OPTICS clustering + density measurement
        distances = []
        if dbscan:
            clustering = OPTICS().fit(x_attack_adv)
            labels = np.unique(clustering.labels_[clustering.labels_ != -1])
            for i in labels:
                idx = np.argwhere(clustering.labels_ == i)
                group = x_attack_adv[idx]
                i = 0
                distance = 0
                count = 0
                if group.shape[0] > 1:
                    for point1 in group:
                        for point2 in group[i + 1:]:
                            distance += np.linalg.norm(point1 - point2).item()
                            count += 1
                    distance = count / (distance + 1)
                    distances.append(distance)
                else:
                    distances.append(0)
            clusters.append(len(labels))

        adv_distance = np.mean(distances) if distances else 0.0

        train_acces.append(train_acc)
        train_f1s.append(train_f1)
        test_acces.append(test_acc)
        test_f1s.append(test_f1)
        avg_acces.append(acc_adv)
        avg_f1s.append(f1_adv)
        avg_distances.append(adv_distance)

    avg_actual = np.mean(actual_levels) if actual_levels else defect_level

    res = {
        "train acc": train_acces,
        "train f1": train_f1s,
        "test acc": test_acces,
        "test f1": test_f1s,
        "adv acc": avg_acces,
        "adv f1": avg_f1s,
        "adv distance": avg_distances,
        "clusters": clusters,
        "defect_level": defect_level,
        "actual_level": avg_actual,
        "defect_type": defect_type,
    }
    return pl.from_dict(res)
