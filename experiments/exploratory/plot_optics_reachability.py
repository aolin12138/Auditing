"""
OPTICS reachability plots for label noise experiment.
Shows how the reachability structure changes with noise.
Runs on iris (continuous) vs Car Evaluation (categorical) for contrast.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from art.attacks.evasion import DecisionTreeAttack
from art.estimators.classification.scikitlearn import ScikitlearnDecisionTreeClassifier
from ucimlrepo import fetch_ucirepo
from util import labelencoding

NOISE_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8]
SEED = 42

def get_adversarial_points(X, y, noise_level, rng):
    """Run one fold: inject noise, train tree, attack, return adv points."""
    # Use first fold only for illustration
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(skf.split(X, y))
    X_train, y_train = X[train_idx], y[train_idx].flatten()
    X_test, y_test = X[test_idx], y[test_idx].flatten()

    # Inject label noise
    y_noisy = y_train.copy()
    n_flip = int(len(y_noisy) * noise_level)
    flip_idx = rng.choice(len(y_noisy), size=n_flip, replace=False)
    classes = np.unique(y_noisy)
    for c in classes:
        other = [cl for cl in classes if cl != c]
        mask = (y_train == c) & np.isin(np.arange(len(y_train)), flip_idx)
        y_noisy[mask] = rng.choice(other, size=mask.sum())

    tree = DecisionTreeClassifier(max_depth=10, min_samples_leaf=1,
                                   min_samples_split=2, random_state=SEED)
    tree.fit(X_train, y_noisy)
    classifier = ScikitlearnDecisionTreeClassifier(tree)
    attack = DecisionTreeAttack(classifier=classifier)
    x_adv = attack.generate(x=X_test)
    return x_adv, tree.get_n_leaves()


def plot_reachability(ax, x_adv, noise_level, n_leaves, title):
    """Plot OPTICS reachability on an axis."""
    optics = OPTICS(min_samples=5, xi=0.05).fit(x_adv)
    
    # Get reachability and ordering
    reachability = optics.reachability_[optics.ordering_]
    # Replace inf with max finite value
    finite = reachability[np.isfinite(reachability)]
    if len(finite) > 0:
        reachability[~np.isfinite(reachability)] = np.max(finite) * 1.1
    
    n_clusters = len(np.unique(optics.labels_[optics.labels_ != -1]))
    
    ax.fill_between(range(len(reachability)), 0, reachability, alpha=0.3, color='purple')
    ax.plot(reachability, color='purple', linewidth=0.8)
    
    # Mark cluster boundaries (where labels change)
    labels_ordered = optics.labels_[optics.ordering_]
    prev = labels_ordered[0]
    for i in range(1, len(labels_ordered)):
        if labels_ordered[i] != prev and labels_ordered[i] != -1:
            ax.axvline(x=i, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
        prev = labels_ordered[i]
    
    ax.set_title(f"Noise={noise_level:.1f}\n{n_clusters} clusters, {n_leaves} leaves", fontsize=8)
    ax.set_xlabel("Order", fontsize=7)
    ax.set_ylabel("Reachability", fontsize=7)
    ax.tick_params(labelsize=6)


# ── Run for both datasets ──────────────────────────────────────

fig, axes = plt.subplots(2, 5, figsize=(18, 8))

for row, (ds_name, fetch_name) in enumerate([("iris", "iris"), ("Car Evaluation", "Car Evaluation")]):
    data = fetch_ucirepo(fetch_name)
    X = labelencoding(data.data.features.to_numpy().copy()).astype(np.float64)
    y = labelencoding(data.data.targets.to_numpy().copy())
    rng = np.random.default_rng(SEED + row)
    
    for col, nl in enumerate(NOISE_LEVELS):
        x_adv, n_leaves = get_adversarial_points(X, y, nl, rng)
        plot_reachability(axes[row, col], x_adv, nl, n_leaves,
                         title=f"{ds_name}" if col == 0 else "")
    
    # Row label
    axes[row, 0].set_ylabel(f"{ds_name}\nReachability", fontsize=10, fontweight='bold')

fig.suptitle("OPTICS Reachability Plots: How Adversarial Cluster Structure Changes with Label Noise",
             fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig("experiments/plots/optics_reachability_noise.png", dpi=200, bbox_inches="tight")
print("Saved experiments/plots/optics_reachability_noise.png")
