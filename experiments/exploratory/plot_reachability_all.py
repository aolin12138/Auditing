"""
Comprehensive reachability plots: all datasets × noise levels × 3 runs.
Shows variance across different noise injections.
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
from sklearn.datasets import make_classification
from util import labelencoding

SEED = 42
NOISE_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8]
N_RUNS = 3  # 3 different folds/noise seeds per cell
DATASETS = {}

# ── Load real datasets ──────────────────────────────────────

for name in ["iris", "wine", "Car Evaluation"]:
    data = fetch_ucirepo(name)
    X = labelencoding(data.data.features.to_numpy().copy()).astype(np.float64)
    y = labelencoding(data.data.targets.to_numpy().copy())
    DATASETS[name] = (X, y)

# ── Create synthetic datasets ──────────────────────────────

# Continuous well-separated
X, y = make_classification(1000, 3, n_informative=3, n_redundant=0, n_classes=4,
                           n_clusters_per_class=1, class_sep=1.5, random_state=42)
DATASETS["synth_sep"] = (X.astype(np.float64), y.astype(np.int64).reshape(-1,1))

# Continuous overlapping
X, y = make_classification(1000, 3, n_informative=3, n_redundant=0, n_classes=4,
                           n_clusters_per_class=2, class_sep=0.6, random_state=43)
DATASETS["synth_over"] = (X.astype(np.float64), y.astype(np.int64).reshape(-1,1))

# Categorical
X, y = make_classification(1000, 3, n_informative=3, n_redundant=0, n_classes=4,
                           n_clusters_per_class=1, class_sep=0.5, random_state=44)
Xc = np.zeros_like(X)
for f in range(3):
    bins = np.percentile(X[:, f], [0, 25, 50, 75, 100])
    Xc[:, f] = np.digitize(X[:, f], bins[:-1]) - 1
DATASETS["synth_cat"] = (Xc.astype(np.float64), y.astype(np.int64).reshape(-1,1))

# ── Plot ────────────────────────────────────────────────────

fig, axes = plt.subplots(len(DATASETS), len(NOISE_LEVELS),
                         figsize=(18, 2.8 * len(DATASETS)))
colors = plt.cm.viridis(np.linspace(0.15, 0.85, N_RUNS))

for row, (ds_name, (X, y)) in enumerate(DATASETS.items()):
    rng = np.random.default_rng(SEED + row)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED + row)
    
    for col, nl in enumerate(NOISE_LEVELS):
        ax = axes[row, col]
        
        for run in range(N_RUNS):
            # Use different split + noise seed per run
            fold_idx = run  # different folds
            splits = list(skf.split(X, y))
            train_idx, test_idx = splits[fold_idx]
            X_train = X[train_idx]
            y_train = y[train_idx].flatten()
            X_test = X[test_idx]
            
            # Inject noise
            noise_rng = np.random.default_rng(SEED + row * 100 + col * 10 + run)
            y_noisy = y_train.copy()
            n_flip = int(len(y_noisy) * nl)
            flip_idx = noise_rng.choice(len(y_noisy), size=n_flip, replace=False)
            classes = np.unique(y_noisy)
            for c in classes:
                other = [cl for cl in classes if cl != c]
                mask = (y_train == c) & np.isin(np.arange(len(y_train)), flip_idx)
                y_noisy[mask] = noise_rng.choice(other, size=mask.sum())
            
            tree = DecisionTreeClassifier(max_depth=10, min_samples_leaf=1,
                                           min_samples_split=2, random_state=SEED + run)
            tree.fit(X_train, y_noisy)
            classifier = ScikitlearnDecisionTreeClassifier(tree)
            attack = DecisionTreeAttack(classifier=classifier)
            try:
                x_adv = attack.generate(x=X_test)
            except:
                continue
            
            optics = OPTICS(min_samples=5, xi=0.05).fit(x_adv)
            reach = optics.reachability_[optics.ordering_]
            finite = reach[np.isfinite(reach)]
            if len(finite) > 0:
                reach[~np.isfinite(reach)] = np.max(finite) * 1.05
            
            n_cl = len(np.unique(optics.labels_[optics.labels_ != -1]))
            
            ax.plot(reach, color=colors[run], linewidth=0.6, alpha=0.8,
                    label=f"fold {run+1} ({n_cl} cl)" if col == 0 else "")
        
        if col == 0:
            ax.set_ylabel(f"{ds_name}\nReachability", fontsize=9, fontweight='bold')
            ax.legend(fontsize=6, loc='upper right')
        if row == 0:
            ax.set_title(f"Noise={nl:.1f}", fontsize=10, fontweight='bold')
        if row == len(DATASETS) - 1:
            ax.set_xlabel("Order", fontsize=7)
        ax.tick_params(labelsize=5)

fig.suptitle("OPTICS Reachability: All Datasets × Noise Levels × 3 Runs\n"
             "Each color = one fold/run, showing variance",
             fontsize=12, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig("experiments/plots/reachability_all_datasets.png", dpi=200, bbox_inches="tight")
print("Saved experiments/plots/reachability_all_datasets.png")
