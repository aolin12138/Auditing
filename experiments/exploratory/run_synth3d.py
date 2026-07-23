"""
Realistic synthetic 3D datasets + label noise + reachability plots.
3 variants: well-separated continuous, overlapping continuous, categorical (like Car Eval).
1000 samples each.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.cluster import OPTICS
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from art.attacks.evasion import DecisionTreeAttack
from art.estimators.classification.scikitlearn import ScikitlearnDecisionTreeClassifier
from experiments.defects import measure_defect

SEED = 42
NOISE_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8]
N_SAMPLES = 1000

# ── Create datasets ──────────────────────────────────────────

datasets = {}

# 1. Well-separated continuous — like iris
X1, y1 = make_classification(
    n_samples=N_SAMPLES, n_features=3, n_informative=3, n_redundant=0,
    n_classes=4, n_clusters_per_class=1, class_sep=1.5,
    weights=[0.3, 0.3, 0.2, 0.2],  # imbalanced
    flip_y=0.02,  # slight label noise
    random_state=42
)
datasets["synth_separated"] = (X1.astype(np.float64), y1.astype(np.int64).reshape(-1, 1))

# 2. Overlapping continuous — like wine
X2, y2 = make_classification(
    n_samples=N_SAMPLES, n_features=3, n_informative=3, n_redundant=0,
    n_classes=4, n_clusters_per_class=2, class_sep=0.6,  # 2 subclusters per class
    weights=[0.35, 0.25, 0.25, 0.15],
    flip_y=0.03,
    random_state=43
)
datasets["synth_overlapping"] = (X2.astype(np.float64), y2.astype(np.int64).reshape(-1, 1))

# 3. Categorical — discretize features to 4-5 bins each (like Car Evaluation)
X3, y3 = make_classification(
    n_samples=N_SAMPLES, n_features=3, n_informative=3, n_redundant=0,
    n_classes=4, n_clusters_per_class=1, class_sep=0.5,
    weights=[0.3, 0.3, 0.2, 0.2],
    flip_y=0.02,
    random_state=44
)
# Discretize each feature into 4-5 bins
X3_cat = np.zeros_like(X3)
for f in range(3):
    bins = np.percentile(X3[:, f], [0, 25, 50, 75, 100])
    X3_cat[:, f] = np.digitize(X3[:, f], bins[:-1]) - 1
datasets["synth_categorical"] = (X3_cat.astype(np.float64), y3.astype(np.int64).reshape(-1, 1))


# ── 3D distribution plot ────────────────────────────────────

fig = plt.figure(figsize=(18, 6))
colors = plt.cm.tab10(np.linspace(0, 1, 4))
for idx, (name, (X, y)) in enumerate(datasets.items()):
    ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
    yf = y.flatten()
    for c in np.unique(yf):
        ax.scatter(X[yf==c, 0], X[yf==c, 1], X[yf==c, 2],
                   c=[colors[c]], label=f"Class {c}", alpha=0.6, s=10)
    ax.set_title(f"{name}\n({len(X)} samples, {len(np.unique(yf))} classes)", fontsize=10)
    ax.set_xlabel("f0"); ax.set_ylabel("f1"); ax.set_zlabel("f2")
    ax.legend(fontsize=7, loc='upper right')
fig.suptitle("Realistic Synthetic 3D Datasets", fontweight="bold", fontsize=13)
fig.tight_layout()
fig.savefig("experiments/plots/synth3d_realistic_dist.png", dpi=200, bbox_inches="tight")
print("Saved distribution plot")


# ── Reachability plots ──────────────────────────────────────

def get_adv_and_optics(X, y, noise_level, rng):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(skf.split(X, y))
    X_train, y_train = X[train_idx], y[train_idx].flatten()
    X_test = X[test_idx]

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
    optics = OPTICS(min_samples=5, xi=0.05).fit(x_adv)
    return optics, tree.get_n_leaves()

fig2, axes = plt.subplots(3, 5, figsize=(18, 11))
rng = np.random.default_rng(99)

for row, (name, (X, y)) in enumerate(datasets.items()):
    for col, nl in enumerate(NOISE_LEVELS):
        optics, n_leaves = get_adv_and_optics(X, y, nl, rng)
        reach = optics.reachability_[optics.ordering_]
        finite = reach[np.isfinite(reach)]
        if len(finite) > 0:
            reach[~np.isfinite(reach)] = np.max(finite) * 1.05
        
        n_cl = len(np.unique(optics.labels_[optics.labels_ != -1]))
        ax = axes[row, col]
        ax.fill_between(range(len(reach)), 0, reach, alpha=0.3, color='purple')
        ax.plot(reach, color='purple', linewidth=0.8)
        ax.set_title(f"Noise={nl:.1f} | {n_cl} clusters", fontsize=8)
        if col == 0:
            ax.set_ylabel(f"{name}\nReachability", fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=6)

fig2.suptitle("OPTICS Reachability: How Synthetic Datasets Respond to Label Noise",
              fontsize=12, fontweight='bold')
fig2.tight_layout()
fig2.savefig("experiments/plots/synth3d_reachability.png", dpi=200, bbox_inches="tight")
print("Saved reachability plot")


# ── Full label noise experiment ─────────────────────────────

noise_levels_full = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
seeds = [42, 125, 58]
noise_seeds = list(range(10))
res = pl.DataFrame()

for ds_name, (X, y) in datasets.items():
    print(f"\n{ds_name} ({X.shape[0]} samples)...")
    for seed in seeds:
        for ns in noise_seeds:
            if ns == 0:
                print(f"  seed={seed} noise_seeds=0-9 ...", end=" ", flush=True)
            for noise in noise_levels_full:
                results = measure_defect(
                    X, y, defect_type="label_noise",
                    defect_level=float(noise),
                    SEED=seed, noise_seed=ns,
                    n_splits=10, dbscan=True,
                )
                results = results.with_columns(
                    pl.lit(seed).alias("seed"),
                    pl.lit(ns).alias("noise_seed"),
                    pl.lit(ds_name).alias("dataset"),
                    pl.lit("label_noise").alias("defect_type"),
                )
                res = pl.concat([res, results])
        print("done")

res.write_parquet("data/data_label_noise_synth3d.parquet")
print(f"\nSaved {res.shape[0]} rows to data/data_label_noise_synth3d.parquet")
