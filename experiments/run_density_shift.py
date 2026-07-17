"""
Compare density of original test points vs adversarial points (global, all points pooled).
Shows density SHIFT caused by the attack.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from art.attacks.evasion import DecisionTreeAttack
from art.estimators.classification.scikitlearn import ScikitlearnDecisionTreeClassifier
from ucimlrepo import fetch_ucirepo
from tqdm import tqdm
from util import labelencoding
from scipy.spatial.distance import pdist

SEED = 42
NOISE_LEVELS = np.linspace(0.0, 0.9, 10)
N_RUNS = 5

DATASETS = {}
for name in ["iris", "wine", "Car Evaluation"]:
    data = fetch_ucirepo(name)
    X = labelencoding(data.data.features.to_numpy().copy()).astype(np.float64)
    y = labelencoding(data.data.targets.to_numpy().copy()).flatten()
    DATASETS[name] = (X, y)


def density_global(X):
    """Global density: all points pooled, n_pairs / (total_dist + 1)."""
    n = len(X)
    if n < 2:
        return np.nan
    total_dist = pdist(X).sum()
    n_pairs = n * (n - 1) // 2
    return n_pairs / (total_dist + 1)


rows = []
total = len(DATASETS) * N_RUNS * len(NOISE_LEVELS)
pbar = tqdm(total=total, desc="Density shift")

for ds_name, (X, y) in DATASETS.items():
    all_classes = np.unique(y)
    ds_hash = hash(ds_name) % 10000

    for run in range(N_RUNS):
        run_seed = SEED + run
        rng = np.random.default_rng(SEED + run * 1000 + ds_hash)

        X_train, X_test, y_train_clean, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=run_seed)

        # Original test point density (no attack needed)
        orig_density = density_global(X_test)

        for nl in NOISE_LEVELS:
            y_train = y_train_clean.copy()
            n_flip = int(len(y_train) * nl)
            if n_flip > 0:
                idx = rng.choice(len(y_train), size=n_flip, replace=False)
                for i in idx:
                    others = [c for c in all_classes if c != y_train[i]]
                    y_train[i] = rng.choice(others)

            tree = DecisionTreeClassifier(max_depth=10, min_samples_leaf=1,
                                          random_state=run_seed)
            tree.fit(X_train, y_train)

            try:
                classifier = ScikitlearnDecisionTreeClassifier(tree)
                attack = DecisionTreeAttack(classifier=classifier)
                x_adv = attack.generate(x=X_test)
                adv_density = density_global(x_adv)
            except:
                adv_density = np.nan

            rows.append(dict(
                dataset=ds_name, noise_level=float(nl), run=run,
                train_acc=tree.score(X_train, y_train),
                test_acc=tree.score(X_test, y_test),
                orig_density=orig_density,     # same for all noise levels in this run
                adv_density=adv_density,
                density_diff=adv_density - orig_density,  # positive = attack makes points denser
                density_ratio=adv_density / orig_density if orig_density > 0 else np.nan,
                n_points=len(X_test)
            ))
            pbar.update(1)

pbar.close()

df = pd.DataFrame(rows)
df.to_parquet("data/data_density_shift.parquet", index=False)
print(f"\nSaved {len(df)} rows")

# ── Summary ───────────────────────────────────────────────
print("\n=== DENSITY SHIFT (orig -> adv at noise 0.0 vs 0.9) ===")
for ds in DATASETS:
    sub = df[df.dataset == ds]
    print(f"\n{ds}:")
    print(f"  Original test density: {sub['orig_density'].iloc[0]:.4f} (same across noise)")
    for nl in [0.0, 0.9]:
        sn = sub[sub.noise_level == nl]
        print(f"  noise={nl:.1f}: adv_density={sn['adv_density'].mean():.4f} +/- {sn['adv_density'].std():.4f}, "
              f"diff={sn['density_diff'].mean():+.4f}, ratio={sn['density_ratio'].mean():.2f}x")
