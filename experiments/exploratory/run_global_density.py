"""
Compare per-cluster vs global density for label noise on real datasets.
Global = all adversarial points in one bucket, not split by OPTICS clusters.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import OPTICS
from art.attacks.evasion import DecisionTreeAttack
from art.estimators.classification.scikitlearn import ScikitlearnDecisionTreeClassifier
from ucimlrepo import fetch_ucirepo
from tqdm import tqdm
from util import labelencoding
from scipy.spatial.distance import pdist

SEED = 42
NOISE_LEVELS = np.linspace(0.0, 0.9, 10)
N_RUNS = 5

# ── Load real datasets ──────────────────────────────────────
DATASETS = {}
for name in ["iris", "wine", "Car Evaluation"]:
    data = fetch_ucirepo(name)
    X = labelencoding(data.data.features.to_numpy().copy()).astype(np.float64)
    y = labelencoding(data.data.targets.to_numpy().copy()).flatten()
    DATASETS[name] = (X, y)


# ── Metrics ─────────────────────────────────────────────────

def density_per_cluster(X_adv):
    """Current method: mean density across OPTICS clusters."""
    if len(X_adv) < 5:
        return np.nan, 0, np.nan
    optics = OPTICS(min_samples=5, xi=0.05).fit(X_adv)
    labels = optics.labels_
    cids = sorted(set(labels) - {-1})
    n_clusters = len(cids)
    if n_clusters == 0:
        return np.nan, 0, np.nan
    densities = []
    for cid in cids:
        idx = np.where(labels == cid)[0]
        if len(idx) < 2:
            continue
        total_dist = pdist(X_adv[idx]).sum()
        n_pairs = len(idx) * (len(idx) - 1) // 2
        densities.append(n_pairs / (total_dist + 1))
    return np.mean(densities), n_clusters, np.std(densities) if len(densities)>1 else 0


def density_global(X_adv):
    """Global density: all points pooled, no clustering."""
    n = len(X_adv)
    if n < 2:
        return np.nan
    total_dist = pdist(X_adv).sum()
    n_pairs = n * (n - 1) // 2
    return n_pairs / (total_dist + 1)


# ── Run ─────────────────────────────────────────────────────

rows = []
total = len(DATASETS) * N_RUNS * len(NOISE_LEVELS)
pbar = tqdm(total=total, desc="Global density")

for ds_name, (X, y) in DATASETS.items():
    all_classes = np.unique(y)
    ds_hash = hash(ds_name) % 10000

    for run in range(N_RUNS):
        run_seed = SEED + run
        rng = np.random.default_rng(SEED + run * 1000 + ds_hash)

        X_train, X_test, y_train_clean, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=run_seed)

        for nl in NOISE_LEVELS:
            # Independent noise
            y_train = y_train_clean.copy()
            n_flip = int(len(y_train) * nl)
            if n_flip > 0:
                idx = rng.choice(len(y_train), size=n_flip, replace=False)
                for i in idx:
                    others = [c for c in all_classes if c != y_train[i]]
                    y_train[i] = rng.choice(others)

            # Train overfit tree
            tree = DecisionTreeClassifier(max_depth=10, min_samples_leaf=1,
                                          random_state=run_seed)
            tree.fit(X_train, y_train)

            # Attack
            try:
                classifier = ScikitlearnDecisionTreeClassifier(tree)
                attack = DecisionTreeAttack(classifier=classifier)
                x_adv = attack.generate(x=X_test)
            except:
                rows.append(dict(dataset=ds_name, noise_level=float(nl), run=run,
                                 train_acc=np.nan, test_acc=np.nan,
                                 cluster_density=np.nan, n_clusters=0,
                                 global_density=np.nan, n_points=len(X_test)))
                pbar.update(1)
                continue

            cluster_dens, n_clusters, _ = density_per_cluster(x_adv)
            global_dens = density_global(x_adv)

            rows.append(dict(
                dataset=ds_name, noise_level=float(nl), run=run,
                train_acc=tree.score(X_train, y_train),
                test_acc=tree.score(X_test, y_test),
                adv_acc=tree.score(x_adv, y_test),
                cluster_density=cluster_dens,
                n_clusters=n_clusters,
                global_density=global_dens,
                n_points=len(X_test)
            ))
            pbar.update(1)

pbar.close()

df = pd.DataFrame(rows)
df.to_parquet("data/data_global_density.parquet", index=False)
print(f"\nSaved {len(df)} rows to data/data_global_density.parquet")
print(df.groupby("dataset").size())

# ── Quick summary ───────────────────────────────────────────
print("\n=== GLOBAL DENSITY (noise 0.0 -> 0.9) ===")
for ds in DATASETS:
    sub = df[df.dataset == ds]
    agg = sub.groupby("noise_level")["global_density"].agg(["mean", "std"])
    print(f"{ds:20s}  {agg['mean'].iloc[0]:.4f} -> {agg['mean'].iloc[-1]:.4f}")

print("\n=== CLUSTER DENSITY (noise 0.0 -> 0.9) ===")
for ds in DATASETS:
    sub = df[df.dataset == ds]
    agg = sub.groupby("noise_level")["cluster_density"].agg(["mean", "std"])
    print(f"{ds:20s}  {agg['mean'].iloc[0]:.4f} -> {agg['mean'].iloc[-1]:.4f}")
