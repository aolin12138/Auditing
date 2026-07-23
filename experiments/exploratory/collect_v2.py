"""
Data collection v2 — optimized. Tests noise mode × tree strategy × 6 datasets.
Vectorized density, capped tree depth, faster CV.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.cluster import OPTICS
from sklearn.datasets import make_classification
from art.attacks.evasion import DecisionTreeAttack
from art.estimators.classification.scikitlearn import ScikitlearnDecisionTreeClassifier
from ucimlrepo import fetch_ucirepo
from tqdm import tqdm
from util import labelencoding
from scipy.spatial.distance import pdist

BASE_SEED = 42
NOISE_LEVELS = np.linspace(0.0, 0.9, 10)
N_RUNS = 5
DATASETS = {}

# ── Load real datasets ──────────────────────────────────────
for name in ["iris", "wine", "Car Evaluation"]:
    data = fetch_ucirepo(name)
    X = labelencoding(data.data.features.to_numpy().copy()).astype(np.float64)
    y = labelencoding(data.data.targets.to_numpy().copy()).flatten()
    DATASETS[name] = (X, y)

# ── Synthetic datasets ──────────────────────────────────────
X, y = make_classification(1000, 3, n_informative=3, n_redundant=0, n_classes=4,
                           n_clusters_per_class=1, class_sep=1.5, random_state=42)
DATASETS["synth_sep"] = (X.astype(np.float64), y.astype(np.int64))

X, y = make_classification(1000, 3, n_informative=3, n_redundant=0, n_classes=4,
                           n_clusters_per_class=2, class_sep=0.6, random_state=43)
DATASETS["synth_over"] = (X.astype(np.float64), y.astype(np.int64))

X, y = make_classification(1000, 3, n_informative=3, n_redundant=0, n_classes=4,
                           n_clusters_per_class=1, class_sep=0.5, random_state=44)
Xc = np.zeros_like(X)
for f in range(3):
    Xc[:, f] = np.digitize(X[:, f], np.percentile(X[:, f], [0, 25, 50, 75, 100])[:-1]) - 1
DATASETS["synth_cat"] = (Xc.astype(np.float64), y.astype(np.int64))


# ── Tree strategies (optimized) ─────────────────────────────

def train_overfit(X, y, seed):
    return DecisionTreeClassifier(max_depth=10, min_samples_leaf=1,
                                  random_state=seed).fit(X, y)

def train_unconstrained(X, y, seed):
    """No max_depth but cap population per leaf = 2 to prevent explosion."""
    return DecisionTreeClassifier(min_samples_leaf=2, random_state=seed).fit(X, y)

def train_90pct(X, y, seed):
    for d in range(1, 21):
        tree = DecisionTreeClassifier(max_depth=d, min_samples_leaf=1, random_state=seed)
        tree.fit(X, y)
        if tree.score(X, y) >= 0.90:
            return tree
    return DecisionTreeClassifier(min_samples_leaf=2, random_state=seed).fit(X, y)

def train_pruned(X, y, seed):
    """Faster: use only a few candidate alphas."""
    base = DecisionTreeClassifier(min_samples_leaf=2, random_state=seed).fit(X, y)
    path = base.cost_complexity_pruning_path(X, y)
    alphas = path.ccp_alphas
    if len(alphas) <= 3:
        return base
    # Test only ~8 evenly spaced alphas
    indices = np.linspace(0, len(alphas) - 1, 8, dtype=int)
    candidates = alphas[indices]
    best_alpha = 0.0
    best_score = -1
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    for alpha in candidates:
        scores = []
        for tr, va in skf.split(X, y):
            t = DecisionTreeClassifier(ccp_alpha=alpha, random_state=seed)
            t.fit(X[tr], y[tr])
            scores.append(t.score(X[va], y[va]))
        if np.mean(scores) > best_score:
            best_score = np.mean(scores)
            best_alpha = alpha
    return DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=seed).fit(X, y)

STRATEGIES = {
    "overfit": train_overfit,
    "unconstrained": train_unconstrained,
    "accuracy_90": train_90pct,
    "pruned": train_pruned,
}


# ── Noise injection ─────────────────────────────────────────

def inject_independent(y, noise_level, rng, all_classes):
    y_noisy = y.copy()
    n_flip = int(len(y_noisy) * noise_level)
    if n_flip == 0:
        return y_noisy
    idx = rng.choice(len(y_noisy), size=n_flip, replace=False)
    for i in idx:
        others = [c for c in all_classes if c != y_noisy[i]]
        y_noisy[i] = rng.choice(others)
    return y_noisy

def inject_cumulative(y_clean, noise_fractions, rng, all_classes):
    cumulative = [y_clean.copy()]
    already_flipped = np.zeros(len(y_clean), dtype=bool)
    for i in range(1, len(noise_fractions)):
        additional_frac = ((noise_fractions[i] - noise_fractions[i-1])
                           / (1 - noise_fractions[i-1]))
        y_cur = cumulative[-1].copy()
        unflipped = np.where(~already_flipped)[0]
        n_to_flip = int(len(unflipped) * additional_frac)
        if n_to_flip > 0:
            idx = rng.choice(unflipped, size=n_to_flip, replace=False)
            for i_idx in idx:
                others = [c for c in all_classes if c != y_cur[i_idx]]
                y_cur[i_idx] = rng.choice(others)
            already_flipped[idx] = True
        cumulative.append(y_cur)
    return cumulative


# ── OPTICS density (vectorized) ─────────────────────────────

def optics_density(X_adv):
    """Mean cluster density using scipy pdist (vectorized)."""
    if len(X_adv) < 5:
        return np.nan, 0
    optics = OPTICS(min_samples=5, xi=0.05).fit(X_adv)
    labels = optics.labels_
    cluster_ids = sorted(set(labels) - {-1})
    n_clusters = len(cluster_ids)
    if n_clusters == 0:
        return np.nan, 0
    densities = []
    for cid in cluster_ids:
        idx = np.where(labels == cid)[0]
        if len(idx) < 2:
            continue
        points = X_adv[idx]
        total_dist = pdist(points).sum()
        n_pairs = len(idx) * (len(idx) - 1) // 2
        densities.append(n_pairs / (total_dist + 1))
    return np.mean(densities), n_clusters


# ── Main collection ─────────────────────────────────────────

rows = []
total = len(DATASETS) * N_RUNS * len(NOISE_LEVELS) * len(STRATEGIES) * 2
pbar = tqdm(total=total, desc="Collecting")

for ds_name, (X, y) in DATASETS.items():
    all_classes = np.unique(y)
    ds_hash = hash(ds_name) % 10000

    for run in range(N_RUNS):
        run_seed = BASE_SEED + run
        rng = np.random.default_rng(BASE_SEED + run * 1000 + ds_hash)

        X_train, X_test, y_train_clean, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=run_seed)

        cumul_y = inject_cumulative(y_train_clean, NOISE_LEVELS, rng, all_classes)

        for noise_mode in ["independent", "cumulative"]:
            for li, nl in enumerate(NOISE_LEVELS):
                if noise_mode == "independent":
                    rng2 = np.random.default_rng(BASE_SEED + run * 1000 + li * 100 + ds_hash)
                    y_train = inject_independent(y_train_clean, nl, rng2, all_classes)
                else:
                    y_train = cumul_y[li]

                for strat_name, train_fn in STRATEGIES.items():
                    row = dict(dataset=ds_name, noise_mode=noise_mode,
                               tree_strategy=strat_name, noise_level=float(nl),
                               run=run, tree_depth=0, n_leaves=0,
                               train_acc=np.nan, test_acc=np.nan,
                               adv_acc=np.nan, density=np.nan, n_clusters=0)

                    try:
                        tree = train_fn(X_train, y_train, run_seed)
                        row["tree_depth"] = tree.tree_.max_depth
                        row["n_leaves"] = tree.get_n_leaves()
                        row["train_acc"] = tree.score(X_train, y_train)
                        row["test_acc"] = tree.score(X_test, y_test)
                    except Exception:
                        rows.append(row); pbar.update(1)
                        continue

                    try:
                        classifier = ScikitlearnDecisionTreeClassifier(tree)
                        attack = DecisionTreeAttack(classifier=classifier)
                        x_adv = attack.generate(x=X_test)
                        row["adv_acc"] = tree.score(x_adv, y_test)
                        row["density"], row["n_clusters"] = optics_density(x_adv)
                    except Exception:
                        pass

                    rows.append(row)
                    pbar.update(1)

pbar.close()

df = pd.DataFrame(rows)
df.to_parquet("data/data_v2.parquet", index=False)
print(f"\nSaved {len(df)} rows to data/data_v2.parquet")
print(df.groupby(["dataset", "noise_mode", "tree_strategy"]).size().describe())
