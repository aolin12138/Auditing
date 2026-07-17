"""
Replicate Aiden's bias (coverage gap) experiment exactly,
but swap the density metric to use OPTICS-internal core/reachability distances.

Aiden's original:
  - Delete a contiguous slab of one class along one feature (coverage gap)
  - Train DecisionTree(max_depth=10), attack with DecisionTreeAttack
  - OPTICS on adv points → pairwise distance density (broken Frobenius norm bug)
  - Record: clusters count, adv distance

This script:
  - Same bias injection, same model, same attack, same OPTICS
  - Computes BOTH old pairwise metric (fixed) AND new OPTICS-internal metrics:
    a) mean core distance per cluster (density = 1 / avg_core_dist)
    b) mean reachability distance per cluster (density = 1 / avg_reach_dist)
    c) max core distance in cluster (tightness metric)
  - Side-by-side comparison to see if new metric reveals the same trend
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import polars as pl
from ucimlrepo import fetch_ucirepo
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import OPTICS
from sklearn.metrics import accuracy_score, f1_score
from art.attacks.evasion import DecisionTreeAttack
from art.estimators.classification.scikitlearn import ScikitlearnDecisionTreeClassifier
from scipy.spatial.distance import pdist

from util import labelencoding

SEEDS = [42, 125, 58, 86, 138]
# Aiden's original scale: 9 levels from 0.1 to 0.9, step 0.1
BIAS_LEVELS = np.arange(0.1, 1.0, 0.1)
N_SPLITS = 10
DEPTH = 10
OPTICS_MIN_SAMPLES = 5


def eval_model_sklearn(model, X, y_true):
    """Direct sklearn evaluation (avoids ART wrappers for non-adv eval)."""
    y_pred = model.predict(X)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return acc, f1


def compute_old_pairwise_density(X_adv, labels):
    """Aiden's original metric (FIXED: norm(point1 - point2) instead of bug).
    Returns: per-cluster density values and overall mean."""
    cluster_ids = sorted(set(labels) - {-1})
    if len(cluster_ids) == 0:
        return [], 0.0
    densities = []
    for cid in cluster_ids:
        idx = np.where(labels == cid)[0]
        if len(idx) < 2:
            densities.append(0.0)
            continue
        points = X_adv[idx]
        total_dist = pdist(points).sum()
        n_pairs = len(idx) * (len(idx) - 1) // 2
        densities.append(n_pairs / (total_dist + 1))
    return densities, np.mean(densities)


def compute_optics_internal_metrics(optics, X_adv):
    """
    Compute density using OPTICS' own internal distance measures.
    
    Returns dict with:
      - density_core: 1 / mean_core_distance (per cluster, then averaged)
      - density_reach: 1 / mean_reachability_distance
      - max_core_dist: max core_distance in cluster (tightness proxy)
      - mean_core_dist: raw mean core distance
      - n_clusters: number of clusters found
    """
    labels = optics.labels_
    core_dists = optics.core_distances_
    reach_dists = optics.reachability_
    
    cluster_ids = sorted(set(labels) - {-1})
    n_clusters = len(cluster_ids)
    
    if n_clusters == 0:
        return dict(
            density_core=0.0, density_reach=0.0,
            max_core_dist=np.nan, mean_core_dist=np.nan,
            n_clusters=0
        )
    
    core_densities = []
    reach_densities = []
    max_cores = []
    mean_cores = []
    
    for cid in cluster_ids:
        mask = labels == cid
        cluster_core = core_dists[mask]
        cluster_reach = reach_dists[mask]
        
        # Remove inf reachability (can happen for boundary points)
        cluster_reach = cluster_reach[np.isfinite(cluster_reach)]
        
        if len(cluster_core) == 0:
            continue
        
        mean_c = cluster_core.mean()
        mean_cores.append(mean_c)
        max_cores.append(cluster_core.max())
        core_densities.append(1.0 / (mean_c + 1e-12))
        
        if len(cluster_reach) > 0:
            reach_densities.append(1.0 / (cluster_reach.mean() + 1e-12))
        else:
            reach_densities.append(0.0)
    
    return dict(
        density_core=np.mean(core_densities) if core_densities else 0.0,
        density_reach=np.mean(reach_densities) if reach_densities else 0.0,
        max_core_dist=np.mean(max_cores) if max_cores else np.nan,
        mean_core_dist=np.mean(mean_cores) if mean_cores else np.nan,
        n_clusters=n_clusters,
    )


def run_one_fold(X_train, y_train, X_test, y_test, 
                  bias_level, target_class, feature_idx, seed):
    """
    Replicate Aiden's exact bias injection + measurement for one fold.
    Returns dict of all metrics.
    """
    # ── Aiden's bias injection ─────────────────────────────
    # Sort the target class along the selected feature
    target_mask = y_train.squeeze(1) == target_class
    X_train_target = X_train[target_mask]
    
    if bias_level > 0 and X_train_target.shape[0] > 0:
        X_sort = np.sort(X_train_target[:, feature_idx])
        bias_idx = int(X_sort.shape[0] * bias_level)
        
        # Keep target-class points ABOVE the threshold (DELETE bottom slab)
        keep_mask = (X_train[:, feature_idx] > X_sort[bias_idx])[target_mask]
        X_train_b = X_train_target[keep_mask]
        y_train_b = y_train[target_mask][keep_mask]
    else:
        X_train_b = X_train_target
        y_train_b = y_train[target_mask]
    
    # Add all other classes
    for cls in np.unique(y_train):
        if cls != target_class:
            cls_mask = y_train.squeeze(1) == cls
            X_train_b = np.vstack((X_train_b, X_train[cls_mask]))
            y_train_b = np.vstack((y_train_b, y_train[cls_mask]))
    
    # ── Train tree ─────────────────────────────────────────
    tree = DecisionTreeClassifier(
        max_depth=DEPTH,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=seed,
    )
    tree.fit(X_train_b, y_train_b.ravel())
    
    # ── Eval ───────────────────────────────────────────────
    train_acc, train_f1 = eval_model_sklearn(tree, X_train, y_train.ravel())
    test_acc, test_f1 = eval_model_sklearn(tree, X_test, y_test.ravel())
    
    # ── Attack ─────────────────────────────────────────────
    classifier = ScikitlearnDecisionTreeClassifier(tree)
    attack = DecisionTreeAttack(classifier=classifier)
    
    try:
        x_adv = attack.generate(x=X_test)
    except Exception:
        return None  # attack failed
    
    adv_acc, adv_f1 = eval_model_sklearn(tree, x_adv, y_test.ravel())
    
    # ── OPTICS ─────────────────────────────────────────────
    if len(x_adv) < OPTICS_MIN_SAMPLES:
        return None
    
    optics = OPTICS(min_samples=OPTICS_MIN_SAMPLES, xi=0.05).fit(x_adv)
    labels = optics.labels_
    cluster_ids = sorted(set(labels) - {-1})
    
    # Old pairwise metric (fixed)
    old_densities, old_density_mean = compute_old_pairwise_density(x_adv, labels)
    
    # New OPTICS-internal metrics
    optics_metrics = compute_optics_internal_metrics(optics, x_adv)
    
    return dict(
        train_acc=train_acc, train_f1=train_f1,
        test_acc=test_acc, test_f1=test_f1,
        adv_acc=adv_acc, adv_f1=adv_f1,
        tree_leaves=tree.get_n_leaves(),
        tree_depth=tree.get_depth(),
        n_test=len(X_test),
        n_train_biased=len(X_train_b),
        bias_level=float(bias_level),
        target_class=int(target_class),
        feature=int(feature_idx),
        seed=int(seed),
        # Old metric (fixed bug)
        old_density=old_density_mean,
        old_density_std=np.std(old_densities) if len(old_densities) > 1 else 0.0,
        # New OPTICS-internal metrics
        n_clusters=optics_metrics["n_clusters"],
        density_core=optics_metrics["density_core"],
        density_reach=optics_metrics["density_reach"],
        max_core_dist=optics_metrics["max_core_dist"],
        mean_core_dist=optics_metrics["mean_core_dist"],
    )


def main():
    print("=" * 60)
    print("Replicating Aiden's bias experiment with OPTICS-internal metrics")
    print(f"Datasets: iris (subset test), seeds: {SEEDS}")
    print(f"Bias levels: {BIAS_LEVELS}, folds: {N_SPLITS}")
    print("=" * 60)
    
    all_rows = []
    
    for dataset_name in ["iris", "wine", "Car Evaluation"]:
        print(f"\n{'='*40}")
        print(f"Dataset: {dataset_name}")
        
        data = fetch_ucirepo(dataset_name)
        X = labelencoding(data.data.features.to_numpy().copy()).astype(np.float64)
        y = labelencoding(data.data.targets.to_numpy().copy())
        
        n_classes = len(np.unique(y))
        n_features = X.shape[1]
        print(f"  Shape: {X.shape}, Classes: {n_classes}, Features: {n_features}")
        
        for target_class in np.unique(y):
            tc = int(target_class)
            for feature in range(n_features):
                for seed in SEEDS:
                    skf = StratifiedKFold(
                        n_splits=N_SPLITS, shuffle=True, random_state=seed
                    )
                    
                    for bias_level in BIAS_LEVELS:
                        for train_idx, test_idx in skf.split(X, y):
                            X_train, y_train = X[train_idx], y[train_idx]
                            X_test, y_test = X[test_idx], y[test_idx]
                            
                            result = run_one_fold(
                                X_train, y_train, X_test, y_test,
                                bias_level=float(bias_level),
                                target_class=tc,
                                feature_idx=feature,
                                seed=seed,
                            )
                            
                            if result is not None:
                                result["dataset"] = dataset_name
                                all_rows.append(result)
    
    df = pl.DataFrame(all_rows)
    output_path = "data/data_bias_optics_metric.parquet"
    df.write_parquet(output_path)
    print(f"\nSaved {df.shape[0]} rows to {output_path}")
    
    for dataset_name in ["iris", "wine", "Car Evaluation"]:
        ds_df = df.filter(pl.col("dataset") == dataset_name)
        print(f"\n{'='*40}")
        print(f"Dataset: {dataset_name} ({ds_df.shape[0]} rows)")
        
        # ── Quick summary ─────────────────────────────────
        print(f"\n{'Bias':>6s}  {'n_clust':>8s}  {'old_density':>12s}  {'core_density':>13s}  {'reach_density':>13s}  {'max_core':>9s}  {'train_acc':>9s}  {'test_acc':>8s}")
        print("-" * 85)
        
        for bl in BIAS_LEVELS:
            sub = ds_df.filter(pl.col("bias_level") == bl)
            print(
                f"{bl:>5.1f}  "
                f"{sub['n_clusters'].mean():>8.2f}  "
                f"{sub['old_density'].mean():>12.4f}  "
                f"{sub['density_core'].mean():>13.4f}  "
                f"{sub['density_reach'].mean():>13.4f}  "
                f"{sub['max_core_dist'].mean():>9.4f}  "
                f"{sub['train_acc'].mean():>9.3f}  "
                f"{sub['test_acc'].mean():>8.3f}"
            )
        
        # ── Direction check ───────────────────────────────
        print(f"\n--- Direction (bias 0.1 vs 0.9) ---")
        d0 = ds_df.filter(pl.col("bias_level") == 0.1)
        d8 = ds_df.filter(pl.col("bias_level") == 0.9)
        
        for metric, d0_col, d8_col in [
            ("n_clusters", "n_clusters", "n_clusters"),
            ("old_density", "old_density", "old_density"),
            ("core_density", "density_core", "density_core"),
            ("reach_density", "density_reach", "density_reach"),
            ("max_core_dist", "max_core_dist", "max_core_dist"),
        ]:
            v0 = d0[d0_col].mean()
            v8 = d8[d8_col].mean()
            direction = "↑" if v8 > v0 else "↓" if v8 < v0 else "→"
            print(f"  {metric:>16s}: {v0:.4f} {direction} {v8:.4f}  (delta={v8-v0:+.4f})")


if __name__ == "__main__":
    main()
