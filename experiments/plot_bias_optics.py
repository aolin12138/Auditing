"""
Plot bias experiment with OPTICS-internal core_density metric.
Z-scored density + 95% CI + accuracy, matching Aiden's 2dp scale.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

df = pl.read_parquet("data/data_bias_optics_metric.parquet")

# Z-score core_density within each dataset
df_z = df.with_columns(
    ((pl.col("density_core") - pl.col("density_core").mean())
     / pl.col("density_core").std())
    .over(["dataset"]).alias("core_density_z")
)

# Aggregate by dataset & bias_level (averaging folds, seeds, classes, features)
agg = df_z.group_by(["dataset", "bias_level"]).agg(
    pl.col("core_density_z").mean().alias("z_mean"),
    pl.col("core_density_z").std().alias("z_std"),
    pl.col("core_density_z").count().alias("n"),
    pl.col("train_acc").mean().alias("train_acc"),
    pl.col("test_acc").mean().alias("test_acc"),
    # Also raw metrics for reference
    pl.col("density_core").mean().alias("raw_density"),
    pl.col("density_core").std().alias("raw_std"),
    pl.col("n_clusters").mean().alias("n_clusters"),
    pl.col("n_clusters").std().alias("n_clusters_std"),
).sort(["dataset", "bias_level"])

# Compute 95% CI
agg = agg.with_columns(
    pl.struct(["z_mean", "z_std", "n"]).map_elements(
        lambda r: r["z_mean"] - stats.t.ppf(0.975, r["n"]-1) * r["z_std"] / np.sqrt(r["n"]),
        return_dtype=pl.Float64
    ).alias("z_ci_lb"),
    pl.struct(["z_mean", "z_std", "n"]).map_elements(
        lambda r: r["z_mean"] + stats.t.ppf(0.975, r["n"]-1) * r["z_std"] / np.sqrt(r["n"]),
        return_dtype=pl.Float64
    ).alias("z_ci_ub"),
)

datasets = agg["dataset"].unique().to_list()
n_ds = len(datasets)
fig, axes = plt.subplots(1, n_ds, figsize=(6 * n_ds, 5))
if n_ds == 1:
    axes = [axes]

for ax, ds in zip(axes, datasets):
    ds_data = agg.filter(pl.col("dataset") == ds).sort("bias_level")
    x = ds_data["bias_level"].to_numpy()
    z_mean = ds_data["z_mean"].to_numpy()
    z_lb = ds_data["z_ci_lb"].to_numpy()
    z_ub = ds_data["z_ci_ub"].to_numpy()
    train = ds_data["train_acc"].to_numpy()
    test = ds_data["test_acc"].to_numpy()

    # Left axis: accuracy
    ax.plot(x, train, "o-", color="#1f77b4", linewidth=2, markersize=5, label="Train Acc")
    ax.plot(x, test, "s-", color="#d62728", linewidth=2, markersize=5, label="Test Acc")
    ax.set_ylim(0.75, 1.05)
    ax.set_ylabel("Accuracy", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")

    # Right axis: core density z-score with CI
    ax2 = ax.twinx()
    ax2.fill_between(x, z_lb, z_ub, color="purple", alpha=0.12)
    ax2.plot(x, z_mean, "o--", color="purple", linewidth=2, markersize=5, label="Core Density (z)")
    ax2.axhline(y=0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax2.set_ylabel("Core Density (z-score)", color="purple")
    ax2.tick_params(axis="y", labelcolor="purple")

    ax.set_xlabel("Bias Level (coverage gap fraction)")
    ax.set_title(ds, fontweight="bold")
    ax.set_xlim(0.05, 0.95)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=8)

fig.suptitle("Coverage Gap Bias: Core Density & Accuracy (Aiden's original scale)", 
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("experiments/plots/bias_core_density.png", dpi=200, bbox_inches="tight")
print("Saved experiments/plots/bias_core_density.png")

# ── Also: cluster count vs bias ────────────────────────────
fig2, ax = plt.subplots(figsize=(8, 5))
for ds in datasets:
    ds_data = agg.filter(pl.col("dataset") == ds).sort("bias_level")
    x = ds_data["bias_level"].to_numpy()
    y = ds_data["n_clusters"].to_numpy()
    y_std = ds_data["n_clusters_std"].to_numpy()
    ax.errorbar(x, y, yerr=y_std, fmt="o-", capsize=3, linewidth=2, markersize=5, label=ds)

ax.set_xlabel("Bias Level (coverage gap fraction)")
ax.set_ylabel("OPTICS Cluster Count")
ax.set_title("Cluster Count vs Coverage Gap (Aiden's original signal)", fontweight="bold")
ax.set_xlim(0.05, 0.95)
ax.legend()
ax.axhline(y=min(y) - 0.1, color="gray", linestyle=":", alpha=0.5)
fig2.tight_layout()
fig2.savefig("experiments/plots/bias_cluster_count.png", dpi=200, bbox_inches="tight")
print("Saved experiments/plots/bias_cluster_count.png")

# ── Print summary table ────────────────────────────────────
print("\n=== RAW METRICS (2dp for comparison with Aiden's scale) ===")
print(f"{'Dataset':>15s} {'Bias':>5s} | {'n_clust':>8s} {'core_dens':>10s} {'test_acc':>8s} {'train_acc':>8s}")
print("-" * 65)
for ds in datasets:
    ds_data = agg.filter(pl.col("dataset") == ds).sort("bias_level")
    for row in ds_data.iter_rows(named=True):
        print(f"{ds:>15s} {row['bias_level']:>5.1f} | "
              f"{row['n_clusters']:>8.2f} {row['raw_density']:>10.4f} "
              f"{row['test_acc']:>8.3f} {row['train_acc']:>8.3f}")
