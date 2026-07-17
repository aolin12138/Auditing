"""
Plot Aiden's ORIGINAL bias experiment data with z-scores and 95% CI.
Validates whether the original experiment actually revealed a trend.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

df = pl.read_parquet("data/data_bias.parquet")

# Map dataset_feature to base dataset name
df = df.with_columns(
    pl.col("dataset").str.split("_").list.first().alias("base_dataset")
)

# Filter to undersampling only for cleaner comparison
df = df.filter(pl.col("bias_type") == "undersampling")

# Z-score cluster count within each base dataset
df_z = df.with_columns(
    ((pl.col("clusters") - pl.col("clusters").mean())
     / pl.col("clusters").std())
    .over(["base_dataset"]).alias("clusters_z")
)

# Aggregate
agg = df_z.group_by(["base_dataset", "bias"]).agg(
    pl.col("clusters_z").mean().alias("z_mean"),
    pl.col("clusters_z").std().alias("z_std"),
    pl.col("clusters_z").count().alias("n"),
    pl.col("clusters").mean().alias("clusters_raw"),
    pl.col("clusters").std().alias("clusters_std"),
    pl.col("adv distance").mean().alias("adv_dist"),
    pl.col("test acc").mean().alias("test_acc"),
    pl.col("train acc").mean().alias("train_acc"),
).sort(["base_dataset", "bias"])

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

datasets = ["iris", "wine", "Car Evaluation"]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, ds in zip(axes, datasets):
    ds_data = agg.filter(pl.col("base_dataset") == ds).sort("bias")
    x = ds_data["bias"].to_numpy()
    z_mean = ds_data["z_mean"].to_numpy()
    z_lb = ds_data["z_ci_lb"].to_numpy()
    z_ub = ds_data["z_ci_ub"].to_numpy()
    train = ds_data["train_acc"].to_numpy()
    test = ds_data["test_acc"].to_numpy()

    # Left axis: accuracy
    ax.plot(x, train, "o-", color="#1f77b4", linewidth=2, markersize=5, label="Train Acc")
    ax.plot(x, test, "s-", color="#d62728", linewidth=2, markersize=5, label="Test Acc")
    ax.set_ylabel("Accuracy", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    
    # Set reasonable y limits per dataset
    if ds == "Car Evaluation":
        ax.set_ylim(0.55, 1.05)
    else:
        ax.set_ylim(0.75, 1.05)

    # Right axis: cluster count z-score with CI
    ax2 = ax.twinx()
    ax2.fill_between(x, z_lb, z_ub, color="purple", alpha=0.12)
    ax2.plot(x, z_mean, "o--", color="purple", linewidth=2, markersize=5, label="Clusters (z)")
    ax2.axhline(y=0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax2.set_ylabel("Cluster Count (z-score)", color="purple")
    ax2.tick_params(axis="y", labelcolor="purple")

    ax.set_xlabel("Bias Level (coverage gap)")
    ax.set_title(ds, fontweight="bold", fontsize=12)
    ax.set_xlim(0.05, 0.95)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=8)

fig.suptitle("Aiden's Original Bias Experiment: Cluster Count vs Coverage Gap\n(undersampling only, all features, classes, seeds)",
             fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig("experiments/plots/aiden_original_bias_clusters.png", dpi=200, bbox_inches="tight")
print("Saved experiments/plots/aiden_original_bias_clusters.png")

# ── Also: raw cluster count with error bars ───────────────
fig2, ax = plt.subplots(figsize=(14, 5))
for ds in datasets:
    ds_data = agg.filter(pl.col("base_dataset") == ds).sort("bias")
    x = ds_data["bias"].to_numpy()
    y = ds_data["clusters_raw"].to_numpy()
    y_std = ds_data["clusters_std"].to_numpy()
    ax.errorbar(x, y, yerr=y_std, fmt="o-", capsize=3, linewidth=2, markersize=5, label=ds)

ax.set_xlabel("Bias Level (coverage gap fraction)")
ax.set_ylabel("OPTICS Cluster Count (raw)")
ax.set_title("Aiden's Original: Raw Cluster Count vs Coverage Gap (undersampling only)", fontweight="bold")
ax.set_xlim(0.05, 0.95)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig("experiments/plots/aiden_original_bias_raw.png", dpi=200, bbox_inches="tight")
print("Saved experiments/plots/aiden_original_bias_raw.png")

# ── Print summary ──────────────────────────────────────────
print("\n=== AIDEN'S ORIGINAL DATA (undersampling, aggregated) ===")
print(f"{'Dataset':>15s} {'Bias':>5s} | {'n_clust':>8s} {'test_acc':>8s} {'train_acc':>8s}")
print("-" * 55)
for ds in datasets:
    ds_data = agg.filter(pl.col("base_dataset") == ds).sort("bias")
    for row in ds_data.iter_rows(named=True):
        print(f"{ds:>15s} {row['bias']:>5.2f} | "
              f"{row['clusters_raw']:>8.2f}±{row['clusters_std']:>6.2f} "
              f"{row['test_acc']:>8.3f} {row['train_acc']:>8.3f}")

# ── Statistical test ───────────────────────────────────────
print("\n=== PAIRED T-TEST (clusters, bias 0.1 vs 0.9) ===")
for ds in datasets:
    d1 = df.filter((pl.col("base_dataset") == ds) & (pl.col("bias") == 0.1))
    d9 = df.filter((pl.col("base_dataset") == ds) & (pl.col("bias") == 0.9))
    # Match by seed, distribution, feature (extract from dataset string)
    if d1.shape[0] > 0 and d9.shape[0] > 0:
        # Aggregate per unique seed/distribution/feature combo
        g1 = d1.group_by(["seed", "distribution", "dataset"]).agg(pl.col("clusters").mean())
        g9 = d9.group_by(["seed", "distribution", "dataset"]).agg(pl.col("clusters").mean())
        # Simple independent test since we can't pair easily
        t, p = stats.ttest_ind(g1["clusters"].to_numpy(), g9["clusters"].to_numpy())
        print(f"  {ds:>15s}: d1={g1['clusters'].mean():.3f}, d9={g9['clusters'].mean():.3f}, "
              f"t={t:.3f}, p={p:.4f}")
