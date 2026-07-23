"""
Comparison plot: label noise density + accuracy with CI bands.
Matching original visualization_bias style (dual y-axis, faceted).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load data
noise = pl.read_parquet("data/data_label_noise.parquet")

# Z-score adv_distance within each dataset
df_z = noise.with_columns(
    ((pl.col("adv distance") - pl.col("adv distance").mean())
     / pl.col("adv distance").std())
    .over(["dataset"]).alias("adv_distance_z")
)

# Aggregate
agg = df_z.group_by(["dataset", "defect_level"]).agg(
    pl.col("adv_distance_z").mean().alias("z_mean"),
    pl.col("adv_distance_z").std().alias("z_std"),
    pl.col("adv_distance_z").count().alias("n"),
    pl.col("train acc").mean().alias("train_acc"),
    pl.col("test acc").mean().alias("test_acc"),
).sort(["dataset", "defect_level"])

# Compute CI
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
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, ds in zip(axes, datasets):
    ds_data = agg.filter(pl.col("dataset") == ds).sort("defect_level")
    x = ds_data["defect_level"].to_numpy()
    z_mean = ds_data["z_mean"].to_numpy()
    z_lb = ds_data["z_ci_lb"].to_numpy()
    z_ub = ds_data["z_ci_ub"].to_numpy()
    train = ds_data["train_acc"].to_numpy()
    test = ds_data["test_acc"].to_numpy()

    # Left axis: accuracy
    ax.plot(x, train, "o-", color="#1f77b4", linewidth=2, markersize=5, label="Train Acc")
    ax.plot(x, test, "s-", color="#d62728", linewidth=2, markersize=5, label="Test Acc")
    ax.set_ylim(0.15, 1.05)
    ax.set_ylabel("Accuracy", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")

    # Right axis: density with CI
    ax2 = ax.twinx()
    ax2.fill_between(x, z_lb, z_ub, color="purple", alpha=0.12)
    ax2.plot(x, z_mean, "o--", color="purple", linewidth=2, markersize=5, label="Density (z)")
    ax2.axhline(y=0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax2.set_ylabel("Density (z-score)", color="purple")
    ax2.tick_params(axis="y", labelcolor="purple")

    ax.set_xlabel("Label Noise Level")
    ax.set_title(ds, fontweight="bold")
    ax.set_xlim(-0.02, 0.92)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=8)

fig.suptitle("Label Noise: Density & Accuracy vs Noise Level", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("experiments/plots/label_noise_full_comparison.png", dpi=200, bbox_inches="tight")
print("Saved experiments/plots/label_noise_full_comparison.png")
