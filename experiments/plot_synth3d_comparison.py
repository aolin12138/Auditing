"""
Density + accuracy comparison plot for synthetic 3D datasets.
Matching label_noise_full_comparison.png style.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

noise = pl.read_parquet("data/data_label_noise_synth3d.parquet")

# Z-score
df_z = noise.with_columns(
    ((pl.col("adv distance") - pl.col("adv distance").mean())
     / pl.col("adv distance").std())
    .over(["dataset"]).alias("z")
)

agg = df_z.group_by(["dataset", "defect_level"]).agg(
    pl.col("z").mean().alias("z_mean"),
    pl.col("z").std().alias("z_std"),
    pl.col("z").count().alias("n"),
    pl.col("train acc").mean().alias("train_acc"),
    pl.col("test acc").mean().alias("test_acc"),
).sort(["dataset", "defect_level"])

# CI
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
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, ds in zip(axes, datasets):
    d = agg.filter(pl.col("dataset") == ds).sort("defect_level")
    x = d["defect_level"].to_numpy()
    
    # Accuracy
    ax.plot(x, d["train_acc"], "o-", color="#1f77b4", lw=2, ms=5, label="Train Acc")
    ax.plot(x, d["test_acc"], "s-", color="#d62728", lw=2, ms=5, label="Test Acc")
    ax.set_ylim(0.1, 1.05)
    ax.set_ylabel("Accuracy", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    
    # Density
    ax2 = ax.twinx()
    ax2.fill_between(x, d["z_ci_lb"], d["z_ci_ub"], color="purple", alpha=0.12)
    ax2.plot(x, d["z_mean"], "o--", color="purple", lw=2, ms=5, label="Density (z)")
    ax2.axhline(y=0, color="gray", linestyle=":", lw=1, alpha=0.5)
    ax2.set_ylabel("Density (z-score)", color="purple")
    ax2.tick_params(axis="y", labelcolor="purple")
    
    ax.set_xlabel("Label Noise Level")
    ax.set_title(ds, fontweight="bold")
    ax.set_xlim(-0.02, 0.92)
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=8)

fig.suptitle("Synthetic 3D Datasets: Density & Accuracy vs Label Noise", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("experiments/plots/synth3d_full_comparison.png", dpi=200, bbox_inches="tight")
print("Saved experiments/plots/synth3d_full_comparison.png")
