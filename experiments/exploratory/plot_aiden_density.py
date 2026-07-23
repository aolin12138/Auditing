"""
Plot Aiden's ORIGINAL density metric (adv distance = count/(distance+1), buggy norm)
with z-scores and 95% CI, side-by-side with cluster count for comparison.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

df = pl.read_parquet("data/data_bias.parquet")
df = df.with_columns(
    pl.col("dataset").str.split("_").list.first().alias("base_dataset")
)
df = df.filter(pl.col("bias_type") == "undersampling")

# Z-score both metrics
df_z = df.with_columns(
    ((pl.col("adv distance") - pl.col("adv distance").mean())
     / pl.col("adv distance").std()).over(["base_dataset"]).alias("density_z"),
    ((pl.col("clusters") - pl.col("clusters").mean())
     / pl.col("clusters").std()).over(["base_dataset"]).alias("clusters_z"),
)

agg = df_z.group_by(["base_dataset", "bias"]).agg(
    pl.col("density_z").mean().alias("d_z_mean"),
    pl.col("density_z").std().alias("d_z_std"),
    pl.col("clusters_z").mean().alias("c_z_mean"),
    pl.col("clusters_z").std().alias("c_z_std"),
    pl.col("density_z").count().alias("n"),
    pl.col("adv distance").mean().alias("raw_density"),
    pl.col("clusters").mean().alias("raw_clusters"),
    pl.col("test acc").mean().alias("test_acc"),
    pl.col("train acc").mean().alias("train_acc"),
).sort(["base_dataset", "bias"])

# CIs
agg = agg.with_columns(
    pl.struct(["d_z_mean", "d_z_std", "n"]).map_elements(
        lambda r: r["d_z_mean"] - stats.t.ppf(0.975, r["n"]-1) * r["d_z_std"] / np.sqrt(r["n"]),
        return_dtype=pl.Float64).alias("d_ci_lb"),
    pl.struct(["d_z_mean", "d_z_std", "n"]).map_elements(
        lambda r: r["d_z_mean"] + stats.t.ppf(0.975, r["n"]-1) * r["d_z_std"] / np.sqrt(r["n"]),
        return_dtype=pl.Float64).alias("d_ci_ub"),
    pl.struct(["c_z_mean", "c_z_std", "n"]).map_elements(
        lambda r: r["c_z_mean"] - stats.t.ppf(0.975, r["n"]-1) * r["c_z_std"] / np.sqrt(r["n"]),
        return_dtype=pl.Float64).alias("c_ci_lb"),
    pl.struct(["c_z_mean", "c_z_std", "n"]).map_elements(
        lambda r: r["c_z_mean"] + stats.t.ppf(0.975, r["n"]-1) * r["c_z_std"] / np.sqrt(r["n"]),
        return_dtype=pl.Float64).alias("c_ci_ub"),
)

datasets = ["iris", "wine", "Car Evaluation"]
colors_d = "#800080"  # purple for density
colors_c = "#008080"  # teal for clusters

# ── Plot 1: Density z-score + accuracy ─────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, ds in zip(axes, datasets):
    d = agg.filter(pl.col("base_dataset") == ds).sort("bias")
    x = d["bias"].to_numpy()

    ax.plot(x, d["train_acc"], "o-", color="#1f77b4", lw=2, ms=5, label="Train Acc")
    ax.plot(x, d["test_acc"], "s-", color="#d62728", lw=2, ms=5, label="Test Acc")
    ax.set_ylabel("Accuracy", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    if ds == "Car Evaluation":
        ax.set_ylim(0.3, 1.05)
    else:
        ax.set_ylim(0.7, 1.05)

    ax2 = ax.twinx()
    ax2.fill_between(x, d["d_ci_lb"], d["d_ci_ub"], color=colors_d, alpha=0.12)
    ax2.plot(x, d["d_z_mean"], "o--", color=colors_d, lw=2, ms=5, label="Density (z)")
    ax2.axhline(y=0, color="gray", ls=":", lw=1, alpha=0.5)
    ax2.set_ylabel("Density (z-score)", color=colors_d)
    ax2.tick_params(axis="y", labelcolor=colors_d)

    ax.set_xlabel("Bias Level")
    ax.set_title(ds + f"\nraw: {d['raw_density'][0]:.4f}→{d['raw_density'][-1]:.4f}", fontweight="bold")
    ax.set_xlim(0.05, 0.95)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=8)

fig.suptitle("Aiden's Original Density (adv distance, buggy Frobenius norm) vs Coverage Gap",
             fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig("experiments/plots/aiden_original_density.png", dpi=200, bbox_inches="tight")
print("Saved experiments/plots/aiden_original_density.png")

# ── Plot 2: Side-by-side — Density vs Cluster Count ────────
fig2, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, ds in zip(axes, datasets):
    d = agg.filter(pl.col("base_dataset") == ds).sort("bias")
    x = d["bias"].to_numpy()

    ax.fill_between(x, d["d_ci_lb"], d["d_ci_ub"], color=colors_d, alpha=0.12)
    ax.plot(x, d["d_z_mean"], "o-", color=colors_d, lw=2, ms=5, label="Density (z)")
    ax.fill_between(x, d["c_ci_lb"], d["c_ci_ub"], color=colors_c, alpha=0.12)
    ax.plot(x, d["c_z_mean"], "s--", color=colors_c, lw=2, ms=5, label="Clusters (z)")
    ax.axhline(y=0, color="gray", ls=":", lw=1, alpha=0.5)
    ax.set_xlabel("Bias Level")
    ax.set_ylabel("z-score")
    ax.set_title(ds, fontweight="bold")
    ax.set_xlim(0.05, 0.95)
    ax.legend(fontsize=9)

fig2.suptitle("Aiden's Original: Density (purple) vs Cluster Count (teal)\nboth z-scored, 95% CI bands",
              fontsize=14, fontweight="bold")
fig2.tight_layout()
fig2.savefig("experiments/plots/aiden_original_comparison.png", dpi=200, bbox_inches="tight")
print("Saved experiments/plots/aiden_original_comparison.png")
