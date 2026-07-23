"""
Generate comparison plots for label noise experiment.
1. Faceted density-vs-noise plot (Altair, matching original style)
2. Data distribution plots (matplotlib, 2D projections)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl
import altair as alt
import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from util import labelencoding


# ═══ Plot 1: Density vs noise (faceted, Altair) ═══

def make_density_plot():
    noise = pl.read_parquet("data/data_label_noise.parquet")

    # Z-score adv_distance within each dataset
    df_z = noise.with_columns(
        ((pl.col("adv distance") - pl.col("adv distance").mean())
         / pl.col("adv distance").std())
        .over(["dataset"]).alias("adv_distance_z")
    )

    # Aggregate by dataset + noise level
    agg = df_z.group_by(["dataset", "defect_level"]).agg(
        pl.col("adv_distance_z").mean().alias("adv_distance_z_mean"),
        pl.col("test acc").mean().alias("test_acc_mean"),
    )

    # Clean dataset names for display
    agg = agg.with_columns(
        pl.col("dataset").str.replace("_feature", " f").alias("panel"),
        pl.col("defect_level").alias("noise_level"),
    )

    # Build Altair chart
    base = alt.Chart(agg.to_pandas()).encode(
        x=alt.X("noise_level:Q", title="Label Noise Level", scale=alt.Scale(domain=[0, 0.5])),
    )

    density_line = base.mark_line(color="#800080", strokeWidth=2).encode(
        y=alt.Y("adv_distance_z_mean:Q", title="Density (z-score)"),
    )

    density_points = base.mark_circle(color="#800080", size=30).encode(
        y=alt.Y("adv_distance_z_mean:Q"),
    )

    zero_rule = base.mark_rule(color="gray", strokeDash=[4, 4]).encode(
        y=alt.datum(0),
    )

    chart = (density_line + density_points + zero_rule).properties(
        width=180, height=150,
    ).facet(
        facet=alt.Facet("panel:N", title=None, sort=None),
        columns=4,
    ).resolve_scale(y="independent")

    chart.save("experiments/plots/label_noise_density_faceted.png", scale_factor=2)
    print("Saved experiments/plots/label_noise_density_faceted.png")


# ═══ Plot 2: Data distribution (matplotlib) ═══

def make_distribution_plot():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for ax, name in zip(axes, ["iris", "wine", "Car Evaluation"]):
        data = fetch_ucirepo(name)
        X = data.data.features.to_numpy()
        y = labelencoding(data.data.targets.to_numpy().copy()).flatten()

        feature_names = data.data.features.columns.tolist()
        f0, f1 = feature_names[0], feature_names[1]

        classes = np.unique(y)
        colors = plt.cm.Set1(np.linspace(0, 1, len(classes)))

        for c_idx, c in enumerate(classes):
            mask = y == c
            ax.scatter(X[mask, 0], X[mask, 1], c=[colors[c_idx]],
                       label=f"Class {c}", alpha=0.6, s=20, edgecolors="k", linewidth=0.3)

        ax.set_xlabel(f"{f0} (feature 0)")
        ax.set_ylabel(f"{f1} (feature 1)")
        ax.set_title(f"{name} (first 2 features)")
        ax.legend(fontsize=8)

    fig.suptitle("Data Distribution by Class (First 2 Features)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig("experiments/plots/data_distribution.png", dpi=150)
    print("Saved experiments/plots/data_distribution.png")


# ═══ Plot 3: Label noise effect on data points ═══

def make_noise_effect_plot():
    """Show what label noise does to the data: scatter with original vs flipped labels."""
    rng = np.random.default_rng(42)

    name = "iris"
    data = fetch_ucirepo(name)
    X = data.data.features.to_numpy()
    y = labelencoding(data.data.targets.to_numpy().copy()).flatten()

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    for ax, nl in zip(axes.flat, noise_levels):
        y_noisy = y.copy()
        n_flip = int(len(y_noisy) * nl)
        flip_idx = rng.choice(len(y_noisy), size=n_flip, replace=False)

        # Mark flipped
        flipped_mask = np.zeros(len(y), dtype=bool)
        flipped_mask[flip_idx] = True

        classes = np.unique(y)
        for c in classes:
            other = [cl for cl in classes if cl != c]
            mask_c = (y == c) & flipped_mask
            y_noisy[mask_c] = rng.choice(other, size=mask_c.sum())

        # Plot: correct labels in blue, flipped in red
        correct = ~flipped_mask
        ax.scatter(X[correct, 0], X[correct, 1], c="steelblue", alpha=0.5, s=15, label="Correct")
        ax.scatter(X[flipped_mask, 0], X[flipped_mask, 1], c="crimson", alpha=0.8, s=30,
                   marker="x", linewidth=1.5, label="Flipped")
        ax.set_title(f"Noise = {nl:.1f} ({n_flip}/{len(y)} flipped)")
        ax.set_xlabel("sepal length")
        ax.set_ylabel("sepal width")
        ax.legend(fontsize=7)

    fig.suptitle("Label Noise Effect on Iris (First 2 Features)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig("experiments/plots/label_noise_effect_iris.png", dpi=150)
    print("Saved experiments/plots/label_noise_effect_iris.png")


if __name__ == "__main__":
    make_density_plot()
    make_distribution_plot()
    make_noise_effect_plot()
    print("\nAll plots saved.")
