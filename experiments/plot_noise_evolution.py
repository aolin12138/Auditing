"""
Show class distribution evolution under label noise (0.0 to 0.9).
Each point colored by its TRUE class. Flipped labels shown with × marker.
All 3 datasets in one figure.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from util import labelencoding

NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DATASETS = ["iris", "wine", "Car Evaluation"]

fig, axes = plt.subplots(len(DATASETS), len(NOISE_LEVELS),
                         figsize=(22, 7.5))
rng = np.random.default_rng(42)

for row, name in enumerate(DATASETS):
    data = fetch_ucirepo(name)
    X = data.data.features.to_numpy()
    y = labelencoding(data.data.targets.to_numpy().copy()).flatten()
    classes = np.unique(y)
    n_classes = len(classes)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_classes, 3)))

    feature_names = data.data.features.columns.tolist()

    for col, nl in enumerate(NOISE_LEVELS):
        ax = axes[row, col]
        y_noisy = y.copy()
        n_flip = int(len(y_noisy) * nl)
        flip_idx = rng.choice(len(y_noisy), size=n_flip, replace=False)
        flipped_mask = np.zeros(len(y), dtype=bool)
        flipped_mask[flip_idx] = True

        for c in classes:
            other = [cl for cl in classes if cl != c]
            mask_c = (y == c) & flipped_mask
            y_noisy[mask_c] = rng.choice(other, size=mask_c.sum())

        # Plot each point colored by its ASSIGNED (noisy) label, not true class
        for c_idx, c in enumerate(classes):
            mask = y_noisy == c
            ax.scatter(X[mask, 0], X[mask, 1],
                       c=[colors[c_idx]], marker='o', s=8, alpha=0.6,
                       edgecolors='none',
                       label=f"Class {c}" if col == len(NOISE_LEVELS) - 1 else "")

        ax.set_title(f"{nl:.1f}", fontsize=9)
        if row == len(DATASETS) - 1:
            ax.set_xlabel(feature_names[0][:12], fontsize=7)
        else:
            ax.set_xlabel("")
        ax.set_xticks([])
        if col == 0:
            ax.set_ylabel(f"{name}\n{feature_names[1][:12]}", fontsize=8)
        else:
            ax.set_yticks([])
            ax.set_ylabel("")

# Column title: noise level
for col, nl in enumerate(NOISE_LEVELS):
    axes[0, col].set_title(f"Noise = {nl:.1f}\n({int(nl*100)}% flipped)", fontsize=8,
                           fontweight='bold')

fig.suptitle("Noisy Label Distribution (What the Model Sees)\n"
             "Color = assigned label after noise injection",
             fontsize=12, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("experiments/plots/label_noise_class_evolution.png", dpi=200,
            bbox_inches="tight")
print("Saved experiments/plots/label_noise_class_evolution.png")
