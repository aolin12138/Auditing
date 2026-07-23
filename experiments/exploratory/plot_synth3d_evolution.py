"""
3D noise evolution for synthetic datasets: show label distribution 
as noise increases (0.0 to 0.9), colored by assigned label.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

NOISE_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]

# Create datasets
datasets = {}
for name, sep in [("Well-Separated (sep=1.5)", 1.5), ("Overlapping (sep=0.7)", 0.7)]:
    X, y = make_classification(
        n_samples=500, n_features=3, n_informative=3, n_redundant=0,
        n_classes=4, n_clusters_per_class=1, class_sep=sep,
        random_state=42, flip_y=0.0,
    )
    datasets[name] = (X.astype(np.float64), y.astype(np.int64))

fig = plt.figure(figsize=(18, 11))
rng = np.random.default_rng(42)
colors = plt.cm.tab10(np.linspace(0, 1, 4))

for row, (name, (X, y)) in enumerate(datasets.items()):
    for col, nl in enumerate(NOISE_LEVELS):
        ax = fig.add_subplot(2, 6, row * 6 + col + 1, projection='3d')
        
        y_noisy = y.copy()
        n_flip = int(len(y) * nl)
        flip_idx = rng.choice(len(y), size=n_flip, replace=False)
        classes = np.unique(y)
        for c in classes:
            other = [cl for cl in classes if cl != c]
            mask = (y == c) & np.isin(np.arange(len(y)), flip_idx)
            y_noisy[mask] = rng.choice(other, size=mask.sum())
        
        # Plot colored by ASSIGNED (noisy) label
        for c in classes:
            mask = y_noisy == c
            ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2],
                       c=[colors[c]], alpha=0.6, s=8, edgecolors='none',
                       label=f"Class {c}" if col == 5 else "")
        
        ax.set_title(f"Noise={nl:.1f}", fontsize=9)
        ax.set_xlabel("f0", fontsize=6); ax.set_ylabel("f1", fontsize=6); ax.set_zlabel("f2", fontsize=6)
        ax.tick_params(labelsize=5)
        if col == 0:
            ax.text2D(-0.3, 0.5, name, transform=ax.transAxes, fontsize=9, 
                      fontweight='bold', rotation=90, va='center')

fig.suptitle("Synthetic 3D Datasets: Noisy Label Distribution\n(Color = assigned label after noise injection)",
             fontsize=12, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("experiments/plots/synth3d_noise_evolution.png", dpi=200, bbox_inches="tight")
print("Saved experiments/plots/synth3d_noise_evolution.png")
