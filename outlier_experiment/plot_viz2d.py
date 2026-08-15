"""2D visualisation: what DOES the overfit tree actually do when a 'toward' outlier is added?
Uses iris petal length x petal width (feat 2,3). Renders the actual decision regions (axis-aligned
rectangles), the outlier, and adversarial landing points, for 3 regimes:
  1. clean (no outlier)
  2. k=4 (outlier inside versicolor cluster — the DIP)
  3. k=8 (outlier past versicolor, in empty space — spread FIRES)
-> plots/viz2d_outlier_mechanism.png
"""
import os
for _v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_v] = '1'
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, polars as pl
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import DecisionTreeAttack
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PLOTS = HERE / 'plots'

X, y = load_iris(return_X_y=True)
FEATS = [2, 3]                                     # petal length, petal width
X2, y2 = X[:, FEATS], y
TC = 2; K = 8; N_OUT = 10; DIR = 'toward'; SEED = 42; NOC = 1  # versicolor
names = ['setosa', 'versicolor', 'virginica']
col_map = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c'}

rng = np.random.default_rng(SEED)
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
tr, te = next(skf.split(X2, y2))
X_tr, y_tr, X_te, y_te = X2[tr], y2[tr], X2[te], y2[te]

# ---- inject outliers (copied from run_experiment.py) ----
def inject(X_tr2, y_tr2, k):
    if k == 0: return X_tr2.copy(), y_tr2.copy()
    Xc = X_tr2[y_tr2 == TC]; mu_c = Xc.mean(0)
    s_ = float(np.std(Xc, axis=0).mean())
    mu_o = X_tr2[y_tr2 == NOC].mean(0)
    d = mu_o - mu_c; d = d / (np.linalg.norm(d) + 1e-12)
    pts = [mu_c + k * s_ * d + rng.normal(0, 0.3, size=mu_c.shape) * np.std(Xc, axis=0)
           for _ in range(N_OUT)]
    Xo = np.vstack(pts); yo = np.full(N_OUT, TC)
    return np.vstack([X_tr2, Xo]), np.hstack([y_tr2, yo])


# ---- train and attack for each regime ----
fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))

for ax_i, (k, label) in enumerate(zip([0, 4, 8], ['clean (k=0)', 'k=4 (inside versicolor cluster)', 'k=8 (far through versicolor)'])):
    ax = axes[ax_i]
    Xa, ya = inject(X_tr, y_tr, k)
    m = DecisionTreeClassifier(max_depth=None, random_state=42).fit(Xa, ya)

    # -- draw decision regions as a mesh --
    x0 = np.linspace(X_tr[:, 0].min() - 0.5, max(X_tr[:, 0].max(), Xa[:, 0].max()) + 0.5, 200)
    x1 = np.linspace(X_tr[:, 1].min() - 0.5, max(X_tr[:, 1].max(), Xa[:, 1].max()) + 0.5, 200)
    xx, yy = np.meshgrid(x0, x1)
    grid_pred = m.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.pcolormesh(xx, yy, grid_pred, alpha=0.18, cmap=matplotlib.colors.ListedColormap(
        [col_map[0], col_map[1], col_map[2]]))

    # -- train points + outliers --
    for cl in [0, 1, 2]:
        mask_tr = ya == cl
        ax.scatter(Xa[mask_tr, 0], Xa[mask_tr, 1], s=22, c=col_map[cl], edgecolors='k', linewidths=0.2,
                   alpha=0.85, label=f'{names[cl]} (train)' if ax_i == 0 else '')
    if k > 0:
        outlier_mask = ya == TC
        is_outlier = np.zeros(len(Xa), dtype=bool)
        is_outlier[-N_OUT:] = True
        ax.scatter(Xa[is_outlier, 0], Xa[is_outlier, 1], s=70, marker='X', c='red', edgecolors='black',
                   linewidths=0.8, zorder=10, label='outliers' if ax_i == 0 else '')

    # -- attack correctly-classified test points --
    art = SklearnClassifier(m)
    yoh = np.eye(3)[ya]
    art.fit(Xa, yoh)
    pte = np.argmax(art.predict(X_te), axis=1)
    correct = pte == y_te
    if correct.sum():
        adv = DecisionTreeAttack(classifier=art).generate(X_te[correct])
        ap = np.argmax(art.predict(adv), axis=1)
        flipped = ap != y_te[correct]
        if flipped.sum():
            ax.scatter(adv[flipped, 0], adv[flipped, 1], s=16, c='purple', marker='^', alpha=0.55,
                       edgecolors='none', label='adversarial (flipped)' if ax_i == 0 else '')
        # also show test points
        for cl in [0, 1, 2]:
            mask_te = y_te == cl
            ax.scatter(X_te[mask_te, 0], X_te[mask_te, 1], s=12, c=col_map[cl], marker='D', alpha=0.5,
                       edgecolors='none', label=f'{names[cl]} (test)' if ax_i == 0 else '')

    # mark the outlier ray direction
    if k > 0:
        Xc = X_tr[y_tr == TC]; mu_c = Xc.mean(0)
        mu_o = X_tr[y_tr == NOC].mean(0)
        ax.annotate('', xy=mu_o, xytext=mu_c, arrowprops=dict(arrowstyle='->', color='gray', lw=1.2, alpha=0.6))
        ax.text((mu_c[0] + mu_o[0]) / 2, (mu_c[1] + mu_o[1]) / 2, 'toward', fontsize=7, color='gray', ha='center')

    ax.set_title(label, fontsize=10)
    ax.set_xlabel('petal length (cm)'); ax.set_ylabel('petal width (cm)')
    if ax_i == 0:
        ax.legend(fontsize=6.5, loc='upper left', ncol=2, markerscale=0.7)

fig.suptitle('What the overfit tree ACTUALLY does: the outlier warps the axis-aligned '
             'decision regions, and adversarial examples (purple triangles)\nspread further '
             'apart when the outlier is IN EMPTY SPACE (right) vs inside the cluster (middle)',
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.91])
fig.savefig(PLOTS / 'viz2d_outlier_mechanism.png', dpi=150)
print('wrote', PLOTS / 'viz2d_outlier_mechanism.png')
