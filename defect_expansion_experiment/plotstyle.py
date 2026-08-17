"""Shared plotting style for the defect-expansion experiments (submission-grade, reusable).

Fixes flagged in `.wiki/10-tooling-and-skills.md` (from the `nature-figure` skill review):
  - colorblind-safe palette (Okabe-Ito, Wong 2011 Nature Methods) instead of red/green
  - differentiate series by BOTH colour and linestyle/marker (survives greyscale)
  - vector (PDF) export alongside PNG, with EDITABLE text in the vector (pdf.fonttype=42,
    svg.fonttype='none') for the Part IV report
  - top/right spines off, readable font-size floor

Usage in any plot_*.py:
    import plotstyle as ps
    ps.apply()
    ... color=ps.DEFECT[s], ls=ps.DEFECT_LS[s], marker=ps.DEFECT_MK[s] ...
    ps.panel_label(ax, 'a')
    ps.save(fig, PLOTS / 'name')          # writes name.png + name.pdf
"""
import matplotlib as mpl
from pathlib import Path

# Okabe-Ito colorblind-safe palette (distinguishable under all common CVD types)
OKABE_ITO = {
    'black': '#000000', 'orange': '#E69F00', 'skyblue': '#56B4E9', 'green': '#009E73',
    'yellow': '#F0E442', 'blue': '#0072B2', 'vermillion': '#D55E00', 'purple': '#CC79A7',
}

# Semantic mapping for the defect study (replaces the old red/green = NOT colorblind-safe)
DEFECT = {'spatial': OKABE_ITO['vermillion'], 'random': OKABE_ITO['blue']}   # coverage gap / imbalance
DEFECT_LS = {'spatial': '-', 'random': '--'}
DEFECT_MK = {'spatial': 'o', 'random': 's'}
DEFECT_LAB = {'spatial': 'spatial (coverage gap)', 'random': 'random (imbalance)'}

# General categorical palette (datasets, models, classes) — colorblind-safe order
CATEGORICAL = [OKABE_ITO['blue'], OKABE_ITO['vermillion'], OKABE_ITO['green'],
               OKABE_ITO['purple'], OKABE_ITO['orange'], OKABE_ITO['skyblue']]


def apply():
    mpl.rcParams.update({
        'figure.dpi': 110, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.titlesize': 10.5, 'axes.labelsize': 10,
        'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 8.5,
        'axes.spines.top': False, 'axes.spines.right': False,
        'lines.linewidth': 2.2, 'lines.markersize': 5.5,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',   # editable text in vector output
    })


def save(fig, path_stem, formats=('png', 'pdf')):
    """Save PNG (raster preview) + PDF (vector, editable text for the report). path_stem = no ext."""
    p = Path(path_stem); out = []
    for f in formats:
        q = p.with_suffix('.' + f); fig.savefig(q); out.append(q)
    return out


def panel_label(ax, letter, x=-0.11, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=12, fontweight='bold', va='bottom', ha='right')
