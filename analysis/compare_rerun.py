"""Compare rerun outputs against the pre-rerun snapshot.

Usage: python analysis/compare_rerun.py rerun/2026-10-15 results_archive/pre-rerun-2026-10-15 [names...]
Reports, per result file: row/column match, exact-equality of every value (NaN == NaN), and the
largest absolute difference in numeric columns when not exact. Sorts both frames on all non-float
key columns first so row order does not matter.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

rerun, snap = Path(sys.argv[1]), Path(sys.argv[2])
names = sys.argv[3:]
files = sorted(rerun.glob("*/*.parquet"))
for f in files:
    if names and f.stem.replace("results_", "") not in names:
        continue
    rel = f.relative_to(rerun)
    a, b = pd.read_parquet(f), pd.read_parquet(snap / rel)
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        print(f"{rel}: SHAPE/COLUMNS DIFFER rerun={a.shape} snap={b.shape}")
        continue
    keys = [c for c in a.columns if not pd.api.types.is_float_dtype(a[c])]
    a = a.sort_values(keys, kind="mergesort").reset_index(drop=True)
    b = b.sort_values(keys, kind="mergesort").reset_index(drop=True)
    worst, nexact = 0.0, 0
    for c in a.columns:
        x, y = a[c], b[c]
        if pd.api.types.is_float_dtype(x):
            xv, yv = x.to_numpy(float), y.to_numpy(float)
            same = (xv == yv) | (np.isnan(xv) & np.isnan(yv))
            if not same.all():
                worst = max(worst, float(np.nanmax(np.abs(xv - yv))))
                nexact += int((~same).sum())
        elif not x.equals(y):
            nexact += int((x != y).sum())
    verdict = "IDENTICAL" if nexact == 0 else f"DIFFERS: {nexact} cells, max |diff| = {worst:.3g}"
    print(f"{rel} ({len(a)} rows): {verdict}")
