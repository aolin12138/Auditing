"""Check that every decimal number in a report section exists in the analysis outputs.

Pool = value / CI bounds in analysis/results_numbers.csv and the per-cell CSVs written by figures.py,
each rounded to 0, 1 and 2 decimals (sign ignored). Numbers in the text that are not in the pool are
printed for manual review; they are either design constants (levels, seeds) or typing errors.

Usage: .venv/Scripts/python.exe analysis/check_text_numbers.py report/sections/04_results.tex [...]
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
pool = set()


def add(x):
    if pd.isna(x):
        return
    for nd in (0, 1, 2):
        pool.add(f"{abs(round(float(x), nd)):.{nd}f}")


r = pd.read_csv(ROOT / "analysis" / "results_numbers.csv")
for c in ("value", "ci_lo", "ci_hi"):
    r[c].apply(add)
for f in ("panel_table_values.csv", "sensitivity_table_values.csv"):
    t = pd.read_csv(ROOT / "analysis" / f)
    for c in ("d", "lo", "hi"):
        t[c].apply(add)

for path in sys.argv[1:]:
    tex = Path(path).read_text(encoding="utf-8")
    body = re.sub(r"\\(label|ref|includegraphics|input|cite)\{[^}]*\}", "", tex)
    nums = re.findall(r"(?<![\w.])(\d+\.\d+)", body)
    missing = sorted({n for n in nums if n not in pool}, key=float)
    print(f"{path}: {len(nums)} decimals; not found in the analysis outputs: {missing}")
