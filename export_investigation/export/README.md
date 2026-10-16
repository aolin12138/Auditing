# MLP experiment — exported results

Two CSVs, exported 2026-09-02 from `../results.parquet`.

| file | rows | what |
|---|---|---|
| `mlp_results_all.csv` | 1600 | the complete MLP grid — coverage gap (both protocols) + every outlier arm |
| `placebo_subset.csv` | 500 | just the placebo comparison: the `k=2` arms plus the clean baseline |

**There is no input data file, and you don't need one.** Iris ships with scikit-learn
(`sklearn.datasets.load_iris`). Every defect is injected in code from a seed, so the whole grid
regenerates from the runner alone.

## Re-running it

```bash
git clone <repo> && cd Auditing
uv sync                                   # Python >=3.12; creates .venv from uv.lock
uv run python mlp_experiment/run_experiment.py --smoke      # pipeline check, ~1 min
uv run python mlp_experiment/run_experiment.py --placebo --defect outlier
```

Delete `mlp_experiment/results.parquet` first for a clean run — the runner is **resumable** and
skips any row already present. `--placebo` alone is 80 cells, ~4 min on an M-series laptop.
Live progress lands in `mlp_experiment/progress.txt`.

Other arms: `--defect coverage_gap`, `--defect outlier`, `--ceiling`, `--dose`, or no flag for
everything.

Analysis and the figure:

```bash
uv run python mlp_experiment/plot_placebo.py     # -> plots/p1_placebo_control.png + the table
uv run python mlp_experiment/analyze.py          # the full H1-H7 printout
# or open mlp_experiment/analysis.ipynb and Run All
```

**Reproducibility.** Seeds are fixed (`SEEDS = [42, 58, 125, 7, 13, 21, 99, 123, 200, 777]`), the
torch model is seeded per fit (`torch.manual_seed(42)`), and DeepFool is deterministic — those rows
reproduce exactly. HopSkipJump is Monte-Carlo and will differ slightly run to run; the paired
per-seed normalisation is what keeps that stable at the reported precision.

## The three arms in `placebo_subset.csv`

All at `k=2.0`, injected into the **training fold only**, 10 seeds x 5 folds.

| `direction` | `kind` | label given | what it is |
|---|---|---|---|
| `toward` | `correct` | virginica | contested band, **clean** label (71% of nearest real points are virginica) |
| `toward` | `mislabeled` | versicolor | contested band, **wrong** label — the true contamination |
| `outward` | `correct` | virginica | far field, matched count — position/count control |
| — | — | — | `n_out == 0` rows are the clean baseline every ratio divides by |

⚠️ The `kind` values come from `inject_outliers` and read backwards here: the arm ART calls
`mislabeled` is the genuinely contaminated one, but the arm called `correct` is the one earlier
write-ups describe as "contamination". See `../FINDINGS_placebo.md` P0.

## Reproducing the headline number

```python
import polars as pl
d = pl.read_parquet('results.parquet')        # or read_csv the export
ol = d.filter(pl.col('defect') == 'outlier')

# every ratio is paired to ITS OWN seed's clean baseline — that pairing is why 10 seeds suffice
base = (ol.filter(pl.col('n_out') == 0)
          .group_by(['attack', 'seed']).agg(pl.col('mean_dist').mean().alias('_b')))
n = (ol.filter(pl.col('n_out') > 0).join(base, on=['attack', 'seed'])
       .with_columns((pl.col('mean_dist') / pl.col('_b')).alias('norm')))

cell = n.filter((pl.col('k') == 2.0) & (pl.col('n_out') == 10) &
                (pl.col('attack') == 'deepfool') & (pl.col('direction') == 'toward'))
print(cell.group_by('kind').agg(pl.col('norm').mean()))
#   correct    -> 0.85   (compresses)
#   mislabeled -> 1.11   (expands)
```

## Column key

Full glossary in `../../CONTEXT.md`. The ones that matter here:

| column | meaning |
|---|---|
| `mean_dist` | **the metric** — OPTICS within-cluster mean pairwise distance, in raw iris cm |
| `spread_std_space` | same thing before inverse-transform, kept as a control |
| `nclust` / `clust_size` | clusters found / mean points per cluster — flat = compression is real, not OPTICS re-cutting |
| `vacc` / `min_recall` | test accuracy / recall on the target class (`tc=2`, virginica) |
| `asucc` / `nadv` | attack success rate / number of successful adversarials per fold |
| `k` / `n_out` / `direction` / `kind` | the injection: distance, count, direction, label |
| `bias` / `protocol` | coverage-gap rows only — severity, and `before_split` vs `train_only` |
| `perturbation` | mean L2 distance each point travelled (not the same as spread) |
| `density`, `aiden_density` | earlier metric variants, kept for continuity — **don't use these** |
| `seed` | one of the 10 split seeds; the unit of statistical replication |

Each row is one (cell, attack) pair already averaged over the 5 folds. `attack` is `deepfool`
(white-box) or `hsj` (black-box) — **both attack the same fitted model per fold**, so the
comparison is not confounded by different weights.

Caveats worth passing on with the data: everything is iris (150 rows, 4 features), clusters
average ~5.7 points, and `k` is measured in a blended isotropic std that is a poor ruler along the
injection direction — see `../FINDINGS_placebo.md` P0 and its "Next" section.
