# Investigation — "spread decreased during coverage gap" (MLP export)

**Question.** A peer's MLP experiment reportedly showed the **spread** metric *decreasing*
with coverage-gap bias — the opposite of the repo's established finding that clean spread
*rises* with bias. Did he run it correctly? Re-check the way the repo runs coverage gap.

**Data.** `export/mlp_results_all.csv` (1600 rows) — the peer's MLP grid. Coverage-gap subset:
220 rows, `tc=2`, `feat=2`, 10 seeds, bias ∈ {0,0.1,0.3,0.5,0.7,0.9}, attacks {deepfool, hsj},
two protocols {`before_split`, `train_only`}. No MLP runner was shared, so the experiment could
not be regenerated from scratch — instead the repo's coverage-gap **analysis method**
(`model_family_experiment/plot_cg.py`: filter `defect==coverage_gap`, group by `bias`, mean of
`mean_dist`, drop NaN) was re-applied to his exported results.

## Verdict: the data is fine; the *metric column* is the error

Using the metric the repo trusts — **`mean_dist` = clean spread** — with the repo's injection
protocol **`before_split`**, spread **RISES** with bias on the MLP, consistent with every prior
model+attack (Tree+DTA, Tree+HSJ, SVM+HSJ, RF+HSJ):

| bias | deepfool spread | hsj spread | test acc |
|---|---|---|---|
| 0.1 | 0.687 | 0.592 | 0.96 |
| 0.3 | 0.780 | 0.654 | 0.98 |
| 0.5 | 0.773 | 0.699 | 0.99 |
| 0.7 | 0.779 | 0.714 | 0.99 |
| 0.9 | 0.759 | 0.609 | 1.00 |
| **Cohen's d (0.1→0.9)** | **+1.40** | **+0.25** | flat/↑ |

The "decrease" the peer saw reproduces **only** on the density-form columns:

| column | trend across bias | Cohen's d |
|---|---|---|
| `mean_dist` (clean spread — **the metric to use**) | **rises** | **+1.40** |
| `density` = n/(mean_dist+1) (confounds cluster size) | **falls** | **−1.01** |
| `aiden_density` (original Frobenius-norm bug) | **falls** | **−0.54** |
| `spread_std_space` (spread, std space) | rises | +2.19 |

## Why the density columns fall while spread rises (mechanism, verified)

`density ≈ n_pairs / (mean_dist + 1)`. Cluster size stays ~flat across bias (`clust_size`
≈ 5.4–6.1), and `density*(mean_dist+1) ≈ 5.6` (≈ constant). So `density` is essentially
`const / (mean_dist + 1)` — it moves **inversely** to spread. As true spread rises with bias,
density mechanically falls. This is the exact inversion documented in `README.md` (finding #2),
`CONTEXT.md`, and the peer's own `export/README.md` ("`density`, `aiden_density` … **don't use
these**").

## Conclusion

- **He did not run it wrong — he read the wrong column.** "Spread decreased" is what you get from
  `density`/`aiden_density`, which are inverse to spread by construction. On the trusted metric
  (`mean_dist`, `before_split`), MLP spread **increases** with coverage-gap bias — the coverage-gap
  signal holds on the MLP too.
- Secondary noise, not a real decrease: the very last point (bias 0.9) dips, more so for
  Monte-Carlo **HSJ** (0.714→0.609). Comparing only the tail on HSJ could *look* like a drop, but
  the overall trend / Cohen's d are positive; n=10 seeds, HSJ is stochastic.
- The **`train_only`** protocol also rises, but there test accuracy **collapses** (0.94→0.73) —
  a confounded protocol (like label noise), not the repo's coverage-gap setup. Not the source of
  the "decrease" either.

## Reproduce
```
python export_investigation/analyze_cg.py        # full per-protocol/attack tables + Cohen's d
python export_investigation/confirm_inversion.py  # cluster-size stability + density↔spread inversion
python export_investigation/plot_verdict.py       # -> verdict_cg_spread_vs_density.png
```
