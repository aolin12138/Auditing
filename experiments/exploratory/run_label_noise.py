"""
Label noise experiment driver.
Runs across iris, wine, Car Evaluation datasets.
Independent of bias.py — can be run alongside original experiments.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import polars as pl
from ucimlrepo import fetch_ucirepo

from util import labelencoding
from experiments.defects import measure_defect


def main():
    seeds = [42, 125, 58]          # experiment seeds (train/test split)
    noise_seeds = list(range(10))  # noise injection seeds (which points get flipped)
    noise_levels = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    res = pl.DataFrame()

    for dataset_name in ["iris", "wine", "Car Evaluation"]:
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*50}")

        data = fetch_ucirepo(dataset_name)
        X = labelencoding(data.data.features.to_numpy().copy())
        y = labelencoding(data.data.targets.to_numpy().copy())

        for seed in seeds:
            for noise_seed in noise_seeds:
                for noise in noise_levels:
                    label = f"{dataset_name}_s{seed}_ns{noise_seed}_n{noise:.1f}"
                    if noise_seed == 0:
                        print(f"  {label} ...", end=" ", flush=True)

                    results = measure_defect(
                        X,
                        y,
                        defect_type="label_noise",
                        defect_level=float(noise),
                        SEED=seed,
                        noise_seed=noise_seed,
                        n_splits=10,
                        dbscan=True,
                    )

                    results = results.with_columns(
                        pl.lit(seed).alias("seed"),
                        pl.lit(noise_seed).alias("noise_seed"),
                        pl.lit(dataset_name).alias("dataset"),
                        pl.lit("label_noise").alias("defect_type"),
                    )
                    res = pl.concat([res, results])
                print(f"done seed={seed}, noise_seeds 0-9")

    output_path = "data/data_label_noise.parquet"
    res.write_parquet(output_path)
    print(f"\nSaved {res.shape[0]} rows to {output_path}")


if __name__ == "__main__":
    main()
