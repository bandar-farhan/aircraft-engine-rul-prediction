"""
Create clean combined reporting CSV files for the dashboard.

This script does not train models and does not change rul_prediction.py.
It only combines existing result CSVs under outputs/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

SPLIT_COLUMNS = ["Subset", "Split", "Model", "MAE", "RMSE"]
KFOLD_COLUMNS = ["Subset", "Model", "MAE_mean", "MAE_std", "RMSE_mean", "RMSE_std"]


def read_required_csv(name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Required result file is missing: {path}")
    return pd.read_csv(path)


def build_split_results() -> pd.DataFrame:
    fd001_validation = read_required_csv("fd001_validation_results.csv")
    fd001_validation.insert(0, "Split", "Validation")
    fd001_validation.insert(0, "Subset", "FD001")

    fd001_test = read_required_csv("fd001_test_results.csv")
    additional = read_required_csv("additional_results_fd002_fd004.csv")

    split_results = pd.concat(
        [
            fd001_validation[SPLIT_COLUMNS],
            fd001_test[SPLIT_COLUMNS],
            additional[SPLIT_COLUMNS],
        ],
        ignore_index=True,
    )

    split_order = {"Validation": 0, "Test": 1}
    split_results["_split_order"] = split_results["Split"].map(split_order).fillna(9)
    split_results = (
        split_results.sort_values(["Subset", "_split_order", "RMSE", "MAE"])
        .drop(columns=["_split_order"])
        .reset_index(drop=True)
    )

    return split_results


def build_kfold_results() -> pd.DataFrame:
    fd001 = read_required_csv("kfold_summary_fd001.csv")
    fd002_fd004 = read_required_csv("kfold_summary_fd002_fd004.csv")

    kfold_results = pd.concat(
        [fd001[KFOLD_COLUMNS], fd002_fd004[KFOLD_COLUMNS]],
        ignore_index=True,
    )

    kfold_results = (
        kfold_results.sort_values(["Subset", "RMSE_mean", "MAE_mean"])
        .reset_index(drop=True)
    )

    return kfold_results


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    OUTPUT_DIR.mkdir(exist_ok=True)

    split_results = build_split_results()
    kfold_results = build_kfold_results()

    split_path = OUTPUT_DIR / "final_split_results_all_subsets.csv"
    kfold_path = OUTPUT_DIR / "final_kfold_results_all_subsets.csv"

    split_results.to_csv(split_path, index=False)
    kfold_results.to_csv(kfold_path, index=False)

    print(f"Saved {split_path} ({len(split_results)} rows)")
    print(f"Saved {kfold_path} ({len(kfold_results)} rows)")
    print("Split subsets:", ", ".join(sorted(split_results["Subset"].unique())))
    print("K-fold subsets:", ", ".join(sorted(kfold_results["Subset"].unique())))


if __name__ == "__main__":
    main()
