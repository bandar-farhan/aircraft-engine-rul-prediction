"""
Generate dashboard-only engine recommendation CSV files.

This script imports and reuses the existing RUL pipeline from rul_prediction.py.
It does not change the model architecture or training behavior in that file.

Run:
    python generate_dashboard_data.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import rul_prediction as rp


SUBSETS = ["FD001", "FD002", "FD003", "FD004"]


def priority_from_rul(value: float) -> str:
    if value <= 20:
        return "critical"
    if value <= 50:
        return "high"
    if value <= 90:
        return "medium"
    return "low"


def action_from_priority(priority: str) -> str:
    return {
        "critical": "Inspect immediately and schedule maintenance before next operating window.",
        "high": "Plan maintenance soon and monitor engine condition closely.",
        "medium": "Add to upcoming maintenance plan and continue monitoring.",
        "low": "No immediate maintenance action required; keep routine monitoring.",
    }[priority]


def schedule_window_from_priority(priority: str) -> str:
    return {
        "critical": "0-20 cycles",
        "high": "21-50 cycles",
        "medium": "51-90 cycles",
        "low": "90+ cycles",
    }[priority]


def make_last_engine_sequences(df: pd.DataFrame, feature_cols: list[str], window: int):
    rows = []
    X = []

    for engine_id, group in df.groupby("engine_id"):
        group = group.sort_values("cycle")
        if len(group) < window:
            continue

        last_window = group[feature_cols].values[-window:]
        last_row = group.iloc[-1]
        X.append(last_window)
        rows.append(
            {
                "engine_id": int(engine_id),
                "last_cycle": int(last_row["cycle"]),
                "actual_RUL": float(last_row["RUL"]),
            }
        )

    return np.array(X), pd.DataFrame(rows)


def round_prediction(value: float) -> float:
    return round(max(float(value), 0.0), 3)


def build_outputs(
    subset: str,
    rows: pd.DataFrame,
    pred_bi: np.ndarray,
    pred_gr: np.ndarray,
    pred_xg: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    engine_predictions = rows.copy()
    engine_predictions.insert(0, "subset", subset)
    engine_predictions["bilstm_predicted_RUL"] = [round_prediction(v) for v in pred_bi]
    engine_predictions["gru_attention_predicted_RUL"] = [round_prediction(v) for v in pred_gr]
    engine_predictions["xgboost_predicted_RUL"] = [round_prediction(v) for v in pred_xg]
    engine_predictions["ensemble_predicted_RUL"] = [
        round_prediction(v) for v in (pred_bi + pred_gr + pred_xg) / 3.0
    ]
    engine_predictions["absolute_error"] = (
        engine_predictions["actual_RUL"] - engine_predictions["ensemble_predicted_RUL"]
    ).abs().round(3)

    engine_predictions["maintenance_priority"] = engine_predictions[
        "ensemble_predicted_RUL"
    ].apply(priority_from_rul)
    engine_predictions["suggested_action"] = engine_predictions[
        "maintenance_priority"
    ].apply(action_from_priority)
    engine_predictions["suggested_schedule_window"] = engine_predictions[
        "maintenance_priority"
    ].apply(schedule_window_from_priority)

    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    maintenance_priority = (
        engine_predictions.assign(
            priority_rank=engine_predictions["maintenance_priority"].map(priority_order)
        )
        .sort_values(["priority_rank", "ensemble_predicted_RUL", "engine_id"])
        .drop(columns=["priority_rank"])
        .reset_index(drop=True)
    )

    maintenance_schedule = (
        maintenance_priority.groupby(["subset", "maintenance_priority", "suggested_schedule_window"])
        .agg(
            engine_count=("engine_id", "count"),
            min_predicted_RUL=("ensemble_predicted_RUL", "min"),
            avg_predicted_RUL=("ensemble_predicted_RUL", "mean"),
            max_predicted_RUL=("ensemble_predicted_RUL", "max"),
            engines=("engine_id", lambda values: ", ".join(str(int(v)) for v in values)),
        )
        .reset_index()
    )
    maintenance_schedule["priority_rank"] = maintenance_schedule[
        "maintenance_priority"
    ].map(priority_order)
    maintenance_schedule = (
        maintenance_schedule.sort_values(["priority_rank", "min_predicted_RUL"])
        .drop(columns=["priority_rank"])
        .reset_index(drop=True)
    )
    maintenance_schedule["avg_predicted_RUL"] = maintenance_schedule[
        "avg_predicted_RUL"
    ].round(3)

    return engine_predictions, maintenance_priority, maintenance_schedule


def generate_subset(
    subset: str,
    output_dir: Path,
    cap: int,
    window: int,
    epochs: int,
    batch_size: int,
    xgb_rounds: int,
    xgb_early_stopping: int,
) -> None:
    rp.print_section(f"Generating dashboard data for {subset}")
    start = time.time()

    train_path = rp.find_file(f"train_{subset}.csv")
    test_path = rp.find_file(f"test_{subset}.csv")
    rul_path = rp.find_file(f"RUL_{subset}.csv")

    train_df, test_df, rul_df = rp.load_cmapss(train_path, test_path, rul_path)
    train_df = rp.add_rul_train(train_df, cap=cap)
    test_df = rp.add_rul_test(test_df, rul_df, cap=cap)

    op_cols = [c for c in train_df.columns if c.startswith("op_")]
    s_cols = [c for c in train_df.columns if c.startswith("s_")]
    feature_cols = op_cols + s_cols

    tr_df, va_df = rp.split_by_engine(train_df, val_size=0.2, random_state=42)

    scaler = StandardScaler()
    scaler.fit(tr_df[feature_cols].values)

    for df in (tr_df, va_df, test_df):
        df[feature_cols] = scaler.transform(df[feature_cols].values)

    X_tr, y_tr = rp.create_sequences(tr_df, feature_cols, window=window)
    X_va, y_va = rp.create_sequences(va_df, feature_cols, window=window)
    X_engine, engine_rows = make_last_engine_sequences(test_df, feature_cols, window=window)

    if len(X_engine) == 0:
        raise RuntimeError(f"No test engines in {subset} have at least {window} cycles.")

    Xtr_tab = rp.flatten_last_step(X_tr)
    Xva_tab = rp.flatten_last_step(X_va)
    Xengine_tab = rp.flatten_last_step(X_engine)

    tr_loader = DataLoader(rp.SeqDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(rp.SeqDataset(X_va, y_va), batch_size=batch_size, shuffle=False)

    rp.set_seed(42)
    bilstm_model = rp.BiLSTMReg(n_features=X_tr.shape[-1], hidden=64, layers=2, dropout=0.2)
    bilstm_model = rp.train_torch(
        bilstm_model, tr_loader, va_loader, epochs=epochs, lr=1e-3
    )

    rp.set_seed(42)
    gru_model = rp.GRUAttnReg(n_features=X_tr.shape[-1], hidden=64, layers=1, dropout=0.2)
    gru_model = rp.train_torch(gru_model, tr_loader, va_loader, epochs=epochs, lr=1e-3)

    xgb_model = rp.train_xgb_native(
        Xtr_tab,
        y_tr,
        Xva_tab,
        y_va,
        num_boost_round=xgb_rounds,
        early_stopping_rounds=xgb_early_stopping,
    )

    pred_bi = rp.predict_torch(bilstm_model, X_engine)
    pred_gr = rp.predict_torch(gru_model, X_engine)
    pred_xg = xgb_model.predict(xgb.DMatrix(Xengine_tab))

    engine_predictions, maintenance_priority, maintenance_schedule = build_outputs(
        subset, engine_rows, pred_bi, pred_gr, pred_xg
    )

    output_dir.mkdir(exist_ok=True)
    engine_predictions.to_csv(output_dir / f"engine_predictions_{subset}.csv", index=False)
    maintenance_priority.to_csv(output_dir / f"maintenance_priority_{subset}.csv", index=False)
    maintenance_schedule.to_csv(output_dir / f"maintenance_schedule_{subset}.csv", index=False)

    print(
        "Engine ensemble MAE:",
        mean_absolute_error(
            engine_predictions["actual_RUL"], engine_predictions["ensemble_predicted_RUL"]
        ),
    )
    print(f"Saved dashboard data for {subset} in {(time.time() - start) / 60:.2f} minutes")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate RUL dashboard engine CSV data.")
    parser.add_argument("--subset", default="all", choices=["all", *SUBSETS])
    parser.add_argument("--output-dir", default=str(rp.OUTPUT_DIR))
    parser.add_argument("--cap", type=int, default=rp.CAP)
    parser.add_argument("--window", type=int, default=rp.WINDOW)
    parser.add_argument("--epochs", type=int, default=rp.STANDARD_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=rp.BATCH)
    parser.add_argument("--xgb-rounds", type=int, default=rp.XGB_ROUNDS)
    parser.add_argument("--xgb-early-stopping", type=int, default=rp.XGB_EARLY_STOPPING)
    return parser.parse_args()


def main() -> None:
    rp.configure_console_encoding()
    rp.set_seed(42)

    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    subsets = SUBSETS if args.subset == "all" else [args.subset]

    print("Device:", rp.DEVICE)
    print("Output directory:", output_dir)
    print("This trains the same model types to export per-engine dashboard predictions.")

    for subset in subsets:
        generate_subset(
            subset=subset,
            output_dir=output_dir,
            cap=args.cap,
            window=args.window,
            epochs=args.epochs,
            batch_size=args.batch_size,
            xgb_rounds=args.xgb_rounds,
            xgb_early_stopping=args.xgb_early_stopping,
        )


if __name__ == "__main__":
    main()
