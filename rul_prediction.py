"""
Windows Python version of RUL_Prediction_Framework_FD001_FD004.ipynb.

Run from this folder:
    python rul_prediction.py

The script follows the notebook workflow from start to finish:
FD001 standard experiment, FD002-FD004 standard experiments, best-model
tables, Group K-Fold experiments, and final comparison tables.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets"
OUTPUT_DIR = BASE_DIR / "outputs"

CAP = 125
WINDOW = 30
BATCH = 128
STANDARD_EPOCHS = 20
KFOLD_EPOCHS = 10
KFOLD_SPLITS = 5
XGB_ROUNDS = 5000
XGB_EARLY_STOPPING = 100

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def print_section(title: str, char: str = "=") -> None:
    print()
    print(char * 70)
    print(title)
    print(char * 70)


def print_and_save_table(df: pd.DataFrame, name: str) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / name
    df.to_csv(path, index=False)
    print()
    print(name)
    print(df.to_string(index=False))
    print(f"Saved: {path}")


def cmapss_columns() -> list[str]:
    return (
        ["engine_id", "cycle"]
        + [f"op_{i}" for i in range(1, 4)]
        + [f"s_{i}" for i in range(1, 22)]
    )


def candidate_file_names(name: str) -> list[str]:
    path = Path(name)
    names = [path.name]

    if path.suffix == ".csv":
        names.append(path.with_suffix(".txt").name)
    elif path.suffix == ".txt":
        names.append(path.with_suffix(".csv").name)
    else:
        names.extend([f"{path.name}.csv", f"{path.name}.txt"])

    return list(dict.fromkeys(names))


def find_file(name: str) -> Path:
    search_dirs = [BASE_DIR, DATA_DIR]

    for directory in search_dirs:
        for candidate in candidate_file_names(name):
            path = directory / candidate
            if path.exists():
                return path

    tried = [
        str(directory / candidate)
        for directory in search_dirs
        for candidate in candidate_file_names(name)
    ]
    raise FileNotFoundError(
        f"File not found for {name}. Tried:\n" + "\n".join(tried)
    )


# ---------------------------------------------------------------------------
# Notebook sections 2-23: FD001 initial workflow.
# These helpers match the first definitions used by the notebook.
# ---------------------------------------------------------------------------


def load_cmapss_initial(train_path: Path, test_path: Path, rul_path: Path):
    cols = cmapss_columns()

    train = pd.read_csv(train_path, sep=r"\s+", header=None)
    test = pd.read_csv(test_path, sep=r"\s+", header=None)
    rul = pd.read_csv(rul_path, sep=r"\s+", header=None)

    train = train.iloc[:, : len(cols)]
    test = test.iloc[:, : len(cols)]

    train.columns = cols
    test.columns = cols
    rul.columns = ["RUL_end"]
    return train, test, rul


def add_rul_train_initial(df: pd.DataFrame, cap: int = CAP) -> pd.DataFrame:
    df = df.copy()
    max_cycle = df.groupby("engine_id")["cycle"].max()
    df["RUL"] = df["engine_id"].map(max_cycle) - df["cycle"]
    df["RUL"] = df["RUL"].clip(upper=cap)
    return df


def add_rul_test_initial(
    test_df: pd.DataFrame, rul_end_df: pd.DataFrame, cap: int = CAP
) -> pd.DataFrame:
    test_df = test_df.copy()
    max_cycle = test_df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycle.columns = ["engine_id", "max_cycle"]
    max_cycle["RUL_end"] = rul_end_df["RUL_end"].values

    test_df = test_df.merge(max_cycle, on="engine_id", how="left")
    test_df["RUL"] = (test_df["max_cycle"] - test_df["cycle"]) + test_df["RUL_end"]
    test_df["RUL"] = test_df["RUL"].clip(upper=cap)
    test_df.drop(columns=["max_cycle", "RUL_end"], inplace=True)
    return test_df


def split_by_engine_group_shuffle(
    df: pd.DataFrame, val_size: float = 0.2, random_state: int = 42
):
    groups = df["engine_id"].values
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=val_size, random_state=random_state
    )
    tr_idx, va_idx = next(splitter.split(df, groups=groups))
    return df.iloc[tr_idx].copy(), df.iloc[va_idx].copy()


class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class BiLSTMReg(nn.Module):
    def __init__(self, n_features, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        h_last = out[:, -1, :]
        return self.head(h_last)


class GRUAttnReg(nn.Module):
    def __init__(self, n_features, hidden=64, layers=1, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        h, _ = self.gru(x)
        scores = self.attn(h)
        w = torch.softmax(scores, dim=1)
        ctx = (w * h).sum(dim=1)
        return self.head(ctx)


def train_torch(model, train_loader, val_loader, epochs=20, lr=1e-3, device=DEVICE):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.HuberLoss(delta=10.0)

    best_val = float("inf")
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                pv = model(xb).cpu().numpy().ravel()
                ys.append(yb.numpy().ravel())
                ps.append(pv)

        yv = np.concatenate(ys)
        pv = np.concatenate(ps)
        val_mae = mean_absolute_error(yv, pv)

        if val_mae < best_val:
            best_val = val_mae
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        print(f"Epoch {ep:02d} | Val MAE={val_mae:.3f} | Val RMSE={rmse(yv, pv):.3f}")

    if best_state is None:
        raise RuntimeError("Training did not produce a best model state.")

    model.load_state_dict(best_state)
    model.to(device)
    return model


def predict_torch(model, X, batch=256, device=DEVICE):
    model.eval()
    preds = []
    loader = DataLoader(torch.tensor(X, dtype=torch.float32), batch_size=batch, shuffle=False)
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            preds.append(model(xb).cpu().numpy().ravel())
    return np.concatenate(preds)


def create_sequences(
    df: pd.DataFrame, feature_cols: list[str], window: int = WINDOW, target_col: str = "RUL"
):
    X, y = [], []

    for _, group in df.groupby("engine_id"):
        group = group.sort_values("cycle")
        features = group[feature_cols].values
        targets = group[target_col].values

        if len(group) < window:
            continue

        for i in range(window, len(group) + 1):
            X.append(features[i - window : i])
            y.append(targets[i - 1])

    return np.array(X), np.array(y)


def flatten_last_step(X):
    return X[:, -1, :]


def train_xgb_native(
    X_train,
    y_train,
    X_val,
    y_val,
    num_boost_round: int = XGB_ROUNDS,
    early_stopping_rounds: int = XGB_EARLY_STOPPING,
):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.03,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "seed": 42,
    }

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=200,
    )
    return booster


def run_fd001_initial_workflow():
    print_section("FD001 initial notebook workflow")

    train_file = find_file("train_FD001.csv")
    test_file = find_file("test_FD001.csv")
    rul_file = find_file("RUL_FD001.csv")

    train_df, test_df, rul_df = load_cmapss_initial(train_file, test_file, rul_file)
    print("Train:", train_df.shape, "| Test:", test_df.shape, "| RUL:", rul_df.shape)

    train_df = add_rul_train_initial(train_df, cap=CAP)
    test_df = add_rul_test_initial(test_df, rul_df, cap=CAP)

    op_cols = [c for c in train_df.columns if c.startswith("op_")]
    s_cols = [c for c in train_df.columns if c.startswith("s_")]
    feature_cols = op_cols + s_cols

    tr_df, va_df = split_by_engine_group_shuffle(train_df, val_size=0.2, random_state=42)

    scaler = StandardScaler()
    scaler.fit(tr_df[feature_cols].values)

    def apply_scaler(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[feature_cols] = scaler.transform(df[feature_cols].values)
        return df

    tr_df = apply_scaler(tr_df)
    va_df = apply_scaler(va_df)
    te_df = apply_scaler(test_df)

    print(
        "Train engines:",
        tr_df.engine_id.nunique(),
        " Val engines:",
        va_df.engine_id.nunique(),
    )

    X_tr, y_tr = create_sequences(tr_df, feature_cols, window=WINDOW)
    X_va, y_va = create_sequences(va_df, feature_cols, window=WINDOW)
    X_te, y_te = create_sequences(te_df, feature_cols, window=WINDOW)

    Xtr_tab = flatten_last_step(X_tr)
    Xva_tab = flatten_last_step(X_va)
    Xte_tab = flatten_last_step(X_te)

    print(X_tr.shape, y_tr.shape)
    print(X_va.shape, y_va.shape)
    print(X_te.shape, y_te.shape)

    tr_loader = DataLoader(SeqDataset(X_tr, y_tr), batch_size=BATCH, shuffle=True)
    va_loader = DataLoader(SeqDataset(X_va, y_va), batch_size=BATCH, shuffle=False)

    set_seed(42)
    bilstm = BiLSTMReg(n_features=X_tr.shape[-1], hidden=64, layers=2, dropout=0.2)
    bilstm = train_torch(bilstm, tr_loader, va_loader, epochs=STANDARD_EPOCHS, lr=1e-3)

    set_seed(42)
    gru_attn = GRUAttnReg(n_features=X_tr.shape[-1], hidden=64, layers=1, dropout=0.2)
    gru_attn = train_torch(gru_attn, tr_loader, va_loader, epochs=STANDARD_EPOCHS, lr=1e-3)

    pred_bi = predict_torch(bilstm, X_va)
    pred_gr = predict_torch(gru_attn, X_va)

    print("VAL BiLSTM  MAE/RMSE:", mean_absolute_error(y_va, pred_bi), rmse(y_va, pred_bi))
    print("VAL GRUAttn MAE/RMSE:", mean_absolute_error(y_va, pred_gr), rmse(y_va, pred_gr))

    xgb_model = train_xgb_native(Xtr_tab, y_tr, Xva_tab, y_va)
    pred_xg = xgb_model.predict(xgb.DMatrix(Xva_tab))

    print("VAL XGBoost MAE/RMSE:", mean_absolute_error(y_va, pred_xg), rmse(y_va, pred_xg))

    ens_val = (pred_bi + pred_gr + pred_xg) / 3.0

    results = pd.DataFrame(
        {
            "Model": ["BiLSTM", "GRU+Attention", "XGBoost (native)", "Ensemble(avg)"],
            "MAE": [
                mean_absolute_error(y_va, pred_bi),
                mean_absolute_error(y_va, pred_gr),
                mean_absolute_error(y_va, pred_xg),
                mean_absolute_error(y_va, ens_val),
            ],
            "RMSE": [
                rmse(y_va, pred_bi),
                rmse(y_va, pred_gr),
                rmse(y_va, pred_xg),
                rmse(y_va, ens_val),
            ],
        }
    ).sort_values("MAE")
    print_and_save_table(results, "fd001_validation_results.csv")

    pred_bi_te = predict_torch(bilstm, X_te)
    pred_gr_te = predict_torch(gru_attn, X_te)
    pred_xg_te = xgb_model.predict(xgb.DMatrix(Xte_tab))
    ens_te = (pred_bi_te + pred_gr_te + pred_xg_te) / 3.0

    test_mae = mean_absolute_error(y_te, ens_te)
    test_rmse = rmse(y_te, ens_te)
    print("TEST Ensemble MAE/RMSE:", test_mae, test_rmse)

    test_results = pd.DataFrame(
        {
            "Subset": ["FD001"],
            "Split": ["Test"],
            "Model": ["Ensemble(avg)"],
            "MAE": [test_mae],
            "RMSE": [test_rmse],
        }
    )
    print_and_save_table(test_results, "fd001_test_results.csv")

    return results, test_results


# ---------------------------------------------------------------------------
# Notebook sections 25-37: redefined helpers and FD002-FD004/K-Fold workflow.
# These helpers match the later notebook definitions.
# ---------------------------------------------------------------------------


def load_cmapss(train_path: Path, test_path: Path, rul_path: Path):
    cols = cmapss_columns()

    train_df = pd.read_csv(train_path, sep=r"\s+", header=None)
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None)
    rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None)

    train_df = train_df.iloc[:, :26]
    test_df = test_df.iloc[:, :26]

    train_df.columns = cols
    test_df.columns = cols
    rul_df.columns = ["RUL"]

    return train_df, test_df, rul_df


def add_rul_train(df: pd.DataFrame, cap: int = CAP) -> pd.DataFrame:
    max_cycle = df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycle.columns = ["engine_id", "max_cycle"]

    df = df.merge(max_cycle, on="engine_id")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df["RUL"] = df["RUL"].clip(upper=cap)
    df.drop(columns=["max_cycle"], inplace=True)

    return df


def add_rul_test(test_df: pd.DataFrame, rul_df: pd.DataFrame, cap: int = CAP) -> pd.DataFrame:
    rul_df = rul_df.copy()
    max_cycle = test_df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycle.columns = ["engine_id", "max_cycle"]

    rul_df["engine_id"] = rul_df.index + 1
    max_cycle = max_cycle.merge(rul_df, on="engine_id")
    max_cycle["final_cycle"] = max_cycle["max_cycle"] + max_cycle["RUL"]

    test_df = test_df.merge(max_cycle[["engine_id", "final_cycle"]], on="engine_id")
    test_df["RUL"] = test_df["final_cycle"] - test_df["cycle"]
    test_df["RUL"] = test_df["RUL"].clip(upper=cap)
    test_df.drop(columns=["final_cycle"], inplace=True)

    return test_df


def split_by_engine(df: pd.DataFrame, val_size: float = 0.2, random_state: int = 42):
    engine_ids = df["engine_id"].unique()
    train_ids, val_ids = train_test_split(
        engine_ids, test_size=val_size, random_state=random_state
    )

    train_df = df[df["engine_id"].isin(train_ids)].copy()
    val_df = df[df["engine_id"].isin(val_ids)].copy()

    return train_df, val_df


def run_subset_experiment(
    subset: str, cap: int = CAP, window: int = WINDOW, epochs: int = 20, batch_size: int = BATCH
) -> pd.DataFrame:
    print_section(f"Running experiment for {subset}")

    start_time = time.time()

    train_path = find_file(f"train_{subset}.csv")
    test_path = find_file(f"test_{subset}.csv")
    rul_path = find_file(f"RUL_{subset}.csv")

    train_df, test_df, rul_df = load_cmapss(train_path, test_path, rul_path)

    train_df = add_rul_train(train_df, cap=cap)
    test_df = add_rul_test(test_df, rul_df, cap=cap)

    op_cols = [c for c in train_df.columns if c.startswith("op_")]
    s_cols = [c for c in train_df.columns if c.startswith("s_")]
    feature_cols = op_cols + s_cols

    tr_df, va_df = split_by_engine(train_df, val_size=0.2, random_state=42)

    scaler = StandardScaler()
    scaler.fit(tr_df[feature_cols].values)

    def scale_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[feature_cols] = scaler.transform(df[feature_cols].values)
        return df

    tr_df = scale_df(tr_df)
    va_df = scale_df(va_df)
    te_df = scale_df(test_df)

    X_tr, y_tr = create_sequences(tr_df, feature_cols, window=window)
    X_va, y_va = create_sequences(va_df, feature_cols, window=window)
    X_te, y_te = create_sequences(te_df, feature_cols, window=window)

    Xtr_tab = flatten_last_step(X_tr)
    Xva_tab = flatten_last_step(X_va)
    Xte_tab = flatten_last_step(X_te)

    print("Train sequences:", X_tr.shape)
    print("Validation sequences:", X_va.shape)
    print("Test sequences:", X_te.shape)

    tr_loader = DataLoader(SeqDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(SeqDataset(X_va, y_va), batch_size=batch_size, shuffle=False)

    set_seed(42)
    bilstm_model = BiLSTMReg(n_features=X_tr.shape[-1], hidden=64, layers=2, dropout=0.2)
    bilstm_model = train_torch(bilstm_model, tr_loader, va_loader, epochs=epochs, lr=1e-3)

    set_seed(42)
    gru_model = GRUAttnReg(n_features=X_tr.shape[-1], hidden=64, layers=1, dropout=0.2)
    gru_model = train_torch(gru_model, tr_loader, va_loader, epochs=epochs, lr=1e-3)

    xgb_model = train_xgb_native(Xtr_tab, y_tr, Xva_tab, y_va)

    pred_bi_va = predict_torch(bilstm_model, X_va)
    pred_gr_va = predict_torch(gru_model, X_va)
    pred_xg_va = xgb_model.predict(xgb.DMatrix(Xva_tab))
    pred_ens_va = (pred_bi_va + pred_gr_va + pred_xg_va) / 3.0

    pred_bi_te = predict_torch(bilstm_model, X_te)
    pred_gr_te = predict_torch(gru_model, X_te)
    pred_xg_te = xgb_model.predict(xgb.DMatrix(Xte_tab))
    pred_ens_te = (pred_bi_te + pred_gr_te + pred_xg_te) / 3.0

    subset_results = pd.DataFrame(
        {
            "Subset": [subset] * 8,
            "Split": [
                "Validation",
                "Validation",
                "Validation",
                "Validation",
                "Test",
                "Test",
                "Test",
                "Test",
            ],
            "Model": [
                "BiLSTM",
                "GRU+Attention",
                "XGBoost",
                "Ensemble(avg)",
                "BiLSTM",
                "GRU+Attention",
                "XGBoost",
                "Ensemble(avg)",
            ],
            "MAE": [
                mean_absolute_error(y_va, pred_bi_va),
                mean_absolute_error(y_va, pred_gr_va),
                mean_absolute_error(y_va, pred_xg_va),
                mean_absolute_error(y_va, pred_ens_va),
                mean_absolute_error(y_te, pred_bi_te),
                mean_absolute_error(y_te, pred_gr_te),
                mean_absolute_error(y_te, pred_xg_te),
                mean_absolute_error(y_te, pred_ens_te),
            ],
            "RMSE": [
                rmse(y_va, pred_bi_va),
                rmse(y_va, pred_gr_va),
                rmse(y_va, pred_xg_va),
                rmse(y_va, pred_ens_va),
                rmse(y_te, pred_bi_te),
                rmse(y_te, pred_gr_te),
                rmse(y_te, pred_xg_te),
                rmse(y_te, pred_ens_te),
            ],
        }
    )

    elapsed = (time.time() - start_time) / 60
    print(f"Finished {subset} in {elapsed:.2f} minutes")

    return subset_results


def run_group_kfold_subset(
    subset: str,
    cap: int = CAP,
    window: int = WINDOW,
    k: int = KFOLD_SPLITS,
    epochs: int = KFOLD_EPOCHS,
    batch_size: int = BATCH,
) -> pd.DataFrame:
    train_path = find_file(f"train_{subset}.csv")
    test_path = find_file(f"test_{subset}.csv")
    rul_path = find_file(f"RUL_{subset}.csv")

    train_df, _, _ = load_cmapss(train_path, test_path, rul_path)
    train_df = add_rul_train(train_df, cap=cap)

    op_cols = [c for c in train_df.columns if c.startswith("op_")]
    s_cols = [c for c in train_df.columns if c.startswith("s_")]
    feature_cols = op_cols + s_cols

    engine_ids = train_df["engine_id"].unique()
    groups = engine_ids

    gkf = GroupKFold(n_splits=k)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
        gkf.split(engine_ids, groups=groups), start=1
    ):
        print_section(f"{subset} | Fold {fold}/{k}", char="-")

        train_engines = engine_ids[train_idx]
        val_engines = engine_ids[val_idx]

        tr_df = train_df[train_df["engine_id"].isin(train_engines)].copy()
        va_df = train_df[train_df["engine_id"].isin(val_engines)].copy()

        scaler = StandardScaler()
        scaler.fit(tr_df[feature_cols].values)

        tr_df[feature_cols] = scaler.transform(tr_df[feature_cols].values)
        va_df[feature_cols] = scaler.transform(va_df[feature_cols].values)

        X_tr, y_tr = create_sequences(tr_df, feature_cols, window=window)
        X_va, y_va = create_sequences(va_df, feature_cols, window=window)

        Xtr_tab = flatten_last_step(X_tr)
        Xva_tab = flatten_last_step(X_va)

        tr_loader = DataLoader(SeqDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
        va_loader = DataLoader(SeqDataset(X_va, y_va), batch_size=batch_size, shuffle=False)

        set_seed(42)
        bilstm_model = BiLSTMReg(n_features=X_tr.shape[-1], hidden=64, layers=2, dropout=0.2)
        bilstm_model = train_torch(
            bilstm_model, tr_loader, va_loader, epochs=epochs, lr=1e-3
        )

        set_seed(42)
        gru_model = GRUAttnReg(n_features=X_tr.shape[-1], hidden=64, layers=1, dropout=0.2)
        gru_model = train_torch(gru_model, tr_loader, va_loader, epochs=epochs, lr=1e-3)

        xgb_model = train_xgb_native(Xtr_tab, y_tr, Xva_tab, y_va)

        pred_bilstm = predict_torch(bilstm_model, X_va)
        pred_gru = predict_torch(gru_model, X_va)
        pred_xgb = xgb_model.predict(xgb.DMatrix(Xva_tab))
        pred_ensemble = (pred_bilstm + pred_gru + pred_xgb) / 3

        fold_results.append(
            ["BiLSTM", fold, mean_absolute_error(y_va, pred_bilstm), rmse(y_va, pred_bilstm)]
        )
        fold_results.append(
            ["GRU+Attention", fold, mean_absolute_error(y_va, pred_gru), rmse(y_va, pred_gru)]
        )
        fold_results.append(
            ["XGBoost", fold, mean_absolute_error(y_va, pred_xgb), rmse(y_va, pred_xgb)]
        )
        fold_results.append(
            [
                "Ensemble(avg)",
                fold,
                mean_absolute_error(y_va, pred_ensemble),
                rmse(y_va, pred_ensemble),
            ]
        )

    results_df = pd.DataFrame(fold_results, columns=["Model", "Fold", "MAE", "RMSE"])
    results_df.insert(0, "Subset", subset)

    return results_df


def summarize_kfold(kfold_df: pd.DataFrame, sort_by_subset: bool = False) -> pd.DataFrame:
    summary = (
        kfold_df.groupby(["Subset", "Model"])
        .agg(
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
        )
        .reset_index()
    )

    if sort_by_subset:
        return summary.sort_values(["Subset", "RMSE_mean"])
    return summary.sort_values("RMSE_mean")


def main() -> None:
    configure_console_encoding()
    set_seed(42)
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Device:", DEVICE)
    print("Base directory:", BASE_DIR)
    print("Data directory:", DATA_DIR)
    print("Output directory:", OUTPUT_DIR)
    print("This script runs the full notebook workflow. On CPU, it can take a long time.")

    run_fd001_initial_workflow()

    additional_subsets = ["FD002", "FD003", "FD004"]
    all_additional_results = []

    for subset in additional_subsets:
        result = run_subset_experiment(
            subset=subset,
            cap=CAP,
            window=WINDOW,
            epochs=STANDARD_EPOCHS,
            batch_size=BATCH,
        )
        all_additional_results.append(result)

    additional_results_df = pd.concat(all_additional_results, ignore_index=True)
    print_and_save_table(additional_results_df, "additional_results_fd002_fd004.csv")

    best_models = (
        additional_results_df[additional_results_df["Split"] == "Test"]
        .sort_values("RMSE")
        .groupby("Subset")
        .first()
        .reset_index()
        .sort_values("RMSE")
    )
    print_and_save_table(best_models, "best_models_fd002_fd004.csv")

    kfold_fd001 = run_group_kfold_subset(
        subset="FD001",
        cap=CAP,
        window=WINDOW,
        k=KFOLD_SPLITS,
        epochs=KFOLD_EPOCHS,
        batch_size=BATCH,
    )
    print_and_save_table(kfold_fd001, "kfold_fd001.csv")

    kfold_summary = summarize_kfold(kfold_fd001)
    print_and_save_table(kfold_summary, "kfold_summary_fd001.csv")

    all_kfold_results = []

    for subset in ["FD002", "FD003", "FD004"]:
        result = run_group_kfold_subset(
            subset=subset,
            cap=CAP,
            window=WINDOW,
            k=KFOLD_SPLITS,
            epochs=KFOLD_EPOCHS,
            batch_size=BATCH,
        )
        all_kfold_results.append(result)

    all_kfold_results_df = pd.concat(all_kfold_results, ignore_index=True)
    print_and_save_table(all_kfold_results_df, "kfold_fd002_fd004.csv")

    all_kfold_summary = summarize_kfold(all_kfold_results_df, sort_by_subset=True)
    print_and_save_table(all_kfold_summary, "kfold_summary_fd002_fd004.csv")

    final_all_results = all_kfold_summary.copy()
    final_all_results = final_all_results.round(
        {
            "MAE_mean": 2,
            "MAE_std": 2,
            "RMSE_mean": 2,
            "RMSE_std": 2,
        }
    )
    print_and_save_table(final_all_results, "final_all_results.csv")

    best_model_per_subset = (
        all_kfold_summary.sort_values(["Subset", "RMSE_mean"])
        .groupby("Subset")
        .first()
        .reset_index()
    )
    best_model_per_subset = best_model_per_subset.round(
        {
            "MAE_mean": 2,
            "MAE_std": 2,
            "RMSE_mean": 2,
            "RMSE_std": 2,
        }
    )
    print_and_save_table(best_model_per_subset, "best_model_per_subset.csv")

    print_section("Notebook workflow finished")


if __name__ == "__main__":
    main()
