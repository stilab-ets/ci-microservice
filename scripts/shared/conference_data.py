from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MAX_DURATION_SEC = 399 * 24 * 60 * 60
KEEP_EVENTS = {"push", "pull_request"}
LAG_HISTORY = 7
N_FOLDS = 10
N_ITERS = 5


def project_label(path: Path) -> str:
    name = path.name
    suffix = "_fixed.csv"
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    return name


def load_filtered_runs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("created_at", "updated_at"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    if "conclusion" in df.columns:
        df = df[df["conclusion"] == "success"]
    if "workflow_event_trigger" in df.columns:
        df = df[df["workflow_event_trigger"].isin(KEEP_EVENTS)]
    if "build_duration" in df.columns:
        df = df[df["build_duration"] <= MAX_DURATION_SEC]
    if "created_at" in df.columns:
        df = df.sort_values("created_at").reset_index(drop=True)

    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "created_at" in df.columns:
        df["hour"] = df["created_at"].dt.hour
        df["dow"] = df["created_at"].dt.dayofweek
        df["month"] = df["created_at"].dt.month

    if "workflow_id" not in df.columns:
        workflow_ids = [None]
    else:
        workflow_ids = df["workflow_id"].dropna().unique().tolist()

    chunks = []
    for workflow_id in workflow_ids:
        if workflow_id is None:
            chunk = df.copy()
        else:
            chunk = df[df["workflow_id"] == workflow_id].copy()

        if "created_at" in chunk.columns:
            chunk = chunk.sort_values("created_at")
            chunk["secs_since_prev"] = (
                chunk["created_at"] - chunk["created_at"].shift(1)
            ).dt.total_seconds()

        for lag in range(1, LAG_HISTORY + 1):
            chunk[f"_duration_history_{lag}"] = chunk["build_duration"].shift(lag)

        history_cols = [f"_duration_history_{i}" for i in range(1, LAG_HISTORY + 1)]
        chunk["duration_lag_1"] = chunk["_duration_history_1"]
        for lag in range(2, LAG_HISTORY + 1):
            chunk[f"duration_lag_{lag}"] = chunk[f"_duration_history_{lag}"]
        chunk["window_avg_7"] = chunk[history_cols].mean(axis=1)
        chunk["window_std_7"] = chunk[history_cols].std(axis=1)
        chunk.drop(columns=history_cols, inplace=True)
        chunks.append(chunk)

    return pd.concat(chunks, ignore_index=True)


def make_folds(n: int, n_folds: int = N_FOLDS) -> list[np.ndarray] | None:
    if n < n_folds:
        return None
    indices = np.arange(n)
    folds = [fold for fold in np.array_split(indices, n_folds) if len(fold) > 0]
    if len(folds) != n_folds:
        return None
    return folds


def expanding_train_test_indices(folds: list[np.ndarray], iteration: int) -> tuple[np.ndarray, np.ndarray]:
    train_idx = np.concatenate([folds[i] for i in range(0, 5 + iteration)])
    test_idx = folds[5 + iteration]
    return train_idx, test_idx


def sliding_train_test_indices(folds: list[np.ndarray], iteration: int) -> tuple[np.ndarray, np.ndarray]:
    train_start = iteration
    train_end = iteration + 5
    test_idx = folds[train_end]
    train_idx = np.concatenate([folds[i] for i in range(train_start, train_end)])
    return train_idx, test_idx
