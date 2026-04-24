from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from preprocessing_local import (
    DROP_COLUMNS,
    FileTypesBinarizer,
    build_curated_file_type_features,
    make_unique_columns,
    warn_duplicates,
)
from predictive_feature_config import PRUNED_PREDICTIVE_FEATURES

ROOT = Path(__file__).resolve().parents[2]
CLEANED_DIR = ROOT / "data" / "intermediate" / "cleaned"
MODELING_DIR = ROOT / "data" / "prepared" / "modeling"


def engineer_features(
    file_path: Union[Path, pd.DataFrame],
    output_path: Path | None = None,
    *,
    verbose: bool = True,
    file_type_mode: str = "curated_microservice",
) -> pd.DataFrame:
    if isinstance(file_path, pd.DataFrame):
        df = file_path.copy()
        source_name = "in_memory_df"
    else:
        df = pd.read_csv(file_path)
        source_name = file_path.name

    if verbose:
        print(f"[FE] {source_name}: input rows -> {len(df):,}")

    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        df["hour"] = df["created_at"].dt.hour
        df["dow"] = df["created_at"].dt.dayofweek
        df["month"] = df["created_at"].dt.month
        df["day_or_night"] = np.where(df["hour"].between(6, 17), 1, 0).astype(np.int8)

    if "updated_at" in df.columns:
        df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce")

    if "gh_first_commit_created_at" in df.columns and "created_at" in df.columns:
        first_commit = pd.to_datetime(df["gh_first_commit_created_at"], errors="coerce", utc=False)
        created_at = pd.to_datetime(df["created_at"], errors="coerce", utc=False)
        df["project_age_days"] = (created_at - first_commit).dt.total_seconds() / 86400.0

    if "created_at" in df.columns and "build_duration" in df.columns:
        if "workflow_id" in df.columns:
            grouped = []
            for workflow_id, chunk in df.groupby("workflow_id", dropna=False, sort=False):
                chunk = chunk.sort_values("created_at").copy()
                chunk["duration_lag_1"] = chunk["build_duration"].shift(1)
                chunk["window_avg_7"] = chunk["build_duration"].shift(1).rolling(window=7, min_periods=7).mean()
                grouped.append(chunk)
            df = pd.concat(grouped, ignore_index=True)
        else:
            df = df.sort_values("created_at").copy()
            df["duration_lag_1"] = df["build_duration"].shift(1)
            df["window_avg_7"] = df["build_duration"].shift(1).rolling(window=7, min_periods=7).mean()

    if "workflow_id" in df.columns:
        workflow_counts = df["workflow_id"].value_counts()
        valid_workflows = workflow_counts[workflow_counts >= 100].index.tolist()
        if valid_workflows:
            df = df[df["workflow_id"].isin(valid_workflows)].copy()

    df.drop(columns=DROP_COLUMNS, inplace=True, errors="ignore")
    df.drop(columns=["yaml_runner_os"], inplace=True, errors="ignore")

    df.dropna(inplace=True)

    categorical_columns = ["workflow_event_trigger", "issuer_name"]
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    if "file_types" in df.columns:
        if file_type_mode == "curated_microservice":
            curated_df = build_curated_file_type_features(df["file_types"])
            df = pd.concat([df.drop(columns=["file_types"]), curated_df], axis=1)
        elif file_type_mode == "raw_onehot":
            ft_bin = FileTypesBinarizer(sep=",")
            ft_array = ft_bin.fit_transform(df[["file_types"]])
            ft_cols = ft_bin.get_feature_names_out()
            df = pd.concat(
                [
                    df.drop(columns=["file_types"]),
                    pd.DataFrame(ft_array, columns=ft_cols, index=df.index),
                ],
                axis=1,
            )
        else:
            raise ValueError("file_type_mode must be one of: 'raw_onehot', 'curated_microservice'")

    if "branch" in df.columns:
        b = df["branch"].astype(str)
        df["branch"] = np.select(
            [
                b.str.contains("fix", case=False, na=False),
                b.str.contains(r"\b(?:main|master)\b", case=False, na=False, regex=True),
            ],
            [0, 1],
            default=2,
        ).astype(np.int8)

    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(np.int8)

    non_numeric_cols = [
        col
        for col in df.columns
        if df[col].dtype == "object" and col not in {"workflow_event_trigger", "issuer_name", "branch"}
    ]
    if non_numeric_cols:
        df.drop(columns=non_numeric_cols, inplace=True, errors="ignore")

    df.drop(columns=PRUNED_PREDICTIVE_FEATURES, inplace=True, errors="ignore")

    df.columns = [re.sub(r"[\[\]<>]", "_", str(c)) for c in df.columns]
    if verbose:
        warn_duplicates(df, tag="after sanitize")
    df.columns = make_unique_columns(df.columns)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        if verbose:
            print(f"[FE] saved -> {output_path}")

    if verbose:
        print(f"[FE] {source_name}: output rows -> {len(df):,}")

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create modeling datasets from cleaned workflow histories.")
    parser.add_argument("--input-dir", type=Path, default=CLEANED_DIR)
    parser.add_argument("--output-dir", type=Path, default=MODELING_DIR)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.input_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found under {args.input_dir}")

    for path in paths:
        engineer_features(path, args.output_dir / path.name, verbose=not args.quiet)


if __name__ == "__main__":
    main()
