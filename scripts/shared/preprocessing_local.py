from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

MAX_DURATION_SEC = 399 * 24 * 60 * 60

DROP_COLUMNS = [
    "repo",
    "id_build",
    "commit_sha",
    "conclusion",
    "workflow_name",
    "created_at",
    "updated_at",
    "gh_job_id",
    "fetch_duration",
    "gh_pull_req_number",
    "tests_passed",
    "status",
    "tests_failed",
    "tests_skipped",
    "tests_total",
    "gh_first_commit_created_at",
    "job_",
    "git_merged_with",
    "job_details",
    "build_language",
    "test_framework",
    "languages",
    "duration_from_ts",
    "build_duration_original",
    "run_attempt_from_api",
    "run_attempt",
    "fixed_with_api",
    "author_date",
    "author_date_x",
    "author_date_y",
    "repo_commit",
    "repo_commit_x",
    "repo_commit_y",
    "ct_service_roots",
    "ct_top_dirs",
    "ct_extensions",
    "ps_semantic_categories",
]

# Revised curated groups. We explicitly remove the old file-type categories that
# were too ecosystem-specific or duplicated stronger direct metrics:
# - ft_ms_native_systems
# - ft_ms_rust_ecosystem
# - ft_ms_android_resources
# - ft_ms_docker_related (Docker is modeled directly through
#   dockerfile_changed and docker_compose_changed)
CURATED_FILE_TYPE_GROUPS = {
    "ft_ms_workflow_yaml": {".yml", ".yaml"},
    "ft_ms_api_contracts": {".proto", ".graphql", ".gql", ".avsc", ".wsdl"},
    "ft_ms_jvm_build_and_code": {".gradle", ".java", ".kt", ".kts", ".jar", ".properties"},
    "ft_ms_frontend_web": {".ts", ".tsx", ".js", ".css", ".html", ".svg", ".hbs"},
    "ft_ms_scripts_ops": {".py", ".sh", ".bash", ".bat"},
    "ft_ms_config_data": {
        ".json",
        ".json5",
        ".ini",
        ".cfg",
        ".config",
        ".properties",
        ".toml",
        ".env",
        ".conf",
        ".default",
        ".example",
    },
}


def make_unique_columns(cols):
    seen = {}
    out = []
    for c in cols:
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}__dup{seen[c]}")
    return out


def warn_duplicates(df, tag=""):
    dups = df.columns[df.columns.duplicated()].tolist()
    if dups:
        print(f"[WARN] Duplicate columns {tag}: {len(dups)} (examples): {dups[:10]}")


def add_lag_features(df, workflow_id):
    workflow_df = df[df["workflow_id"] == workflow_id].copy()
    workflow_df = workflow_df.sort_values("created_at")
    workflow_df["secs_since_prev"] = (
        workflow_df["created_at"] - workflow_df["created_at"].shift(1)
    ).dt.total_seconds()
    for lag in range(1, 8):
        workflow_df[f"_duration_history_{lag}"] = workflow_df["build_duration"].shift(lag)

    history_cols = [f"_duration_history_{i}" for i in range(1, 8)]
    workflow_df["duration_lag_1"] = workflow_df["_duration_history_1"]
    workflow_df["window_avg_7"] = workflow_df[history_cols].mean(axis=1)
    workflow_df["window_std_7"] = workflow_df[history_cols].std(axis=1)
    workflow_df.drop(columns=history_cols, inplace=True)
    return workflow_df


class FileTypesBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, sep="|"):
        self.sep = sep
        self.classes_ = []

    def _to_series(self, X):
        if isinstance(X, pd.Series):
            return X
        if isinstance(X, pd.DataFrame):
            if X.shape[1] != 1:
                raise ValueError("FileTypesBinarizer expects exactly one column.")
            return X.iloc[:, 0]
        return pd.Series(X)

    def fit(self, X, y=None):
        s = self._to_series(X).fillna("")
        classes = set()
        for val in s:
            for tok in str(val).split(self.sep):
                tok = tok.strip()
                if tok:
                    classes.add(tok)
        self.classes_ = sorted(classes)
        return self

    def transform(self, X):
        s = self._to_series(X).fillna("")
        n = len(s)
        M = np.zeros((n, len(self.classes_)), dtype=np.float32)
        idx = {c: i for i, c in enumerate(self.classes_)}
        for row_i, val in enumerate(s):
            for tok in str(val).split(self.sep):
                tok = tok.strip()
                if tok in idx:
                    M[row_i, idx[tok]] = 1.0
        return M

    def get_feature_names_out(self, input_features=None):
        safe = [re.sub(r"[^0-9A-Za-z_]", "_", c) for c in self.classes_]
        return [f"ft_{s}" for s in safe]


def parse_file_type_tokens(value) -> list[str]:
    if pd.isna(value):
        return []
    s = str(value).strip()
    if not s or s == "[]":
        return []
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, (list, tuple, set)):
            tokens = [str(x).strip() for x in obj if str(x).strip()]
        else:
            tokens = [s]
    except Exception:
        cleaned = re.sub(r"[\[\]{}\"']", "", s)
        tokens = [tok.strip() for tok in cleaned.split(",") if tok.strip()]

    normalized = []
    for tok in tokens:
        tok = str(tok).strip().strip("'\"")
        tok = tok.replace("{", "").replace("}", "")
        if tok:
            normalized.append(tok)
    return normalized


def build_curated_file_type_features(series: pd.Series) -> pd.DataFrame:
    rows = []
    for value in series.fillna(""):
        tokens = set(parse_file_type_tokens(value))
        rows.append(
            {
                feature_name: np.float32(bool(tokens & allowed_tokens))
                for feature_name, allowed_tokens in CURATED_FILE_TYPE_GROUPS.items()
            }
        )
    return pd.DataFrame(rows, index=series.index)


def preprocess_data(
    file_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    *,
    verbose: bool = True,
    file_type_mode: str = "raw_onehot",
) -> pd.DataFrame:
    file_path = Path(file_path)
    df = pd.read_csv(file_path)
    if verbose:
        print(f"[RUNS] before filtering: {len(df):,}")

    if "conclusion" in df.columns:
        df = df[df["conclusion"] == "success"]

    if "workflow_event_trigger" in df.columns:
        df = df[df["workflow_event_trigger"].isin(["push", "pull_request"])]

    if verbose:
        print(f"[FILTER] success + trigger -> {len(df):,} rows")

    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"])
        df["hour"] = df["created_at"].dt.hour
        df["dow"] = df["created_at"].dt.dayofweek
        df["month"] = df["created_at"].dt.month
        df["day_or_night"] = np.where(df["hour"].between(6, 17), 1, 0).astype(np.int8)

    if "updated_at" in df.columns:
        df["updated_at"] = pd.to_datetime(df["updated_at"])

    if "gh_first_commit_created_at" in df.columns and "created_at" in df.columns:
        first_commit = pd.to_datetime(df["gh_first_commit_created_at"], errors="coerce", utc=False)
        created_at = pd.to_datetime(df["created_at"], errors="coerce", utc=False)
        df["project_age_days"] = (created_at - first_commit).dt.total_seconds() / 86400.0

    if verbose and {"created_at", "updated_at"}.issubset(df.columns):
        df["duration_from_ts"] = (df["updated_at"] - df["created_at"]).dt.total_seconds()
        diff = (df["build_duration"] - df["duration_from_ts"]).abs()
        print("[DURATION CHECK] head")
        print(df[["build_duration", "duration_from_ts"]].head())
        print(
            "[DURATION CHECK] summary",
            "\n  corr(build_duration, duration_from_ts):",
            np.corrcoef(df["build_duration"], df["duration_from_ts"])[0, 1],
            "\n  mean abs diff:", diff.mean(),
            "\n  95th pct abs diff:", np.percentile(diff.dropna(), 95),
        )

    before_cap = len(df)
    df = df[df["build_duration"] <= MAX_DURATION_SEC].copy()
    if verbose:
        print(f"[FILTER] drop durations > 399 days: {before_cap:,} -> {len(df):,} rows")

    if "created_at" in df.columns:
        df = df.sort_values(by="created_at").reset_index(drop=True)

    if "workflow_id" in df.columns:
        workflow_counts = df["workflow_id"].value_counts()
        valid_workflows = workflow_counts[workflow_counts >= 100].index.tolist()
        if valid_workflows:
            df = df[df["workflow_id"].isin(valid_workflows)].copy()
        else:
            valid_workflows = df["workflow_id"].unique().tolist()

        if verbose:
            print(
                f"[FILTER] workflows with >=100 runs (or all if fewer): "
                f"{len(df):,} rows across {len(valid_workflows)} workflow(s)"
            )

        lagged_data = []
        for wf_id in valid_workflows:
            lagged_data.append(add_lag_features(df, wf_id))

        if lagged_data:
            df = pd.concat(lagged_data, ignore_index=True)
            if verbose:
                print("after concatenating:", len(df))
    elif verbose:
        print("[INFO] No 'workflow_id' column; skipping per-workflow lag features.")

    df.drop(columns=DROP_COLUMNS, inplace=True, errors="ignore")
    df.drop(columns=["yaml_runner_os"], inplace=True, errors="ignore")
    if verbose:
        print("len df after drop columns", len(df))

    df.dropna(inplace=True)
    if verbose:
        print("len df after drop na", len(df))
        print(df.isna().sum())

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
            if verbose:
                active_counts = curated_df.sum(axis=0).sort_values(ascending=False)
                print("[FILE_TYPES] curated microservice groups active counts:")
                print(active_counts.to_string())
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

    df.columns = [re.sub(r"[\[\]<>]", "_", str(c)) for c in df.columns]
    if verbose:
        warn_duplicates(df, tag="after sanitize")
    df.columns = make_unique_columns(df.columns)

    if output_path is not None:
        output_path = Path(output_path)
        df.to_csv(output_path, index=False)
        if verbose:
            print(f"[SAVE] processed data -> {output_path}")

    if verbose:
        print(f"[RUNS] after filtering: {len(df):,}")
        bd = df["build_duration"]
        print(f"[DURATION] median: {bd.median():,.2f} s")
        print(f"[DURATION] average: {bd.mean():,.2f} s")
        print(f"[DURATION] std dev: {bd.std():,.2f} s")
        print(f"[DURATION] max:    {bd.max():,.2f} s")
        print(f"[DURATION] min:    {bd.min():,.2f} s")

    return df
