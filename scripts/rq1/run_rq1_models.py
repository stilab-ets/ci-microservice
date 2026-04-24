from __future__ import annotations

import argparse
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

SHARED_DIR = Path(__file__).resolve().parents[1] / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from conference_data import (
    N_ITERS,
    expanding_train_test_indices,
    make_folds,
    project_label,
    sliding_train_test_indices,
)
from predictive_feature_config import EXCLUDED_PREDICTIVE_FEATURES

ROOT = Path(__file__).resolve().parents[2]
MODELING_DIR = ROOT / "data" / "frozen_paper_inputs" / "modeling"
RESULTS_DIR = ROOT / "results" / "rq1"

warnings.filterwarnings(
    "ignore",
    message=r"`sklearn\.utils\.parallel\.delayed` should be used with `sklearn\.utils\.parallel\.Parallel`.*",
    category=UserWarning,
)

RANDOM_STATE = 42
CORR_THR = 0.7
REDUNDANCY_R2_THR = 0.9
REQUIRED_FEATURES: list[str] = [
    "duration_lag_1",
    "duration_lag_2",
    "duration_lag_3",
    "duration_lag_4",
    "duration_lag_5",
    "duration_lag_6",
    "duration_lag_7",
    "window_avg_7",
]
EXCLUDED_FEATURES = EXCLUDED_PREDICTIVE_FEATURES
MODEL_SELECTION_METRICS = ["nrmse", "r2", "rmse", "mae"]
FEATURE_CLIP_Q_LOW = 0.01
FEATURE_CLIP_Q_HIGH = 0.99
TARGET_CLIP_Q_LOW = 0.01
TARGET_CLIP_Q_HIGH = 0.99


class QuantileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower_q: float = FEATURE_CLIP_Q_LOW, upper_q: float = FEATURE_CLIP_Q_HIGH) -> None:
        self.lower_q = lower_q
        self.upper_q = upper_q

    def fit(self, X: pd.DataFrame, y: object = None) -> "QuantileClipper":
        X_df = pd.DataFrame(X).apply(pd.to_numeric, errors="coerce")
        self.columns_ = list(X_df.columns)
        self.lower_bounds_ = X_df.quantile(self.lower_q, numeric_only=True).to_dict()
        self.upper_bounds_ = X_df.quantile(self.upper_q, numeric_only=True).to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = pd.DataFrame(X, copy=True)
        if hasattr(self, "columns_"):
            X_df = X_df.reindex(columns=self.columns_)
        X_df = X_df.apply(pd.to_numeric, errors="coerce").astype(float)
        for col in X_df.columns:
            lower = self.lower_bounds_.get(col)
            upper = self.upper_bounds_.get(col)
            if lower is not None and upper is not None:
                X_df[col] = X_df[col].clip(lower=lower, upper=upper)
        return X_df


class RobustTargetRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        estimator: object,
        clip_target: bool = True,
        target_lower_q: float = TARGET_CLIP_Q_LOW,
        target_upper_q: float = TARGET_CLIP_Q_HIGH,
        log_target: bool = True,
    ) -> None:
        self.estimator = estimator
        self.clip_target = clip_target
        self.target_lower_q = target_lower_q
        self.target_upper_q = target_upper_q
        self.log_target = log_target

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RobustTargetRegressor":
        y_arr = np.asarray(y, dtype=float)
        if self.clip_target and len(y_arr) > 10:
            self.target_lower_ = float(np.quantile(y_arr, self.target_lower_q))
            self.target_upper_ = float(np.quantile(y_arr, self.target_upper_q))
            y_fit = np.clip(y_arr, self.target_lower_, self.target_upper_)
        else:
            self.target_lower_ = float(np.nanmin(y_arr))
            self.target_upper_ = float(np.nanmax(y_arr))
            y_fit = y_arr

        if self.log_target:
            y_fit = np.log1p(np.maximum(y_fit, 0.0))

        self.fitted_estimator_ = clone(self.estimator)
        self.fitted_estimator_.fit(X, y_fit)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        y_pred = np.asarray(self.fitted_estimator_.predict(X), dtype=float)
        if self.log_target:
            y_pred = np.expm1(y_pred)
        return np.maximum(y_pred, 0.0)


class TeeStream(io.TextIOBase):
    def __init__(self, *streams: io.TextIOBase) -> None:
        self.streams = streams

    def write(self, s: str) -> int:
        for stream in self.streams:
            try:
                stream.write(s)
                stream.flush()
            except ValueError:
                continue
        return len(s)

    def flush(self) -> None:
        for stream in self.streams:
            try:
                stream.flush()
            except ValueError:
                continue


def setup_console_log(results_dir: Path) -> io.TextIOBase:
    results_dir.mkdir(parents=True, exist_ok=True)
    handle = open(results_dir / "run_console.log", "w", encoding="utf-8")
    sys.stdout = TeeStream(sys.__stdout__, handle)
    sys.stderr = TeeStream(sys.__stderr__, handle)
    return handle


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan")

    sd = float(np.std(y_true, ddof=1)) if len(y_true) > 1 else 0.0
    mean = float(np.mean(y_true)) if len(y_true) > 0 else 0.0
    nrmse = float(rmse / sd) if sd > 0 else float("nan")
    cvrmse = float(rmse / mean) if mean > 0 else float("nan")

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "nrmse": nrmse,
        "cvrmse": cvrmse,
        "r2": r2,
    }


def rounded_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {key: round(value, 6) if np.isfinite(value) else value for key, value in metrics.items()}


def _numeric_fill_median(X: pd.DataFrame) -> pd.DataFrame:
    Xn = X.apply(pd.to_numeric, errors="coerce")
    Xn = Xn.replace([np.inf, -np.inf], np.nan)
    med = Xn.median(numeric_only=True)
    return Xn.fillna(med)


def first_window_indices(folds: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if len(folds) < 6:
        return None
    train40 = np.concatenate([folds[i] for i in range(0, 4)])
    val10 = folds[4]
    test10 = folds[5]
    return train40, val10, test10


def spearman_correlation_filter(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    feature_cols: list[str],
    always_keep: list[str],
    thr: float = CORR_THR,
) -> list[str]:
    X = _numeric_fill_median(X_tr[feature_cols])
    std = X.std(axis=0, ddof=0)
    X = X.loc[:, std > 0]
    cols = list(X.columns)
    if len(cols) <= 1:
        return cols

    corr = X.corr(method="spearman").abs().fillna(0.0)
    ynum = pd.to_numeric(y_tr, errors="coerce")
    y_non_null = ynum.dropna()
    y_is_constant = y_non_null.nunique() <= 1
    target_corr: dict[str, float] = {}
    for col in cols:
        series = X[col]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        series_non_null = pd.Series(series).dropna()
        if y_is_constant or series_non_null.nunique() <= 1:
            target_corr[col] = 0.0
            continue
        value = pd.Series(series).corr(ynum, method="spearman")
        target_corr[col] = 0.0 if pd.isna(value) else float(abs(value))

    keep = set(cols)
    required = {col for col in always_keep if col in keep}
    pairs: list[tuple[str, str, float]] = []
    for i, left in enumerate(cols):
        for j in range(i + 1, len(cols)):
            right = cols[j]
            value = float(corr.iat[i, j])
            if value > thr:
                pairs.append((left, right, value))
    pairs.sort(key=lambda item: item[2], reverse=True)

    for left, right, _ in pairs:
        if left not in keep or right not in keep:
            continue
        left_required = left in required
        right_required = right in required
        if left_required and not right_required:
            keep.remove(right)
            continue
        if right_required and not left_required:
            keep.remove(left)
            continue
        if left_required and right_required:
            continue
        if target_corr[left] >= target_corr[right]:
            keep.remove(right)
        else:
            keep.remove(left)

    return [col for col in cols if col in keep]


def redundancy_filter_r2(
    X_tr: pd.DataFrame,
    feature_cols: list[str],
    always_keep: list[str],
    thr: float = REDUNDANCY_R2_THR,
) -> list[str]:
    X = _numeric_fill_median(X_tr[feature_cols])
    cols = list(X.columns)
    required = {col for col in always_keep if col in cols}

    if len(X) < 2:
        return cols

    X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=cols, index=X.index)
    drop: set[str] = set()

    for feature in cols:
        if feature in required:
            continue
        others = [col for col in cols if col != feature]
        if not others:
            continue

        target = X_scaled[feature]
        if pd.Series(target).nunique(dropna=False) <= 1:
            continue

        model = LinearRegression()
        model.fit(X_scaled[others], target)
        score = float(model.score(X_scaled[others], target))
        if not np.isnan(score) and score >= thr:
            drop.add(feature)

    return [col for col in cols if col not in drop]


def screen_features_first_window_only(
    data: pd.DataFrame,
    feature_cols: list[str],
    y_col: str = "build_duration",
    corr_thr: float = CORR_THR,
    red_thr: float = REDUNDANCY_R2_THR,
    always_keep: list[str] = REQUIRED_FEATURES,
) -> list[str]:
    X = data.drop(columns=[y_col]).reset_index(drop=True)
    y = data[y_col].reset_index(drop=True)

    pools_X: list[pd.DataFrame] = []
    pools_y: list[pd.Series] = []

    for workflow_id in X["workflow_id"].unique():
        mask = X["workflow_id"] == workflow_id
        Xw = X.loc[mask].reset_index(drop=True)
        yw = y.loc[mask].reset_index(drop=True)
        folds = make_folds(len(Xw))
        if folds is None:
            continue
        window = first_window_indices(folds)
        if window is None:
            continue
        train40, _, _ = window
        pools_X.append(Xw.iloc[train40][feature_cols])
        pools_y.append(yw.iloc[train40])

    if not pools_X:
        return feature_cols

    X_tr = pd.concat(pools_X, ignore_index=True)
    y_tr = pd.concat(pools_y, ignore_index=True)

    kept = spearman_correlation_filter(X_tr, y_tr, feature_cols, always_keep, thr=corr_thr)
    kept = redundancy_filter_r2(X_tr, kept, always_keep, thr=red_thr)

    kept_set = set(kept)
    for feature in always_keep:
        if feature in feature_cols:
            kept_set.add(feature)
    return [col for col in feature_cols if col in kept_set]


def load_modeling_project_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    if data is None or len(data) == 0:
        raise ValueError(f"Project {project_label(path)} has no rows in modeling data.")
    return data.reset_index(drop=True)


def filter_candidate_features(feature_cols: list[str]) -> list[str]:
    excluded = set(EXCLUDED_FEATURES)
    return [feature for feature in feature_cols if feature not in excluded]


def build_global_feature_columns(project_data: dict[str, pd.DataFrame]) -> list[str]:
    feature_set: set[str] = set()
    for data in project_data.values():
        feature_set.update(col for col in data.columns if col not in {"build_duration", "workflow_id"})
    return sorted(filter_candidate_features(list(feature_set)))


def screen_features_global_first_window(
    project_data: dict[str, pd.DataFrame],
    feature_cols: list[str],
    y_col: str = "build_duration",
    corr_thr: float = CORR_THR,
    red_thr: float = REDUNDANCY_R2_THR,
    always_keep: list[str] = REQUIRED_FEATURES,
) -> list[str]:
    pools_X: list[pd.DataFrame] = []
    pools_y: list[pd.Series] = []

    for data in project_data.values():
        X = data.drop(columns=[y_col]).reset_index(drop=True)
        y = data[y_col].reset_index(drop=True)

        for workflow_id in X["workflow_id"].unique():
            mask = X["workflow_id"] == workflow_id
            Xw = X.loc[mask].reset_index(drop=True)
            yw = y.loc[mask].reset_index(drop=True)
            folds = make_folds(len(Xw))
            if folds is None:
                continue
            window = first_window_indices(folds)
            if window is None:
                continue
            train40, _, _ = window
            pool = Xw.iloc[train40].reindex(columns=feature_cols, fill_value=0.0)
            pools_X.append(pool)
            pools_y.append(yw.iloc[train40])

    if not pools_X:
        return feature_cols

    X_tr = pd.concat(pools_X, ignore_index=True)
    y_tr = pd.concat(pools_y, ignore_index=True)

    kept = spearman_correlation_filter(X_tr, y_tr, feature_cols, always_keep, thr=corr_thr)
    kept = redundancy_filter_r2(X_tr, kept, always_keep, thr=red_thr)

    kept_set = set(kept)
    for feature in always_keep:
        if feature in feature_cols:
            kept_set.add(feature)
    return [col for col in feature_cols if col in kept_set]


def build_model_specs() -> dict[str, object]:
    scaled_prefix = [
        ("clipper", QuantileClipper()),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    unscaled_prefix = [
        ("clipper", QuantileClipper()),
        ("imputer", SimpleImputer(strategy="median")),
    ]

    return {
        "DT": RobustTargetRegressor(
            estimator=Pipeline(
                steps=unscaled_prefix
                + [
                    (
                        "estimator",
                        DecisionTreeRegressor(
                            max_depth=8,
                            min_samples_leaf=5,
                            random_state=RANDOM_STATE,
                        ),
                    )
                ]
            )
        ),
        "KNN": RobustTargetRegressor(
            estimator=Pipeline(
                steps=scaled_prefix
                + [
                    (
                        "estimator",
                        KNeighborsRegressor(
                            n_neighbors=9,
                            weights="distance",
                            p=1,
                        ),
                    )
                ]
            )
        ),
        "LR": RobustTargetRegressor(
            estimator=Pipeline(
                steps=scaled_prefix + [("estimator", LinearRegression())]
            )
        ),
        "RF": RobustTargetRegressor(
            estimator=Pipeline(
                steps=unscaled_prefix
                + [
                    (
                        "estimator",
                        RandomForestRegressor(
                            n_estimators=400,
                            max_depth=12,
                            min_samples_leaf=5,
                            max_features="sqrt",
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        ),
                    )
                ]
            )
        ),
        "SGB": RobustTargetRegressor(
            estimator=Pipeline(
                steps=unscaled_prefix
                + [
                    (
                        "estimator",
                        GradientBoostingRegressor(
                            n_estimators=300,
                            learning_rate=0.03,
                            max_depth=2,
                            min_samples_leaf=10,
                            subsample=0.8,
                            random_state=RANDOM_STATE,
                        ),
                    )
                ]
            )
        ),
        "SVR": RobustTargetRegressor(
            estimator=Pipeline(
                steps=scaled_prefix
                + [("estimator", SVR(kernel="rbf", C=10.0, epsilon=0.05, gamma="scale"))]
            )
        ),
        "XGB": RobustTargetRegressor(
            estimator=Pipeline(
                steps=unscaled_prefix
                + [
                    (
                        "estimator",
                        XGBRegressor(
                            objective="reg:squarederror",
                            n_estimators=300,
                            learning_rate=0.03,
                            max_depth=3,
                            min_child_weight=5,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            reg_lambda=2.0,
                            random_state=RANDOM_STATE,
                            n_jobs=1,
                            verbosity=0,
                        ),
                    )
                ]
            )
        ),
        "HGB": RobustTargetRegressor(
            estimator=Pipeline(
                steps=unscaled_prefix
                + [
                    (
                        "estimator",
                        HistGradientBoostingRegressor(
                            learning_rate=0.03,
                            max_depth=3,
                            max_iter=300,
                            min_samples_leaf=20,
                            l2_regularization=1.0,
                            random_state=RANDOM_STATE,
                        ),
                    )
                ]
            )
        ),
    }


def fit_model(estimator: object, X_train: pd.DataFrame, y_train: pd.Series) -> object:
    fitted = clone(estimator)
    fitted.fit(X_train, y_train.to_numpy())
    return fitted


def predict_model(fitted: object, X_test: pd.DataFrame) -> np.ndarray:
    y_pred = fitted.predict(X_test)
    return np.maximum(y_pred, 0.0)


def fit_predict_model(estimator: object, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
    fitted = fit_model(estimator, X_train, y_train)
    return predict_model(fitted, X_test)


def implementation_label(model_name: str, estimator: object) -> str:
    if isinstance(estimator, RobustTargetRegressor):
        return implementation_label(model_name, estimator.estimator)
    if isinstance(estimator, Pipeline):
        return estimator.named_steps["estimator"].__class__.__name__
    return estimator.__class__.__name__


def load_best_model_selection(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"project", "model"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Model-selection CSV {path} is missing required columns: {sorted(missing)}")
    return df


def prepare_feature_sets(
    project_data: dict[str, pd.DataFrame],
    screening_scope: str,
    results_dir: Path,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    if screening_scope == "none":
        rows: list[dict[str, object]] = []
        project_feature_cols: dict[str, list[str]] = {}
        project_kept_features: dict[str, list[str]] = {}
        for project, data in project_data.items():
            feature_cols = filter_candidate_features(
                [col for col in data.columns if col not in {"build_duration", "workflow_id"}]
            )
            project_feature_cols[project] = feature_cols
            project_kept_features[project] = feature_cols
            for feature in feature_cols:
                rows.append(
                    {
                        "project": project,
                        "feature": feature,
                        "kept": True,
                        "is_yaml_metric": feature.startswith("yaml_"),
                    }
                )
        pd.DataFrame(rows).to_csv(results_dir / "prepared_feature_inventory.csv", index=False)
        return project_feature_cols, project_kept_features

    if screening_scope == "global":
        global_feature_cols = build_global_feature_columns(project_data)
        kept_features = screen_features_global_first_window(
            project_data,
            global_feature_cols,
            y_col="build_duration",
            corr_thr=CORR_THR,
            red_thr=REDUNDANCY_R2_THR,
            always_keep=REQUIRED_FEATURES,
        )
        screening_df = pd.DataFrame(
            {
                "feature": global_feature_cols,
                "kept": [feature in kept_features for feature in global_feature_cols],
                "is_yaml_metric": [feature.startswith("yaml_") for feature in global_feature_cols],
            }
        )
        screening_df.to_csv(results_dir / "global_feature_screening.csv", index=False)
        pd.DataFrame({"feature": kept_features}).to_csv(results_dir / "global_kept_features.csv", index=False)
        project_feature_cols = {project: global_feature_cols for project in project_data}
        project_kept_features = {project: kept_features for project in project_data}
        return project_feature_cols, project_kept_features

    rows: list[dict[str, object]] = []
    kept_rows: list[dict[str, object]] = []
    project_feature_cols = {}
    project_kept_features = {}
    for project, data in project_data.items():
        feature_cols = filter_candidate_features(
            [col for col in data.columns if col not in {"build_duration", "workflow_id"}]
        )
        kept_features = screen_features_first_window_only(
            data,
            feature_cols,
            y_col="build_duration",
            corr_thr=CORR_THR,
            red_thr=REDUNDANCY_R2_THR,
            always_keep=REQUIRED_FEATURES,
        )
        project_feature_cols[project] = feature_cols
        project_kept_features[project] = kept_features
        for feature in feature_cols:
            rows.append(
                {
                    "project": project,
                    "feature": feature,
                    "kept": feature in kept_features,
                    "is_yaml_metric": feature.startswith("yaml_"),
                }
            )
        for feature in kept_features:
            kept_rows.append({"project": project, "feature": feature})
    pd.DataFrame(rows).to_csv(results_dir / "project_feature_screening.csv", index=False)
    pd.DataFrame(kept_rows).to_csv(results_dir / "project_kept_features.csv", index=False)
    return project_feature_cols, project_kept_features


def evaluate_project(
    path: Path,
    data: pd.DataFrame,
    feature_cols: list[str],
    kept_features: list[str],
    screening_scope: str,
    window_mode: str,
) -> tuple[pd.DataFrame, str]:
    project = project_label(path)
    X = data.drop(columns=["build_duration"]).reset_index(drop=True)
    y = data["build_duration"].reset_index(drop=True).astype(float)
    X = X.reindex(columns=["workflow_id", *feature_cols], fill_value=0.0)

    models = build_model_specs()
    folds = make_folds(len(data))
    if folds is None:
        raise ValueError(f"Project {project} does not have enough runs for 10 ordered folds.")

    records: list[dict[str, object]] = []
    log_lines = [
        f"Project: {project}",
        f"Source CSV: {path}",
        f"Rows in modeling dataset: {len(data)}",
        f"Screening scope: {screening_scope}",
        f"Candidate feature count: {len(feature_cols)}",
        f"Kept feature count: {len(kept_features)}",
        f"Window mode: {window_mode}",
        f"Required features: {', '.join(REQUIRED_FEATURES) if REQUIRED_FEATURES else 'none'}",
        f"Excluded features: {', '.join(EXCLUDED_FEATURES) if EXCLUDED_FEATURES else 'none'}",
        "Model label `SVR` denotes the support-vector regressor used in RQ1.",
        "",
        "Kept features:",
        ", ".join(kept_features),
        "",
    ]

    lag_1_preds = y.shift(1)

    for iteration in range(N_ITERS):
        if window_mode == "sliding":
            train_idx, test_idx = sliding_train_test_indices(folds, iteration)
            train_pct = 50
            train_start = iteration * 10
            train_end = train_start + 50
            test_start = train_end
            test_end = test_start + 10
            split_label = (
                f"train {train_start}-{train_end}% | test {test_start}-{test_end}%"
            )
        else:
            train_idx, test_idx = expanding_train_test_indices(folds, iteration)
            train_pct = 50 + iteration * 10
            train_start = 0
            train_end = train_pct
            test_start = train_pct
            test_end = train_pct + 10
            split_label = (
                f"train {train_start}-{train_end}% | test {test_start}-{test_end}%"
            )
        log_lines.append(
            f"Iteration {iteration + 1}: {split_label} | "
            f"n_train={len(train_idx)} | n_test={len(test_idx)}"
        )

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx].to_numpy()

        baseline_pred = lag_1_preds.iloc[test_idx]
        baseline_valid = baseline_pred.notna().to_numpy()
        if baseline_valid.any():
            baseline_metrics = compute_metrics(
                y_test[baseline_valid],
                baseline_pred.iloc[baseline_valid].to_numpy(),
            )
            records.append(
                {
                    "project": project,
                    "screening_scope": screening_scope,
                    "model": "Baseline",
                    "implementation": "lag_1",
                    "window_mode": window_mode,
                    "iteration": iteration + 1,
                    "train_pct": train_pct,
                    "train_pct_start": train_start,
                    "train_pct_end": train_end,
                    "test_pct_start": test_start,
                    "test_pct_end": test_end,
                    "n_train": int(len(train_idx)),
                    "n_test": int(baseline_valid.sum()),
                    "n_features_initial": int(len(feature_cols)),
                    "n_features_kept": int(len(kept_features)),
                    **rounded_metrics(baseline_metrics),
                }
            )
            log_lines.append(
                "  Baseline | "
                + " | ".join(
                    [
                        f"MAE={baseline_metrics['mae']:.3f}",
                        f"RMSE={baseline_metrics['rmse']:.3f}",
                        f"NRMSE={baseline_metrics['nrmse']:.3f}",
                        f"CVRMSE={baseline_metrics['cvrmse']:.3f}",
                        f"MSE={baseline_metrics['mse']:.3f}",
                        f"R2={baseline_metrics['r2']:.3f}",
                    ]
                )
            )

        X_train = X.iloc[train_idx][kept_features]
        X_test = X.iloc[test_idx][kept_features]

        for model_name, estimator in models.items():
            y_pred = fit_predict_model(estimator, X_train, y_train, X_test)
            metrics = compute_metrics(y_test, y_pred)
            records.append(
                {
                    "project": project,
                    "screening_scope": screening_scope,
                    "model": model_name,
                    "implementation": implementation_label(model_name, estimator),
                    "window_mode": window_mode,
                    "iteration": iteration + 1,
                    "train_pct": train_pct,
                    "train_pct_start": train_start,
                    "train_pct_end": train_end,
                    "test_pct_start": test_start,
                    "test_pct_end": test_end,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "n_features_initial": int(len(feature_cols)),
                    "n_features_kept": int(len(kept_features)),
                    **rounded_metrics(metrics),
                }
            )
            log_lines.append(
                f"  {model_name:<8}| "
                + " | ".join(
                    [
                        f"MAE={metrics['mae']:.3f}",
                        f"RMSE={metrics['rmse']:.3f}",
                        f"NRMSE={metrics['nrmse']:.3f}",
                        f"CVRMSE={metrics['cvrmse']:.3f}",
                        f"MSE={metrics['mse']:.3f}",
                        f"R2={metrics['r2']:.3f}",
                    ]
                )
            )
        log_lines.append("")

    iteration_df = pd.DataFrame(records).sort_values(["model", "iteration"]).reset_index(drop=True)
    averages_df = (
        iteration_df.groupby(["project", "screening_scope", "model", "implementation"], as_index=False)[
            ["mae", "rmse", "nrmse", "cvrmse", "mse", "r2", "n_test", "n_features_initial", "n_features_kept"]
        ]
        .mean()
        .sort_values(["nrmse", "rmse", "mae"], ascending=[True, True, True])
    )

    log_lines.append("Average Metrics Per Model")
    for _, row in averages_df.iterrows():
        log_lines.append(
            f"  {row['model']:<8}| "
            f"MAE={row['mae']:.3f} | RMSE={row['rmse']:.3f} | "
            f"NRMSE={row['nrmse']:.3f} | CVRMSE={row['cvrmse']:.3f} | "
            f"MSE={row['mse']:.3f} | R2={row['r2']:.3f}"
        )
    log_lines.append("")

    return iteration_df, "\n".join(log_lines)


def rank_project_models(
    project_summary: pd.DataFrame,
    metric: str,
    learned_only: bool,
) -> pd.DataFrame:
    df = project_summary.copy()
    if learned_only:
        df = df[df["model"] != "Baseline"].copy()
    if df.empty:
        return df

    lower_is_better = metric != "r2"
    df["rank_within_project"] = df.groupby("project")[metric].rank(
        method="min",
        ascending=lower_is_better,
    )

    if lower_is_better:
        sort_cols = ["project", metric, "nrmse", "rmse", "mae", "model"]
        ascending = [True, True, True, True, True, True]
    else:
        sort_cols = ["project", metric, "nrmse", "rmse", "mae", "model"]
        ascending = [True, False, True, True, True, True]

    return df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)


def write_best_model_summaries(project_summary: pd.DataFrame, results_dir: Path) -> None:
    for metric in MODEL_SELECTION_METRICS:
        ranked_all = rank_project_models(project_summary, metric, learned_only=False)
        ranked_all.to_csv(results_dir / f"project_model_rankings_by_{metric}.csv", index=False)
        best_all = ranked_all.groupby("project", as_index=False).head(1).reset_index(drop=True)
        best_all.to_csv(results_dir / f"best_model_by_project_{metric}.csv", index=False)

        ranked_learned = rank_project_models(project_summary, metric, learned_only=True)
        ranked_learned.to_csv(
            results_dir / f"project_learned_model_rankings_by_{metric}.csv",
            index=False,
        )
        best_learned = ranked_learned.groupby("project", as_index=False).head(1).reset_index(drop=True)
        best_learned.to_csv(
            results_dir / f"best_learned_model_by_project_{metric}.csv",
            index=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RQ1 duration prediction models with ordered time-aware folds.")
    parser.add_argument(
        "--modeling-dir",
        type=Path,
        default=MODELING_DIR,
        help="Directory containing modeling CSV files.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory where RQ1 results will be written.",
    )
    parser.add_argument(
        "--screening-scope",
        choices=["global", "per_project", "none"],
        default="global",
        help="Whether feature screening is shared across projects, done separately per project, or skipped for prepared datasets.",
    )
    parser.add_argument(
        "--window-mode",
        choices=["expanding", "sliding"],
        default="expanding",
        help="Use the original expanding window or a fixed-size sliding window (50%% train, next 10%% test).",
    )
    parser.add_argument(
        "--project",
        help="Optional exact project label (derived from CSV filename without _fixed.csv).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional number of projects to process after sorting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    logs_dir = results_dir / "logs"
    console_handle = setup_console_log(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.modeling_dir.glob("*.csv"))
    if args.project:
        paths = [path for path in paths if project_label(path) == args.project]
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise FileNotFoundError(
            "No matching modeling CSV files found for RQ1 evaluation. Run scripts\\prepare_modeling_data.py first."
        )

    print(f"[RQ1] Starting evaluation for {len(paths)} project(s).")
    print(f"[RQ1] Modeling directory: {args.modeling_dir}")
    print(f"[RQ1] Results directory: {results_dir}")
    print(f"[RQ1] Screening scope: {args.screening_scope}")
    print(f"[RQ1] Window mode: {args.window_mode}")

    project_data = {project_label(path): load_modeling_project_data(path) for path in paths}
    project_feature_cols, project_kept_features = prepare_feature_sets(
        project_data,
        args.screening_scope,
        results_dir,
    )
    for project in project_data:
        print(
            f"[RQ1] Feature set {project}: kept {len(project_kept_features[project])} of "
            f"{len(project_feature_cols[project])} candidate features."
        )

    all_iteration_frames: list[pd.DataFrame] = []
    combined_log_parts: list[str] = []

    for index, path in enumerate(paths, start=1):
        project = project_label(path)
        print(f"[RQ1] [{index}/{len(paths)}] Processing {project}")
        iteration_df, log_text = evaluate_project(
            path,
            project_data[project],
            project_feature_cols[project],
            project_kept_features[project],
            args.screening_scope,
            args.window_mode,
        )
        all_iteration_frames.append(iteration_df)
        (logs_dir / f"{project}.log").write_text(log_text, encoding="utf-8")
        combined_log_parts.append(log_text)
        print(f"[RQ1] [{index}/{len(paths)}] Finished {project}")

    iteration_metrics = pd.concat(all_iteration_frames, ignore_index=True).sort_values(
        ["project", "model", "iteration"]
    )
    project_summary = (
        iteration_metrics.groupby(
            ["project", "screening_scope", "window_mode", "model", "implementation"],
            as_index=False,
        )[
            ["mae", "rmse", "nrmse", "cvrmse", "mse", "r2", "n_test", "n_features_initial", "n_features_kept"]
        ]
        .mean()
        .sort_values(["project", "nrmse", "rmse", "mae"], ascending=[True, True, True, True])
        .reset_index(drop=True)
    )
    overall_summary = (
        iteration_metrics.groupby(["screening_scope", "window_mode", "model", "implementation"], as_index=False)[
            ["mae", "rmse", "nrmse", "cvrmse", "mse", "r2", "n_test", "n_features_initial", "n_features_kept"]
        ]
        .mean()
        .sort_values(["nrmse", "rmse", "mae"], ascending=[True, True, True])
        .reset_index(drop=True)
    )

    iteration_metrics.to_csv(results_dir / "iteration_metrics.csv", index=False)
    project_summary.to_csv(results_dir / "project_summary.csv", index=False)
    overall_summary.to_csv(results_dir / "overall_summary.csv", index=False)
    write_best_model_summaries(project_summary, results_dir)
    (results_dir / "combined.log").write_text(
        "\n\n" + ("\n\n" + ("-" * 80) + "\n\n").join(combined_log_parts),
        encoding="utf-8",
    )
    print("[RQ1] Done.")
    print(f"[RQ1] Wrote iteration metrics to: {results_dir / 'iteration_metrics.csv'}")
    print(f"[RQ1] Wrote project summary to: {results_dir / 'project_summary.csv'}")
    print(f"[RQ1] Wrote overall summary to: {results_dir / 'overall_summary.csv'}")
    print(f"[RQ1] Wrote best-model summaries for: {', '.join(MODEL_SELECTION_METRICS)}")
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    console_handle.close()


if __name__ == "__main__":
    main()
