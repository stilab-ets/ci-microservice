from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

SHARED_DIR = Path(__file__).resolve().parents[1] / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))
RQ1_DIR_ON_PATH = Path(__file__).resolve().parents[1] / "rq1"
if str(RQ1_DIR_ON_PATH) not in sys.path:
    sys.path.insert(0, str(RQ1_DIR_ON_PATH))

from feature_engineering import engineer_features
from rq2_feature_metadata import infer_feature_group
from run_rq1_models import build_model_specs, fit_model, predict_model

ROOT = Path(__file__).resolve().parents[2]
CLEANED_ENRICHED_DIR = ROOT / "data" / "frozen_paper_inputs" / "cleaned_commit_patch_semantic_enriched"
RQ1_DIR = ROOT / "results" / "rq1"
PRQ_DIR = ROOT / "results" / "pq" / "stable_regions"
OUT_DIR = ROOT / "results" / "rq2_local_explanations"
OUT_FIGURES_DIR = OUT_DIR / "figures"
PAPER_DIR = ROOT / "paper"
PAPER_FIGURES_DIR = PAPER_DIR / "figures"
PAPER_TEX_PATH = PAPER_DIR / "generated_tables" / "rq2_regime_shift_local_results.tex"

WINDOW_QUANTILE = 0.25
MIN_WINDOW_PER_SIDE = 15
PERMUTATION_REPEATS = 20
RANDOM_STATE = 42

PROJECT_LABELS = {
    "Orange_OpenSourceouds_android_wf108176393": "ouds-android",
    "bmad simbmad ecosystem_wf69576399": "bmad",
    "ccpay_wf6192976": "ccpay",
    "collinbarrettFilterLists_wf75763098": "FilterLists",
    "daos_wf9020028": "daos",
    "jod-yksilo-ui_wf83806327": "jod-yksilo-ui",
    "m2Gilesm2os_wf105026558": "m2os",
    "pr3y_Bruce_wf121541665": "Bruce",
    "radareorg_radare2_wf1989843": "radare2",
    "rustlang_wf51073": "rust",
}


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_TEX_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_best_models() -> pd.DataFrame:
    return pd.read_csv(RQ1_DIR / "best_learned_model_by_project_r2.csv")


def load_kept_features() -> list[str]:
    return pd.read_csv(RQ1_DIR / "global_kept_features.csv")["feature"].astype(str).tolist()


def load_regime_events() -> pd.DataFrame:
    events = pd.read_csv(PRQ_DIR / "regime_events.csv")
    events["project_short"] = events["project"].map(lambda value: PROJECT_LABELS.get(str(value), str(value)))
    return events


def load_run_series() -> pd.DataFrame:
    run_series = pd.read_csv(PRQ_DIR / "run_series_with_regions.csv", low_memory=False)
    for col in ["created_at", "updated_at"]:
        if col in run_series.columns:
            run_series[col] = pd.to_datetime(run_series[col], errors="coerce", utc=True)
    return run_series


def load_engineered_with_regions(project: str, run_series: pd.DataFrame) -> pd.DataFrame:
    cleaned_path = CLEANED_ENRICHED_DIR / f"{project}_fixed.csv"
    if not cleaned_path.exists():
        raise FileNotFoundError(f"Missing enriched cleaned CSV for {project}: {cleaned_path}")

    cleaned = pd.read_csv(cleaned_path)
    if "created_at" in cleaned.columns:
        cleaned["created_at"] = pd.to_datetime(cleaned["created_at"], errors="coerce", utc=True)

    region_cols = [
        "commit_sha",
        "created_at",
        "workflow_id",
        "build_duration",
        "run_order",
        "region_id",
        "project",
        "project_short",
    ]
    project_series = run_series[run_series["project"] == project][region_cols].copy()
    merge_keys = ["commit_sha", "created_at", "workflow_id", "build_duration"]
    cleaned["_merge_occurrence"] = cleaned.groupby(merge_keys).cumcount()
    project_series["_merge_occurrence"] = project_series.groupby(merge_keys).cumcount()
    merged = cleaned.merge(
        project_series,
        on=[*merge_keys, "_merge_occurrence"],
        how="left",
    )
    merged.drop(columns=["_merge_occurrence"], inplace=True, errors="ignore")
    merged["run_order"] = pd.to_numeric(merged["run_order"], errors="coerce")
    merged["region_id"] = pd.to_numeric(merged["region_id"], errors="coerce")

    engineered = engineer_features(merged, output_path=None, verbose=False)
    if "run_order" not in engineered.columns or "region_id" not in engineered.columns:
        raise ValueError(f"Region metadata was lost during feature engineering for {project}")
    return engineered.sort_values("run_order").reset_index(drop=True)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    if len(y_true) > 1:
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else float("nan")
        sd = float(np.std(y_true, ddof=1))
        nrmse = float(rmse / sd) if sd > 0 else float("nan")
    else:
        r2 = float("nan")
        nrmse = float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2, "nrmse": nrmse}


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def fmt(value: object, digits: int = 2) -> str:
    if pd.isna(value):
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return latex_escape(value)


def select_local_window(event: pd.Series, engineered: pd.DataFrame) -> pd.DataFrame:
    before_region = int(event["from_region_id"])
    after_region = int(event["to_region_id"])
    before_rows = engineered[engineered["region_id"] == before_region].sort_values("run_order")
    after_rows = engineered[engineered["region_id"] == after_region].sort_values("run_order")

    before_window = int(math.ceil(len(before_rows) * WINDOW_QUANTILE))
    after_window = int(math.ceil(len(after_rows) * WINDOW_QUANTILE))
    balanced_size = min(before_window, after_window, len(before_rows), len(after_rows))
    if balanced_size < MIN_WINDOW_PER_SIDE:
        raise ValueError(
            f"Event {event['project']}#{event['event_id']} has too few local rows after engineering "
            f"({balanced_size} per side)."
        )

    local_df = pd.concat(
        [before_rows.tail(balanced_size), after_rows.head(balanced_size)],
        ignore_index=True,
    ).sort_values("run_order").reset_index(drop=True)
    return local_df


def top_nonzero_share(df: pd.DataFrame, value_col: str, label_col: str) -> tuple[str, float]:
    if df.empty:
        return "--", 0.0
    ordered = df.sort_values(value_col, ascending=False).reset_index(drop=True)
    top = ordered.iloc[0]
    return str(top[label_col]), float(top[value_col])


def top_n_labels(df: pd.DataFrame, value_col: str, label_col: str, n: int = 3) -> list[str]:
    if df.empty:
        return []
    ordered = df.sort_values([value_col, label_col], ascending=[False, True]).reset_index(drop=True)
    labels = ordered[label_col].astype(str).tolist()[:n]
    return labels


def plot_family_heatmap(family_df: pd.DataFrame) -> Path:
    pivot = family_df.pivot(index="event_label", columns="family", values="positive_share").fillna(0.0)
    pivot = pivot.reindex(columns=sorted(pivot.columns))
    fig, ax = plt.subplots(figsize=(12.5, max(6, 0.34 * len(pivot))))
    vmax = max(0.25, float(pivot.to_numpy().max())) if not pivot.empty else 0.25
    im = ax.imshow(pivot.to_numpy(), cmap="YlGnBu", vmin=0.0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("RQ2: Local family importance around regime-shift windows")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = float(pivot.iloc[i, j])
            if value >= 0.08:
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Normalized positive share")
    fig.tight_layout()
    out_path = OUT_FIGURES_DIR / "rq2_local_family_heatmap.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    fig.savefig(PAPER_FIGURES_DIR / out_path.name, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_top_family_counts(event_summary: pd.DataFrame) -> Path:
    counts = event_summary["top_family"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9.5, max(4.5, 0.45 * len(counts))))
    ax.barh(counts.index, counts.values, color="#2A9D8F")
    ax.set_xlabel("Number of regime events")
    ax.set_title("RQ2: Dominant local explanation family by regime event")
    for idx, value in enumerate(counts.values):
        ax.text(value + 0.1, idx, str(int(value)), va="center", fontsize=9)
    fig.tight_layout()
    out_path = OUT_FIGURES_DIR / "rq2_top_family_counts.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    fig.savefig(PAPER_FIGURES_DIR / out_path.name, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_gain_vs_shift(event_summary: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    colors = event_summary["direction"].map({"increase": "#E76F51", "decrease": "#2A9D8F"}).fillna("#577590")
    ax.scatter(event_summary["abs_pct_change"], event_summary["rmse_gain_pct"], c=colors, s=70, alpha=0.9)
    for _, row in event_summary.iterrows():
        ax.text(row["abs_pct_change"] + 0.3, row["rmse_gain_pct"], row["event_label"], fontsize=7, alpha=0.85)
    ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1)
    ax.set_xlabel("Absolute regime-shift size (%)")
    ax.set_ylabel("RMSE gain over lag-1 baseline (%)")
    ax.set_title("RQ2: Local model gain versus regime-shift magnitude")
    fig.tight_layout()
    out_path = OUT_FIGURES_DIR / "rq2_gain_vs_shift.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    fig.savefig(PAPER_FIGURES_DIR / out_path.name, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_direction_feature_heatmap(
    feature_summary: pd.DataFrame,
    direction: str,
    top_n_features: int = 12,
) -> Path:
    subset = feature_summary[feature_summary["direction"] == direction].copy()
    if subset.empty:
        raise ValueError(f"No feature rows available for direction '{direction}'.")

    feature_order = (
        subset.groupby("feature", as_index=False)["positive_share"]
        .mean()
        .sort_values("positive_share", ascending=False)
        .head(top_n_features)["feature"]
        .tolist()
    )
    event_order = (
        subset[["event_label", "project_short", "event_id"]]
        .drop_duplicates()
        .sort_values(["project_short", "event_id"])["event_label"]
        .tolist()
    )
    pivot = (
        subset[subset["feature"].isin(feature_order)]
        .pivot_table(index="event_label", columns="feature", values="positive_share", aggfunc="mean")
        .reindex(index=event_order, columns=feature_order)
        .fillna(0.0)
    )
    if pivot.empty:
        raise ValueError(f"No heatmap data remained for direction '{direction}'.")

    fig_width = max(8.5, 0.65 * len(pivot.columns) + 2.5)
    fig_height = max(4.5, 0.42 * len(pivot.index) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    vmax = max(0.15, float(pivot.to_numpy().max()))
    im = ax.imshow(pivot.to_numpy(), cmap="YlOrRd", vmin=0.0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=40, ha="right", fontsize=12)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=12)
    ax.tick_params(axis="both", labelsize=12)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = float(pivot.iloc[i, j])
            if value >= max(0.02, 0.15 * vmax):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=10, color="#222222")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Normalized positive importance share")
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label("Normalized positive importance share", fontsize=12)
    fig.tight_layout()
    out_path = OUT_FIGURES_DIR / f"rq2_{direction}_shift_feature_heatmap.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    fig.savefig(PAPER_FIGURES_DIR / out_path.name, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_shift_window_example(
    run_series: pd.DataFrame,
    events: pd.DataFrame,
    project_short: str = "bmad",
) -> Path:
    project_df = (
        run_series[run_series["project_short"] == project_short]
        .copy()
        .sort_values("run_order")
        .reset_index(drop=True)
    )
    if project_df.empty:
        raise ValueError(f"No run-series rows found for example project '{project_short}'.")

    project_events = (
        events[events["project_short"] == project_short]
        .copy()
        .sort_values("event_id")
        .reset_index(drop=True)
    )
    if project_events.empty:
        raise ValueError(f"No regime events found for example project '{project_short}'.")

    project_df["duration_min"] = project_df["build_duration"].astype(float) / 60.0
    fig, ax = plt.subplots(figsize=(11.5, 4.8))

    region_palette = ["#edf2fb", "#e2ece9", "#fff1e6", "#fde2e4", "#e9ecef"]
    region_bounds = (
        project_df.groupby("region_id", as_index=False)
        .agg(
            start_run=("run_order", "min"),
            end_run=("run_order", "max"),
            median_min=("duration_min", "median"),
        )
        .sort_values("start_run")
        .reset_index(drop=True)
    )
    for idx, row in region_bounds.iterrows():
        color = region_palette[idx % len(region_palette)]
        ax.axvspan(float(row["start_run"]), float(row["end_run"]), color=color, alpha=0.45, zorder=0)
        x_mid = float(row["start_run"] + row["end_run"]) / 2.0
        region_label_x = x_mid
        region_label_y = float(project_df["duration_min"].max()) * 1.02
        if idx == 0:
            region_label_x = x_mid + 18.0
            region_label_y = float(project_df["duration_min"].max()) * 0.74
        ax.text(
            region_label_x,
            region_label_y,
            f"R{int(row['region_id'])}",
            ha="center",
            va="bottom",
            fontsize=13,
            color="#33415c",
            fontweight="bold",
        )

    ax.plot(
        project_df["run_order"],
        project_df["duration_min"],
        color="#355070",
        linewidth=1.4,
        alpha=0.9,
        zorder=2,
    )
    ax.scatter(
        project_df["run_order"],
        project_df["duration_min"],
        s=9,
        color="#355070",
        alpha=0.65,
        zorder=3,
    )

    event_colors = ["#d62828", "#6a4c93", "#2a9d8f"]
    seen_labels: set[str] = set()
    ymax = float(project_df["duration_min"].max())
    example_test_bounds: list[tuple[float, float, str]] = []
    for idx, event in project_events.iterrows():
        before_rows = project_df[project_df["region_id"] == int(event["from_region_id"])].sort_values("run_order")
        after_rows = project_df[project_df["region_id"] == int(event["to_region_id"])].sort_values("run_order")
        balanced_size = min(
            int(math.ceil(len(before_rows) * WINDOW_QUANTILE)),
            int(math.ceil(len(after_rows) * WINDOW_QUANTILE)),
            len(before_rows),
            len(after_rows),
        )
        before_window = before_rows.tail(balanced_size)
        after_window = after_rows.head(balanced_size)
        color = event_colors[idx % len(event_colors)]
        test_start = float(before_window["run_order"].min())
        test_end = float(after_window["run_order"].max())
        example_test_bounds.append((test_start, test_end, color))

        before_label = "Old-regime 25% test area" if "before" not in seen_labels else None
        after_label = "New-regime 25% test area" if "after" not in seen_labels else None
        boundary_label = "Accepted regime shift" if "boundary" not in seen_labels else None

        ax.axvspan(
            float(before_window["run_order"].min()),
            float(before_window["run_order"].max()),
            facecolor=color,
            alpha=0.18,
            hatch="////",
            edgecolor=color,
            linewidth=0.0,
            zorder=1,
            label=before_label,
        )
        ax.axvspan(
            float(after_window["run_order"].min()),
            float(after_window["run_order"].max()),
            facecolor=color,
            alpha=0.18,
            hatch="\\\\\\\\",
            edgecolor=color,
            linewidth=0.0,
            zorder=1,
            label=after_label,
        )
        ax.axvline(
            float(event["change_run_order"]),
            color=color,
            linestyle="--",
            linewidth=1.6,
            zorder=4,
            label=boundary_label,
        )
        ax.text(
            float(event["change_run_order"]) + 4,
            ymax * (0.78 - 0.08 * idx) if idx == 0 else ymax * (0.95 - 0.08 * idx),
            f"E{int(event['event_id'])}",
            color=color,
            fontsize=13,
            fontweight="bold",
        )
        seen_labels.update({"before", "after", "boundary"})

    test_arrow_y = ymax * 0.12
    for idx, (test_start, test_end, color) in enumerate(example_test_bounds):
        ax.annotate(
            "",
            xy=(test_start, test_arrow_y),
            xytext=(test_end, test_arrow_y),
            arrowprops={"arrowstyle": "<->", "color": color, "lw": 1.8},
        )
        ax.text(
            (test_start + test_end) / 2.0,
            test_arrow_y + ymax * 0.02,
            f"Shift-area test window {idx + 1}",
            ha="center",
            va="bottom",
            fontsize=13,
            color=color,
            fontweight="bold",
        )

    ax.set_xlabel("Workflow run order", fontsize=19)
    ax.set_ylabel("Workflow duration (minutes)", fontsize=19)
    ax.set_ylim(0, ymax * 1.12)
    ax.grid(axis="y", alpha=0.2, linewidth=0.6)
    ax.tick_params(axis="both", labelsize=16)

    handles, labels = ax.get_legend_handles_labels()
    unique_handles: list[object] = []
    unique_labels: list[str] = []
    for handle, label in zip(handles, labels):
        if label and label not in unique_labels:
            unique_handles.append(handle)
            unique_labels.append(label)
    ax.legend(unique_handles, unique_labels, loc="upper left", fontsize=13, frameon=True)

    fig.tight_layout()
    out_path = OUT_FIGURES_DIR / "rq2_shift_window_example_bmad.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    fig.savefig(PAPER_FIGURES_DIR / out_path.name, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def summarize_features_by_direction(feature_summary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    summaries: dict[str, pd.DataFrame] = {}
    for direction in ["increase", "decrease"]:
        subset = feature_summary[feature_summary["direction"] == direction].copy()
        if subset.empty:
            summaries[direction] = pd.DataFrame()
            continue

        subset["rank_within_event"] = subset.groupby("event_label")["importance_mean"].rank(
            method="average",
            ascending=False,
        )
        total_events = subset["event_label"].nunique()
        grouped = (
            subset.groupby(["feature", "family"], as_index=False)
            .agg(
                n_events=("event_label", "nunique"),
                mean_importance=("importance_mean", "mean"),
                median_importance=("importance_mean", "median"),
                mean_positive_share=("positive_share", "mean"),
                median_positive_share=("positive_share", "median"),
                mean_rank=("rank_within_event", "mean"),
                median_rank=("rank_within_event", "median"),
                top1_count=("rank_within_event", lambda s: int((s <= 1).sum())),
                top3_count=("rank_within_event", lambda s: int((s <= 3).sum())),
                top5_count=("rank_within_event", lambda s: int((s <= 5).sum())),
            )
            .copy()
        )
        grouped["presence_pct"] = 100.0 * grouped["n_events"] / total_events
        grouped["mean_positive_share_pct"] = 100.0 * grouped["mean_positive_share"]
        grouped["median_positive_share_pct"] = 100.0 * grouped["median_positive_share"]
        grouped["top1_pct"] = 100.0 * grouped["top1_count"] / grouped["n_events"]
        grouped["top3_pct"] = 100.0 * grouped["top3_count"] / grouped["n_events"]
        grouped["top5_pct"] = 100.0 * grouped["top5_count"] / grouped["n_events"]
        grouped = grouped.sort_values(
            ["mean_positive_share_pct", "top5_pct", "median_rank", "mean_importance", "feature"],
            ascending=[False, False, True, False, True],
        ).reset_index(drop=True)
        grouped["table_rank"] = np.arange(1, len(grouped) + 1)
        summaries[direction] = grouped
    return summaries


def direction_feature_longtable(df: pd.DataFrame, direction_label: str, label: str) -> list[str]:
    if df.empty:
        return [
            rf"\section*{{{direction_label} Events}}",
            "No events were available for this direction.",
            "",
        ]

    lines = [
        rf"\section*{{{direction_label} Events}}",
        r"\begin{center}",
        r"\tiny",
        r"\setlength{\LTleft}{0pt}",
        r"\setlength{\LTright}{0pt}",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\begin{longtable}{rp{2.9cm}p{2.4cm}rrrrrrrr}",
        rf"\caption{{Direction-specific RQ3 feature consensus for {direction_label.lower()} regime events.}}\label{{{label}}}\\",
        r"\toprule",
        r"\makecell{Rank} & \makecell{Feature} & \makecell{Family} & \makecell{Mean\\Rank} & \makecell{Median\\Rank} & \makecell{Mean Norm.\\Imp.\\(\%)} & \makecell{Median Norm.\\Imp.\\(\%)} & \makecell{Mean\\Imp.} & \makecell{Median\\Imp.} & \makecell{Pres.\\(\%)} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"\makecell{Rank} & \makecell{Feature} & \makecell{Family} & \makecell{Mean\\Rank} & \makecell{Median\\Rank} & \makecell{Mean Norm.\\Imp.\\(\%)} & \makecell{Median Norm.\\Imp.\\(\%)} & \makecell{Mean\\Imp.} & \makecell{Median\\Imp.} & \makecell{Pres.\\(\%)} \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{10}{r}{Continued on next page} \\",
        r"\midrule",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for _, row in df.iterrows():
        lines.append(
            " & ".join(
                [
                    fmt(row["table_rank"], digits=0),
                    r"\texttt{" + latex_escape(row["feature"]) + "}",
                    latex_escape(row["family"]),
                    fmt(row["mean_rank"]),
                    fmt(row["median_rank"]),
                    fmt(row["mean_positive_share_pct"]),
                    fmt(row["median_positive_share_pct"]),
                    fmt(row["mean_importance"], digits=4),
                    fmt(row["median_importance"], digits=4),
                    fmt(row["presence_pct"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\end{longtable}", r"\end{center}", ""])
    return lines


def write_tex(
    event_summary: pd.DataFrame,
    family_summary: pd.DataFrame,
    direction_feature_summaries: dict[str, pd.DataFrame],
    figure_paths: dict[str, Path],
    summary: dict[str, object],
) -> None:
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=0.7in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{longtable}",
        r"\usepackage{array}",
        r"\usepackage{graphicx}",
        r"\usepackage{makecell}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{lmodern}",
        r"\title{RQ2: Local Explanations Around Regime Shifts}",
        r"\author{}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        r"\section*{Motivation}",
        (
            "RQ2 examines the exact neighborhood of each accepted stable-region transition rather than the whole "
            "workflow history. The goal is to identify which predictors remain most useful when duration leaves one "
            "stable median, transitions sharply, and then stabilizes again."
        ),
        r"\section*{Approach}",
        (
            f"We analyzed all {summary['n_events']} regime events from the stable-region result. "
            f"For each event, we defined the shift-area test window as the last {int(WINDOW_QUANTILE * 100)}\\% "
            f"of runs from the old stable region and the first {int(WINDOW_QUANTILE * 100)}\\% of runs from the "
            f"new stable region, using the largest balanced size available on both sides. We then reused the "
            f"project's best RQ1 model selected by $R^2$, restricted the predictors to the globally retained RQ1 "
            f"feature set, fit that model on all runs occurring before the shift-area test window, and computed "
            f"permutation importance on the shift-area test window itself before aggregating the results into "
            f"feature families."
        ),
        r"\section*{Results}",
        (
            f"Across the {summary['n_events']} regime events, the local best-model analysis beat the lag-1 baseline "
            f"on {summary['events_beating_baseline']} events and underperformed it on {summary['events_not_beating_baseline']} "
            f"events. The mean local RMSE gain over the lag-1 baseline was {summary['mean_rmse_gain_pct']:.2f}\\%, "
            f"and the median gain was {summary['median_rmse_gain_pct']:.2f}\\%."
        ),
        (
            f"The most common dominant local explanation family was "
            f"{latex_escape(summary['most_common_top_family'])}, which led in "
            f"{summary['most_common_top_family_count']} events. "
            f"Patch-aware families appeared as the dominant explanation in "
            f"{summary['patch_dominant_event_count']} of the {summary['n_events']} events, showing that local "
            f"regime transitions are often better explained by concrete patch semantics than by a single static "
            f"global ranking."
        ),
        r"\begin{figure}[p]",
        r"\centering",
        rf"\includegraphics[width=\textwidth]{{figures/{figure_paths['heatmap'].name}}}",
        r"\caption{Normalized positive importance share of each explanation family within each local regime-shift window.}",
        r"\end{figure}",
        r"",
        r"\begin{figure}[p]",
        r"\centering",
        rf"\includegraphics[width=0.82\textwidth]{{figures/{figure_paths['counts'].name}}}",
        r"\caption{Dominant explanation family across the local regime-shift events.}",
        r"\end{figure}",
        r"",
        r"\begin{figure}[p]",
        r"\centering",
        rf"\includegraphics[width=0.82\textwidth]{{figures/{figure_paths['gain'].name}}}",
        r"\caption{Local RMSE gain over the lag-1 baseline versus regime-shift magnitude. Positive values indicate that the project-specific best model outperformed the baseline in the local shift window.}",
        r"\end{figure}",
        r"",
        r"\begin{center}",
        r"\small",
        r"\begin{longtable}{l l r r r r p{3.0cm} p{5.0cm}}",
        r"\caption{RQ2 event-level summary for the local regime-shift windows.}\label{tab:rq2-event-summary}\\",
        r"\toprule",
        r"Event & Dir. & Shift (\%) & RMSE & Base RMSE & Gain (\%) & Top family & Top 3 features \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Event & Dir. & Shift (\%) & RMSE & Base RMSE & Gain (\%) & Top family & Top 3 features \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{8}{r}{Continued on next page} \\",
        r"\midrule",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for _, row in event_summary.iterrows():
        lines.append(
            " & ".join(
                [
                    latex_escape(row["event_label"]),
                    latex_escape(row["direction"]),
                    fmt(row["pct_change"]),
                    fmt(row["local_rmse"]),
                    fmt(row["baseline_rmse"]),
                    fmt(row["rmse_gain_pct"]),
                    latex_escape(row["top_family"]),
                    " ; ".join(
                        r"\texttt{" + latex_escape(feature.strip()) + "}"
                        for feature in str(row["top_features"]).split(";")
                        if feature.strip()
                    ),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\end{longtable}",
            r"\end{center}",
            r"",
            r"\begin{table}[p]",
            r"\centering",
            r"\small",
        r"\caption{RQ2 family-level synthesis across local regime-shift windows.}",
        r"\label{tab:rq2-family-summary}",
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"Family & Events present & Mean share (\%) & Dominant count \\",
            r"\midrule",
        ]
    )
    for _, row in family_summary.iterrows():
        lines.append(
            " & ".join(
                [
                    latex_escape(row["family"]),
                    fmt(row["n_events"], digits=0),
                    fmt(row["mean_positive_share_pct"]),
                    fmt(row["top_family_count"], digits=0),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            r"",
            r"\section*{Direction-Specific Feature Rankings}",
            (
                "The next two tables aggregate local feature importance across shift events of the same direction. "
                "They mirror the feature-consensus idea used elsewhere in the paper, except that the grouping unit "
                "is now the regime-shift event instead of the project. Lower mean and median rank indicate a feature "
                "that repeatedly rises toward the top within its local shift window."
            ),
            r"",
        ]
    )
    lines.extend(
        direction_feature_longtable(
            direction_feature_summaries.get("increase", pd.DataFrame()),
            "Increase",
            "tab:rq3-increase-feature-consensus",
        )
    )
    lines.extend(
        direction_feature_longtable(
            direction_feature_summaries.get("decrease", pd.DataFrame()),
            "Decrease",
            "tab:rq3-decrease-feature-consensus",
        )
    )
    lines.extend(
        [
            r"\section*{Interpretation}",
            (
            "This local-window analysis complements the existing RQ2 results rather than replacing them. "
            "The global explanation tells us which feature groups matter over the whole project history, while "
            "RQ2 isolates the narrower, more operational question of which signals matter exactly when the "
            "workflow moves from one stable duration regime to another."
            ),
            r"\end{document}",
        ]
    )
    PAPER_TEX_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    best_models = load_best_models()
    kept_features = load_kept_features()
    events = load_regime_events()
    run_series = load_run_series()

    model_lookup = {str(row["project"]): str(row["model"]) for _, row in best_models.iterrows()}
    engineered_cache: dict[str, pd.DataFrame] = {}

    event_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    family_rows: list[dict[str, object]] = []

    for _, event in events.iterrows():
        project = str(event["project"])
        if project not in engineered_cache:
            engineered_cache[project] = load_engineered_with_regions(project, run_series)
        engineered = engineered_cache[project]
        local_df = select_local_window(event, engineered)
        test_start_run_order = float(local_df["run_order"].min())
        train_df = engineered[engineered["run_order"] < test_start_run_order].copy()
        if len(train_df) < 30:
            raise ValueError(
                f"Too few training rows before shift window for {project} event {event['event_id']}: {len(train_df)}"
            )

        usable_features = [
            feature
            for feature in kept_features
            if feature in local_df.columns
            and feature in train_df.columns
            and feature not in {"run_order", "region_id"}
            and train_df[feature].nunique(dropna=False) > 1
            and local_df[feature].nunique(dropna=False) > 1
        ]
        if len(usable_features) < 5:
            raise ValueError(f"Too few usable local features for {project} event {event['event_id']}: {len(usable_features)}")

        X_train = train_df[usable_features]
        y_train = train_df["build_duration"].astype(float)
        X_test = local_df[usable_features]
        y_test = local_df["build_duration"].astype(float).to_numpy()

        model_name = model_lookup[project]
        estimator = build_model_specs()[model_name]
        fitted = fit_model(estimator, X_train, y_train)
        y_pred = predict_model(fitted, X_test)
        metrics = compute_metrics(y_test, y_pred)

        baseline_pred = local_df["build_duration"].shift(1).astype(float).to_numpy()
        if np.isnan(baseline_pred).any():
            global_lag_lookup = engineered.set_index("run_order")["build_duration"].shift(1)
            missing_run_orders = local_df.loc[np.isnan(baseline_pred), "run_order"]
            baseline_pred[np.isnan(baseline_pred)] = (
                global_lag_lookup.reindex(missing_run_orders).astype(float).to_numpy()
            )
        baseline_metrics = compute_metrics(y_test, baseline_pred)
        rmse_gain_pct = (
            ((baseline_metrics["rmse"] - metrics["rmse"]) / baseline_metrics["rmse"]) * 100.0
            if baseline_metrics["rmse"] > 0
            else float("nan")
        )

        perm = permutation_importance(
            fitted,
            X_test,
            y_test,
            n_repeats=PERMUTATION_REPEATS,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        feature_df = pd.DataFrame(
            {
                "feature": usable_features,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        )
        feature_df["positive_importance"] = feature_df["importance_mean"].clip(lower=0.0)
        total_positive = float(feature_df["positive_importance"].sum())
        feature_df["positive_share"] = feature_df["positive_importance"] / total_positive if total_positive > 0 else 0.0
        feature_df["family"] = feature_df["feature"].map(infer_feature_group)

        family_df = (
            feature_df.groupby("family", as_index=False)[["positive_importance"]]
            .sum()
            .sort_values("positive_importance", ascending=False)
            .reset_index(drop=True)
        )
        family_positive = float(family_df["positive_importance"].sum())
        family_df["positive_share"] = family_df["positive_importance"] / family_positive if family_positive > 0 else 0.0

        event_label = f"{event['project_short']}-E{int(event['event_id'])}"
        top_feature, top_feature_share = top_nonzero_share(feature_df, "positive_share", "feature")
        top_features = top_n_labels(feature_df, "positive_share", "feature", n=3)
        top_family, top_family_share = top_nonzero_share(family_df, "positive_share", "family")

        event_rows.append(
            {
                "project": project,
                "project_short": event["project_short"],
                "event_id": int(event["event_id"]),
                "event_label": event_label,
                "direction": str(event["direction"]),
                "pct_change": float(event["pct_change"]),
                "abs_pct_change": abs(float(event["pct_change"])),
                "change_run_order": int(event["change_run_order"]),
                "change_date": event["change_date"],
                "before_region_id": int(event["from_region_id"]),
                "after_region_id": int(event["to_region_id"]),
                "train_runs": int(len(train_df)),
                "local_window_runs": int(len(local_df)),
                "balanced_runs_per_side": int(len(local_df) / 2),
                "model": model_name,
                "n_local_features": int(len(usable_features)),
                "local_rmse": metrics["rmse"],
                "local_mae": metrics["mae"],
                "local_r2": metrics["r2"],
                "local_nrmse": metrics["nrmse"],
                "baseline_rmse": baseline_metrics["rmse"],
                "baseline_nrmse": baseline_metrics["nrmse"],
                "rmse_gain_pct": rmse_gain_pct,
                "top_feature": top_feature,
                "top_feature_share": top_feature_share,
                "top_features": "; ".join(top_features),
                "top_family": top_family,
                "top_family_share": top_family_share,
            }
        )

        for _, row in feature_df.iterrows():
            feature_rows.append(
                {
                    "project": project,
                    "project_short": event["project_short"],
                    "event_id": int(event["event_id"]),
                    "event_label": event_label,
                    "direction": str(event["direction"]),
                    "feature": row["feature"],
                    "family": row["family"],
                    "importance_mean": float(row["importance_mean"]),
                    "importance_std": float(row["importance_std"]),
                    "positive_importance": float(row["positive_importance"]),
                    "positive_share": float(row["positive_share"]),
                }
            )
        for _, row in family_df.iterrows():
            family_rows.append(
                {
                    "project": project,
                    "project_short": event["project_short"],
                    "event_id": int(event["event_id"]),
                    "event_label": event_label,
                    "family": row["family"],
                    "positive_importance": float(row["positive_importance"]),
                    "positive_share": float(row["positive_share"]),
                }
            )

    event_summary = pd.DataFrame(event_rows).sort_values(["project_short", "event_id"]).reset_index(drop=True)
    feature_summary = pd.DataFrame(feature_rows)
    family_summary_long = pd.DataFrame(family_rows)

    family_consensus = (
        family_summary_long.groupby("family", as_index=False)
        .agg(
            n_events=("event_label", "nunique"),
            mean_positive_share=("positive_share", "mean"),
            median_positive_share=("positive_share", "median"),
        )
        .merge(
            event_summary["top_family"].value_counts().rename_axis("family").reset_index(name="top_family_count"),
            on="family",
            how="left",
        )
        .fillna({"top_family_count": 0})
        .sort_values(["top_family_count", "mean_positive_share"], ascending=[False, False])
        .reset_index(drop=True)
    )
    family_consensus["mean_positive_share_pct"] = 100.0 * family_consensus["mean_positive_share"]
    family_consensus["median_positive_share_pct"] = 100.0 * family_consensus["median_positive_share"]
    family_consensus["top_family_count"] = family_consensus["top_family_count"].astype(int)

    feature_consensus = (
        feature_summary.groupby(["feature", "family"], as_index=False)
        .agg(
            n_events=("event_label", "nunique"),
            mean_positive_share=("positive_share", "mean"),
            median_positive_share=("positive_share", "median"),
            top_feature_count=("positive_share", lambda s: int((s == s.max()).sum())),
        )
        .sort_values(["mean_positive_share", "n_events"], ascending=[False, False])
        .reset_index(drop=True)
    )
    direction_feature_summaries = summarize_features_by_direction(feature_summary)

    event_summary.to_csv(OUT_DIR / "event_summary.csv", index=False)
    feature_summary.to_csv(OUT_DIR / "event_feature_importances.csv", index=False)
    family_summary_long.to_csv(OUT_DIR / "event_family_importances.csv", index=False)
    family_consensus.to_csv(OUT_DIR / "family_consensus.csv", index=False)
    feature_consensus.to_csv(OUT_DIR / "feature_consensus.csv", index=False)
    for direction, df in direction_feature_summaries.items():
        df.to_csv(OUT_DIR / f"{direction}_feature_consensus.csv", index=False)

    figure_paths = {
        "heatmap": plot_family_heatmap(family_summary_long),
        "counts": plot_top_family_counts(event_summary),
        "gain": plot_gain_vs_shift(event_summary),
        "increase_feature_heatmap": plot_direction_feature_heatmap(feature_summary, "increase"),
        "decrease_feature_heatmap": plot_direction_feature_heatmap(feature_summary, "decrease"),
        "shift_window_example": plot_shift_window_example(run_series, events, project_short="bmad"),
    }

    top_family_counts = event_summary["top_family"].value_counts()
    patch_dominant_count = int(
        event_summary["top_family"].str.startswith("Patch", na=False).sum()
    )
    summary = {
        "n_events": int(len(event_summary)),
        "events_beating_baseline": int((event_summary["rmse_gain_pct"] > 0).sum()),
        "events_not_beating_baseline": int((event_summary["rmse_gain_pct"] <= 0).sum()),
        "mean_rmse_gain_pct": float(event_summary["rmse_gain_pct"].mean()),
        "median_rmse_gain_pct": float(event_summary["rmse_gain_pct"].median()),
        "most_common_top_family": str(top_family_counts.index[0]),
        "most_common_top_family_count": int(top_family_counts.iloc[0]),
        "patch_dominant_event_count": patch_dominant_count,
        "largest_gain_event": event_summary.sort_values("rmse_gain_pct", ascending=False).iloc[0].to_dict(),
        "largest_loss_event": event_summary.sort_values("rmse_gain_pct", ascending=True).iloc[0].to_dict(),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    write_tex(event_summary, family_consensus, direction_feature_summaries, figure_paths, summary)
    print(f"[RQ2] Wrote results to {OUT_DIR}")
    print(f"[RQ2] Wrote TeX report to {PAPER_TEX_PATH}")


if __name__ == "__main__":
    main()
