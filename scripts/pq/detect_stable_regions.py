from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SHARED_DIR = Path(__file__).resolve().parents[1] / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from conference_data import load_filtered_runs, project_label

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results" / "pq" / "stable_regions"
FIGURES_DIR = RESULTS_DIR / "figures"
OVERVIEW_DIR = FIGURES_DIR / "overview"

MIN_SEGMENT_RUNS = 75
SIGNIFICANCE_LEVEL = 0.01
MAX_CHANGE_POINTS = 6
MIN_PCT_CHANGE_FOR_SPLIT = 10.0
ROLLING_WINDOW_SHARE = 0.05
MIN_ROLLING_WINDOW = 15
MAX_ROLLING_WINDOW = 101

PROJECT_SHORT_NAMES = {
    "bmad simbmad ecosystem_wf69576399": "bmad",
    "ccpay_wf6192976": "ccpay",
    "collinbarrettFilterLists_wf75763098": "FilterLists",
    "daos_wf9020028": "daos",
    "jod-yksilo-ui_wf83806327": "jod-yksilo-ui",
    "m2Gilesm2os_wf105026558": "m2os",
    "Orange_OpenSourceouds_android_wf108176393": "ouds-android",
    "pr3y_Bruce_wf121541665": "Bruce",
    "radareorg_radare2_wf1989843": "radare2",
    "rustlang_wf51073": "rust",
}


@dataclass
class PettittResult:
    change_index: int | None
    statistic: float
    p_value: float


def short_name(project: str) -> str:
    return PROJECT_SHORT_NAMES.get(project, project)


def rounded_metric(value: float) -> float:
    return round(value, 6) if np.isfinite(value) else value


def choose_rolling_window(run_count: int) -> int:
    window = int(round(run_count * ROLLING_WINDOW_SHARE))
    window = max(MIN_ROLLING_WINDOW, min(MAX_ROLLING_WINDOW, window))
    if window % 2 == 0:
        window += 1
    return min(window, run_count if run_count % 2 == 1 else max(1, run_count - 1))


def robust_stats(series: pd.Series) -> dict[str, float]:
    values = series.astype(float)
    mean_val = float(values.mean())
    median_val = float(values.median())
    std_val = float(values.std())
    mad_val = float((values - median_val).abs().median())
    iqr_val = float(values.quantile(0.75) - values.quantile(0.25))
    return {
        "runs": int(len(values)),
        "mean_sec": mean_val,
        "median_sec": median_val,
        "std_sec": std_val,
        "cv_pct": float((std_val / mean_val) * 100.0) if mean_val else np.nan,
        "mad_sec": mad_val,
        "mad_pct_of_median": float((mad_val / median_val) * 100.0) if median_val else np.nan,
        "iqr_sec": iqr_val,
        "iqr_pct_of_median": float((iqr_val / median_val) * 100.0) if median_val else np.nan,
        "p90_sec": float(values.quantile(0.90)),
        "p95_sec": float(values.quantile(0.95)),
        "max_sec": float(values.max()),
        "pct_over_30m": float((values > 1800).mean() * 100.0),
        "pct_over_60m": float((values > 3600).mean() * 100.0),
        "pct_over_1d": float((values > 86400).mean() * 100.0),
    }


def pettitt_test(values: pd.Series | np.ndarray) -> PettittResult:
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n < 2:
        return PettittResult(change_index=None, statistic=np.nan, p_value=np.nan)

    ranks = stats.rankdata(x)
    u_values = np.zeros(n)
    for t in range(n):
        u_values[t] = 2.0 * np.sum(ranks[: t + 1]) - (t + 1) * (n + 1)

    change_index = int(np.argmax(np.abs(u_values)))
    statistic = float(np.max(np.abs(u_values)))
    p_value = float(min(1.0, max(np.finfo(float).tiny, 2.0 * np.exp((-6.0 * statistic**2) / (n**3 + n**2)))))
    return PettittResult(change_index=change_index, statistic=statistic, p_value=p_value)


def recursive_pettitt_change_points(
    values: np.ndarray,
    offset: int = 0,
    min_segment_runs: int = MIN_SEGMENT_RUNS,
    significance_level: float = SIGNIFICANCE_LEVEL,
    min_pct_change_for_split: float = MIN_PCT_CHANGE_FOR_SPLIT,
    splits_remaining: int = MAX_CHANGE_POINTS,
) -> list[int]:
    if len(values) < 2 * min_segment_runs or splits_remaining <= 0:
        return []

    result = pettitt_test(values)
    if result.change_index is None or not np.isfinite(result.p_value) or result.p_value >= significance_level:
        return []

    left_len = result.change_index + 1
    right_len = len(values) - left_len
    if left_len < min_segment_runs or right_len < min_segment_runs:
        return []

    left_median = float(np.median(values[:left_len]))
    right_median = float(np.median(values[left_len:]))
    pct_change = float(((right_median / left_median) - 1.0) * 100.0) if left_median > 0 else np.nan
    if not np.isfinite(pct_change) or abs(pct_change) < min_pct_change_for_split:
        return []

    global_change_index = offset + result.change_index
    left_points = recursive_pettitt_change_points(
        values[: left_len],
        offset=offset,
        min_segment_runs=min_segment_runs,
        significance_level=significance_level,
        min_pct_change_for_split=min_pct_change_for_split,
        splits_remaining=splits_remaining - 1,
    )
    right_points = recursive_pettitt_change_points(
        values[left_len:],
        offset=global_change_index + 1,
        min_segment_runs=min_segment_runs,
        significance_level=significance_level,
        min_pct_change_for_split=min_pct_change_for_split,
        splits_remaining=splits_remaining - 1 - len(left_points),
    )
    return sorted(left_points + [global_change_index] + right_points)


def build_run_order_series(df: pd.DataFrame) -> pd.DataFrame:
    series = df.sort_values("created_at").reset_index(drop=True).copy()
    series["run_order"] = np.arange(1, len(series) + 1)
    window = choose_rolling_window(len(series))
    rolling = series["build_duration"].astype(float).rolling(window=window, center=True, min_periods=max(5, window // 3))
    series["rolling_median_sec"] = rolling.median()
    series["rolling_window_runs"] = window
    return series


def build_stable_regions(project: str, run_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = run_df["build_duration"].astype(float).to_numpy()
    change_points = recursive_pettitt_change_points(values)
    boundaries = [-1] + change_points + [len(run_df) - 1]

    while len(boundaries) > 2:
        small_gap_candidates: list[tuple[float, int]] = []
        for idx in range(1, len(boundaries) - 1):
            left = values[boundaries[idx - 1] + 1 : boundaries[idx] + 1]
            right = values[boundaries[idx] + 1 : boundaries[idx + 1] + 1]
            left_median = float(np.median(left))
            right_median = float(np.median(right))
            pct_change = float(((right_median / left_median) - 1.0) * 100.0) if left_median > 0 else np.nan
            if np.isfinite(pct_change) and abs(pct_change) < MIN_PCT_CHANGE_FOR_SPLIT:
                small_gap_candidates.append((abs(pct_change), idx))
        if not small_gap_candidates:
            break
        _, boundary_idx_to_remove = min(small_gap_candidates, key=lambda item: item[0])
        boundaries.pop(boundary_idx_to_remove)

    region_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    region_medians: list[float] = []

    for region_id, (start_idx, end_idx) in enumerate(zip(boundaries[:-1], boundaries[1:]), start=1):
        region = run_df.iloc[start_idx + 1 : end_idx + 1].copy()
        stats_dict = robust_stats(region["build_duration"])
        region_medians.append(float(stats_dict["median_sec"]))
        region_rows.append(
            {
                "project": project,
                "project_short": short_name(project),
                "region_id": region_id,
                "start_index_zero_based": int(start_idx + 1),
                "end_index_zero_based": int(end_idx),
                "start_run_order": int(region["run_order"].iloc[0]),
                "end_run_order": int(region["run_order"].iloc[-1]),
                "start_date": region["created_at"].iloc[0].isoformat(),
                "end_date": region["created_at"].iloc[-1].isoformat(),
                "length_runs": int(len(region)),
                "length_days": rounded_metric(
                    float((region["created_at"].iloc[-1] - region["created_at"].iloc[0]).total_seconds() / 86400.0)
                ),
                **{key: rounded_metric(value) for key, value in stats_dict.items()},
            }
        )

    for region_id in range(1, len(region_rows)):
        previous_row = region_rows[region_id - 1]
        current_row = region_rows[region_id]
        before_median = float(previous_row["median_sec"])
        after_median = float(current_row["median_sec"])
        pct_change = float(((after_median / before_median) - 1.0) * 100.0) if before_median > 0 else np.nan
        event_rows.append(
            {
                "project": project,
                "project_short": short_name(project),
                "event_id": region_id,
                "from_region_id": int(previous_row["region_id"]),
                "to_region_id": int(current_row["region_id"]),
                "change_run_order": int(current_row["start_run_order"]),
                "change_date": current_row["start_date"],
                "before_median_sec": rounded_metric(before_median),
                "after_median_sec": rounded_metric(after_median),
                "median_change_sec": rounded_metric(after_median - before_median),
                "pct_change": rounded_metric(pct_change),
                "direction": "increase" if pct_change >= 0 else "decrease",
                "new_region_length_runs": int(current_row["length_runs"]),
                "new_region_length_days": current_row["length_days"],
            }
        )

    regions_df = pd.DataFrame(region_rows)
    events_df = pd.DataFrame(
        event_rows,
        columns=[
            "project",
            "project_short",
            "event_id",
            "from_region_id",
            "to_region_id",
            "change_run_order",
            "change_date",
            "before_median_sec",
            "after_median_sec",
            "median_change_sec",
            "pct_change",
            "direction",
            "new_region_length_runs",
            "new_region_length_days",
        ],
    )
    return regions_df, events_df


def build_workflow_summary(project: str, run_df: pd.DataFrame, regions_df: pd.DataFrame, events_df: pd.DataFrame) -> dict[str, object]:
    raw_stats = robust_stats(run_df["build_duration"])
    if events_df.empty:
        largest_up = np.nan
        largest_down = np.nan
        largest_abs = np.nan
    else:
        largest_up = float(events_df.loc[events_df["pct_change"] >= 0, "pct_change"].max()) if (events_df["pct_change"] >= 0).any() else np.nan
        largest_down = float(events_df.loc[events_df["pct_change"] < 0, "pct_change"].min()) if (events_df["pct_change"] < 0).any() else np.nan
        largest_abs = float(events_df["pct_change"].abs().max())

    region_cv = regions_df["cv_pct"].astype(float) if not regions_df.empty else pd.Series(dtype=float)
    region_mad_pct = regions_df["mad_pct_of_median"].astype(float) if not regions_df.empty else pd.Series(dtype=float)
    region_lengths = regions_df["length_runs"].astype(float) if not regions_df.empty else pd.Series(dtype=float)
    region_medians = regions_df["median_sec"].astype(float) if not regions_df.empty else pd.Series(dtype=float)
    between_region_range_pct = (
        float(((region_medians.max() / region_medians.min()) - 1.0) * 100.0)
        if not region_medians.empty and float(region_medians.min()) > 0
        else np.nan
    )

    return {
        "project": project,
        "project_short": short_name(project),
        **{key: rounded_metric(value) for key, value in raw_stats.items()},
        "stable_regions": int(len(regions_df)),
        "regime_events": int(len(events_df)),
        "median_region_length_runs": rounded_metric(float(region_lengths.median())) if not region_lengths.empty else np.nan,
        "median_within_region_cv_pct": rounded_metric(float(region_cv.median())) if not region_cv.empty else np.nan,
        "median_within_region_mad_pct": rounded_metric(float(region_mad_pct.median())) if not region_mad_pct.empty else np.nan,
        "between_region_median_range_pct": rounded_metric(between_region_range_pct),
        "largest_upward_shift_pct": rounded_metric(largest_up),
        "largest_downward_shift_pct": rounded_metric(largest_down),
        "largest_absolute_shift_pct": rounded_metric(largest_abs),
    }


def plot_workflow(project: str, run_df: pd.DataFrame, regions_df: pd.DataFrame, events_df: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    display_name = short_name(project)
    fig, ax = plt.subplots(figsize=(15.5, 7.8))

    valid_median = run_df["rolling_median_sec"].notna()
    line_parts: list[np.ndarray] = []
    if valid_median.any():
        line_parts.append(run_df.loc[valid_median, "rolling_median_sec"].astype(float).to_numpy())
    if not regions_df.empty:
        line_parts.append(regions_df["median_sec"].astype(float).to_numpy())

    if line_parts:
        line_values = np.concatenate([part[np.isfinite(part)] for part in line_parts if len(part) > 0])
        y_min = float(np.min(line_values))
        y_max = float(np.max(line_values))
    else:
        durations = run_df["build_duration"].astype(float).to_numpy()
        y_min = float(np.min(durations))
        y_max = float(np.max(durations))

    y_span = max(y_max - y_min, max(y_max * 0.08, 1.0))
    y_floor = max(0.0, y_min - 0.12 * y_span)
    y_cap = y_max + 0.24 * y_span

    visible_runs = run_df[run_df["build_duration"].astype(float).between(y_floor, y_cap, inclusive="both")]
    hidden_count = int(len(run_df) - len(visible_runs))

    ax.scatter(
        visible_runs["run_order"],
        visible_runs["build_duration"],
        s=14,
        alpha=0.22,
        color="#A0A0A0",
        label="Run duration",
    )

    if valid_median.any():
        ax.plot(
            run_df.loc[valid_median, "run_order"],
            run_df.loc[valid_median, "rolling_median_sec"],
            color="#4E79A7",
            linewidth=2.8,
            label=f"Rolling median ({int(run_df['rolling_window_runs'].iloc[0])} runs)",
        )

    region_top_y = y_floor + 0.86 * (y_cap - y_floor)
    region_bottom_y = y_floor + 0.14 * (y_cap - y_floor)
    project_region_label_slots = {
        "pr3y_Bruce_wf121541665": {
            1: {"x_kind": "run_order", "x": 276, "y_kind": "axes_frac", "y": 0.86},
        },
        "radareorg_radare2_wf1989843": {
            1: {"x_kind": "run_order", "x": 760, "y_kind": "axes_frac", "y": 0.86},
            2: {"x_kind": "run_order", "x": 1735, "y_kind": "axes_frac", "y": 0.28},
            3: {"x_kind": "run_order", "x": 2090, "y_kind": "axes_frac", "y": 0.78},
            4: {"x_kind": "run_order", "x": 2255, "y_kind": "axes_frac", "y": 0.28},
        },
        "Orange_OpenSourceouds_android_wf108176393": {
            1: {"x_kind": "run_order", "x": 18, "y_kind": "axes_frac", "y": 0.13},
            2: {"x_kind": "run_order", "x": 194, "y_kind": "axes_frac", "y": 0.18},
            3: {"x_kind": "run_order", "x": 600, "y_kind": "axes_frac", "y": 0.86},
            4: {"x_kind": "run_order", "x": 1435, "y_kind": "axes_frac", "y": 0.15},
            5: {"x_kind": "run_order", "x": 1965, "y_kind": "axes_frac", "y": 0.15},
        },
    }
    for idx, region in regions_df.reset_index(drop=True).iterrows():
        color = "#F28E2B" if idx % 2 == 0 else "#EDC948"
        ax.hlines(
            y=float(region["median_sec"]),
            xmin=int(region["start_run_order"]),
            xmax=int(region["end_run_order"]),
            color=color,
            linewidth=4.6,
            alpha=0.95,
            label="Stable-region median" if idx == 0 else None,
        )
        start_run = int(region["start_run_order"])
        end_run = int(region["end_run_order"])
        midpoint = (start_run + end_run) / 2.0
        region_id = int(region["region_id"]) if "region_id" in region else idx + 1
        manual_slot = project_region_label_slots.get(project, {}).get(region_id)
        if manual_slot:
            if manual_slot["x_kind"] == "run_order":
                label_x = float(manual_slot["x"])
            else:
                label_x = start_run + float(manual_slot["x"]) * max(end_run - start_run, 1)
            if manual_slot["y_kind"] == "axes_frac":
                label_y = y_floor + float(manual_slot["y"]) * (y_cap - y_floor)
            else:
                label_y = float(manual_slot["y"])
        else:
            label_x = midpoint
            x_frac = (midpoint - float(run_df["run_order"].min())) / max(
                float(run_df["run_order"].max() - run_df["run_order"].min()), 1.0
            )
            place_top = idx % 2 == 0
            if x_frac < 0.22:
                place_top = False
            if x_frac > 0.86:
                place_top = False
            label_y = region_top_y if place_top else region_bottom_y
        ax.annotate(
            f"{float(region['median_sec'])/60.0:.1f}m",
            xy=(midpoint, float(region["median_sec"])),
            xytext=(label_x, label_y),
            textcoords="data",
            ha="center",
            va="center",
            fontsize=24,
            fontweight="bold",
            color="#222222",
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": color, "alpha": 0.94},
            arrowprops={"arrowstyle": "-", "color": color, "lw": 1.2, "alpha": 0.85},
        )

    for _, event in events_df.iterrows():
        run_order = int(event["change_run_order"])
        color = "#E15759" if event["pct_change"] >= 0 else "#59A14F"
        ax.axvline(run_order, color=color, linestyle=":", linewidth=2.2, alpha=0.92)

    ax.set_xlabel("Workflow runs in chronological order", fontsize=20)
    ax.set_ylabel("Build duration (seconds)", fontsize=20)
    ax.set_ylim(y_floor, y_cap)
    ax.tick_params(axis="both", labelsize=17)
    if hidden_count > 0:
        ax.text(
            0.99,
            0.965,
            f"{hidden_count} runs outside zoomed y-range",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=13,
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#999999", "alpha": 0.92},
        )
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99), borderaxespad=0.2, fontsize=15, framealpha=0.92)
    fig.tight_layout(pad=1.0)
    fig.savefig(FIGURES_DIR / f"{project}_stable_regions.png", dpi=220, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def plot_overview(summary_df: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    OVERVIEW_DIR.mkdir(parents=True, exist_ok=True)

    ordered_regions = summary_df.sort_values("stable_regions", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 6.6))
    bars = ax.bar(ordered_regions["project_short"], ordered_regions["stable_regions"], color="#4E79A7")
    ax.set_ylabel("Stable regions", fontsize=20)
    ax.tick_params(axis="x", labelsize=17)
    ax.tick_params(axis="y", labelsize=17)
    ax.set_ylim(0, float(ordered_regions["stable_regions"].max()) + 1.6)
    plt.xticks(rotation=45, ha="right")
    for bar, value in zip(bars, ordered_regions["stable_regions"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.08,
            f"{int(value)}",
            ha="center",
            va="bottom",
            fontsize=18,
            fontweight="bold",
            color="#222222",
        )
    fig.tight_layout()
    fig.savefig(OVERVIEW_DIR / "stable_region_counts.png", dpi=220)
    plt.close(fig)

    ordered_shifts = summary_df.sort_values("largest_absolute_shift_pct", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(ordered_shifts["project_short"], ordered_shifts["largest_absolute_shift_pct"], color="#E15759")
    ax.set_ylabel("Largest absolute median shift (%)")
    ax.set_title("Largest regime change per workflow")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(OVERVIEW_DIR / "largest_shift_ranking.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(
        summary_df["median_within_region_mad_pct"],
        summary_df["between_region_median_range_pct"],
        s=np.maximum(summary_df["stable_regions"].astype(float), 1.0) * 55.0,
        color="#59A14F",
        alpha=0.8,
    )
    for _, row in summary_df.iterrows():
        ax.annotate(row["project_short"], (row["median_within_region_mad_pct"], row["between_region_median_range_pct"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Median within-region MAD (% of region median)")
    ax.set_ylabel("Between-region median range (%)")
    ax.set_title("Within-region volatility vs between-region structural range")
    fig.tight_layout()
    fig.savefig(OVERVIEW_DIR / "volatility_vs_structural_range.png", dpi=220)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    OVERVIEW_DIR.mkdir(parents=True, exist_ok=True)

    run_frames: list[pd.DataFrame] = []
    region_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for path in sorted(RAW_DIR.glob("*.csv")):
        project = project_label(path)
        df = load_filtered_runs(path)
        run_df = build_run_order_series(df)
        regions_df, events_df = build_stable_regions(project, run_df)
        summary_rows.append(build_workflow_summary(project, run_df, regions_df, events_df))

        run_df = run_df.copy()
        run_df["project"] = project
        run_df["project_short"] = short_name(project)

        region_lookup = {}
        for _, region in regions_df.iterrows():
            for run_order in range(int(region["start_run_order"]), int(region["end_run_order"]) + 1):
                region_lookup[run_order] = int(region["region_id"])
        run_df["region_id"] = run_df["run_order"].map(region_lookup)

        run_frames.append(run_df)
        region_frames.append(regions_df)
        event_frames.append(events_df)
        plot_workflow(project, run_df, regions_df, events_df)

    run_series_df = pd.concat(run_frames, ignore_index=True)
    stable_regions_df = pd.concat(region_frames, ignore_index=True)
    non_empty_event_frames = [frame for frame in event_frames if not frame.empty]
    if non_empty_event_frames:
        regime_events_df = pd.concat(non_empty_event_frames, ignore_index=True)
    else:
        regime_events_df = pd.DataFrame(
            columns=[
                "project",
                "project_short",
                "event_id",
                "from_region_id",
                "to_region_id",
                "change_run_order",
                "change_date",
                "before_median_sec",
                "after_median_sec",
                "median_change_sec",
                "pct_change",
                "direction",
                "new_region_length_runs",
                "new_region_length_days",
            ]
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("project_short").reset_index(drop=True)

    run_series_df.to_csv(RESULTS_DIR / "run_series_with_regions.csv", index=False)
    stable_regions_df.to_csv(RESULTS_DIR / "stable_regions.csv", index=False)
    regime_events_df.to_csv(RESULTS_DIR / "regime_events.csv", index=False)
    summary_df.to_csv(RESULTS_DIR / "workflow_stability_summary.csv", index=False)

    top_findings = {
        "most_segmented_workflow": summary_df.sort_values("stable_regions", ascending=False).iloc[0].to_dict(),
        "largest_structural_shift": summary_df.sort_values("largest_absolute_shift_pct", ascending=False).iloc[0].to_dict(),
        "most_within_region_volatile": summary_df.sort_values("median_within_region_mad_pct", ascending=False).iloc[0].to_dict(),
    }
    with (RESULTS_DIR / "top_findings.json").open("w", encoding="utf-8") as handle:
        json.dump(top_findings, handle, indent=2, default=str)

    plot_overview(summary_df)


if __name__ == "__main__":
    main()
