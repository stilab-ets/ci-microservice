from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "frozen_paper_inputs" / "cleaned_commit_patch_semantic_enriched" / "daos_wf9020028_fixed.csv"
EVENTS_PATH = ROOT / "results" / "pq" / "stable_regions" / "regime_events.csv"
PAPER_FIGURES_DIR = ROOT / "paper" / "figures"
OUT_PATH = PAPER_FIGURES_DIR / "motivating_example_daos_boundary_shift.png"

BOUNDARY_BUILD_ID = "15270386539"
WINDOW_RADIUS = 10


def load_daos_history() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df = df.sort_values("created_at").reset_index(drop=True)
    df["run_order"] = range(1, len(df) + 1)
    df["duration_min"] = df["build_duration"].astype(float) / 60.0
    return df


def load_boundary_metadata() -> dict[str, float]:
    events = pd.read_csv(EVENTS_PATH)
    event = events[(events["project"] == "daos_wf9020028") & (events["event_id"] == 1)].iloc[0]
    return {
        "change_run_order": float(event["change_run_order"]),
        "before_median_min": float(event["before_median_sec"]) / 60.0,
        "after_median_min": float(event["after_median_sec"]) / 60.0,
    }


def main() -> None:
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_daos_history()
    meta = load_boundary_metadata()

    boundary_row = df[df["id_build"].astype(str) == BOUNDARY_BUILD_ID]
    if boundary_row.empty:
        raise ValueError(f"Boundary build {BOUNDARY_BUILD_ID} was not found in {DATA_PATH}.")

    boundary_run_order = int(boundary_row.iloc[0]["run_order"])
    left = max(1, boundary_run_order - WINDOW_RADIUS)
    right = boundary_run_order + WINDOW_RADIUS
    window = df[(df["run_order"] >= left) & (df["run_order"] <= right)].copy()

    fig, ax = plt.subplots(figsize=(10.4, 4.8))

    ax.axvspan(boundary_run_order - 0.5, right + 0.5, color="#fff1e6", alpha=0.65, zorder=0)
    ax.axhline(
        meta["before_median_min"],
        color="#355070",
        linestyle="--",
        linewidth=1.6,
        alpha=0.85,
        label="Earlier stable level",
        zorder=1,
    )
    ax.axhline(
        meta["after_median_min"],
        color="#2a9d8f",
        linestyle="--",
        linewidth=1.6,
        alpha=0.85,
        label="Later stable level",
        zorder=1,
    )
    ax.plot(
        window["run_order"],
        window["duration_min"],
        color="#4a4e69",
        linewidth=1.8,
        marker="o",
        markersize=5.5,
        zorder=2,
    )

    boundary_duration = float(boundary_row.iloc[0]["duration_min"])
    ax.scatter(
        [boundary_run_order],
        [boundary_duration],
        color="#d62828",
        s=95,
        edgecolors="white",
        linewidths=1.0,
        zorder=3,
        label="Boundary run",
    )
    ax.axvline(boundary_run_order, color="#d62828", linestyle=":", linewidth=1.8, alpha=0.9, zorder=2)

    ymax = max(float(window["duration_min"].max()), boundary_duration)
    ax.annotate(
        "Boundary run\n52.3 min",
        xy=(boundary_run_order, boundary_duration),
        xytext=(boundary_run_order + 2.2, ymax + 1.4),
        fontsize=13,
        ha="left",
        va="bottom",
        color="#7f1d1d",
        arrowprops={"arrowstyle": "->", "color": "#7f1d1d", "lw": 1.5},
    )
    ax.text(
        left + 0.2,
        meta["before_median_min"] + 0.55,
        "Earlier stable level: 29.6 min",
        fontsize=12.5,
        color="#355070",
        ha="left",
        va="bottom",
    )
    ax.text(
        boundary_run_order + 0.8,
        meta["after_median_min"] + 0.55,
        "Later stable level: 39.5 min",
        fontsize=12.5,
        color="#2a9d8f",
        ha="left",
        va="bottom",
    )

    ax.set_xlabel("Workflow run order", fontsize=18)
    ax.set_ylabel("Workflow duration (minutes)", fontsize=18)
    tick_values = [left + 1, left + 3, left + 5, left + 7, boundary_run_order, boundary_run_order + 4, boundary_run_order + 6, boundary_run_order + 8, right]
    ax.set_xticks(tick_values)
    ax.set_xticklabels([str(value) for value in tick_values])
    ax.tick_params(axis="both", labelsize=15)
    ax.set_xlim(left - 0.5, right + 0.5)
    ax.set_ylim(0, ymax + 7.0)
    ax.grid(axis="y", alpha=0.22, linewidth=0.6)
    ax.legend(loc="upper left", fontsize=12, frameon=True)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
