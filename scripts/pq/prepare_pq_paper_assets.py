from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PQ_RESULTS_DIR = ROOT / "results" / "pq" / "stable_regions"
PQ_FIGURES_DIR = PQ_RESULTS_DIR / "figures"
PAPER_FIGURES_DIR = ROOT / "paper" / "figures"
TABLE_OUT = ROOT / "paper" / "generated_tables" / "rq2_regime_shift_cases_table.tex"

FIGURE_MAP = {
    PQ_FIGURES_DIR / "overview" / "stable_region_counts.png": PAPER_FIGURES_DIR / "prelim_stable_region_counts.png",
    PQ_FIGURES_DIR / "pr3y_Bruce_wf121541665_stable_regions.png": PAPER_FIGURES_DIR / "prq_example_bruce_no_shift.png",
    PQ_FIGURES_DIR / "radareorg_radare2_wf1989843_stable_regions.png": PAPER_FIGURES_DIR / "prq_example_radare2_three_shifts.png",
    PQ_FIGURES_DIR / "Orange_OpenSourceouds_android_wf108176393_stable_regions.png": PAPER_FIGURES_DIR / "prq_example_ouds_android_five_regions.png",
}


def latex_escape(text: object) -> str:
    value = str(text)
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
        value = value.replace(src, dst)
    return value


def write_regime_table(events_csv: Path, output_path: Path) -> None:
    df = pd.read_csv(events_csv).sort_values(["project_short", "event_id"]).reset_index(drop=True)
    df["shift_label"] = df["from_region_id"].map(lambda x: f"R{int(x)}") + r"$\rightarrow$" + df["to_region_id"].map(lambda x: f"R{int(x)}")
    df["before_min"] = df["before_median_sec"] / 60.0
    df["after_min"] = df["after_median_sec"] / 60.0

    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\fontsize{7}{8}\selectfont",
        r"\tabcolsep=0.01cm",
        r"\caption{Accepted regime-shift cases analyzed in RQ2.}",
        r"\label{tab:rq2-regime-cases}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Project} & \textbf{Regime Shift} & \textbf{Regime 1 Median (min)} & \textbf{Regime 2 Median (min)} & \textbf{Change (\%)} \\",
        r"\midrule",
    ]
    for row in df.itertuples(index=False):
        cell_color = r"\cellcolor{green!18}" if float(row.pct_change) >= 0 else r"\cellcolor{red!18}"
        lines.append(
            f"{latex_escape(row.project_short)} & {row.shift_label} & {row.before_min:.2f} & {row.after_min:.2f} & {cell_color}{row.pct_change:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)

    for source, target in FIGURE_MAP.items():
        if not source.exists():
            raise FileNotFoundError(f"Missing PQ figure: {source}")
        shutil.copy2(source, target)

    write_regime_table(PQ_RESULTS_DIR / "regime_events.csv", TABLE_OUT)
    print(f"[PQ-Assets] Copied {len(FIGURE_MAP)} figures into {PAPER_FIGURES_DIR}")
    print(f"[PQ-Assets] Wrote {TABLE_OUT}")


if __name__ == "__main__":
    main()
