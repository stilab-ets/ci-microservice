from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = ROOT / "data" / "collected_metadata" / "project_overview.csv"
OUTPUT_TEX = ROOT / "paper" / "generated_tables" / "project_overview_table.tex"

LABEL_MAP = {
    "ouds-android": "ouds-android",
    "bmad-ecosystem": "bmad",
    "ccpay-payment-app": "ccpay",
    "FilterLists": "FilterLists",
    "daos": "daos",
    "jod-yksilo-ui": "jod-yksilo-ui",
    "m2os": "m2os",
    "Bruce": "Bruce",
    "radare2": "radare2",
    "crates.io": "rust",
}

PROJECT_ORDER = [
    "ouds-android",
    "bmad",
    "ccpay",
    "FilterLists",
    "daos",
    "jod-yksilo-ui",
    "m2os",
    "Bruce",
    "radare2",
    "rust",
]


def make_latex_table(df: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[!tbp]",
        r"\centering",
        r"\fontsize{7.2}{8}\selectfont",
        r"\tabcolsep=0.05cm",
        r"\caption{Selected microservices-based projects summary.}",
        r"\label{tab:projects_overview}",
        r"\renewcommand{\arraystretch}{1.1}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"\textbf{Project} &",
        r"\makecell{\textbf{Primary}\\\textbf{Language}} &",
        r"\makecell{\textbf{Source}\\\textbf{Lines}\\\textbf{of Code}} &",
        r"\makecell{\textbf{Total}\\\textbf{Workflow}\\\textbf{Runs}} &",
        r"\makecell{\textbf{Workflow}\\\textbf{Lifetime}\\\textbf{(days)}} &",
        r"\makecell{\textbf{Median}\\\textbf{Duration}\\\textbf{(min)}} &",
        r"\makecell{\textbf{Duration}\\\textbf{MAD}\\\textbf{(min)}} \\",
        r"\midrule",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            f"{row.project_label} & {row.primary_language} & {row.sloc_display} & "
            f"{row.total_workflow_runs_display} & {row.workflow_lifetime_days:.1f} & "
            f"{row.median_duration_min:.2f} & {row.median_absolute_deviation_min:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    df["project_label"] = df["project_label"].map(LABEL_MAP).fillna(df["project_label"])
    order_map = {name: idx for idx, name in enumerate(PROJECT_ORDER)}
    df["sort_key"] = df["project_label"].map(lambda value: order_map.get(value, 999))
    df = df.sort_values(["sort_key", "project_label"]).drop(columns=["sort_key"]).reset_index(drop=True)
    OUTPUT_TEX.write_text(make_latex_table(df), encoding="utf-8")
    print(f"[Project-Overview] Wrote {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
