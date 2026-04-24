from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "results" / "rq2_local_explanations"
OUTPUT_DIR = ROOT / "paper" / "generated_tables"


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


def render_table(df: pd.DataFrame, caption: str, label: str) -> str:
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\fontsize{9}{10}\selectfont",
        r"\tabcolsep=0.01cm",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\setlength{\tabcolsep}{2pt}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{clccc}",
        r"\toprule",
        r"\textbf{Rank} & \textbf{Feature} & \textbf{Mean Rank} & \textbf{Median Rank} & \textbf{Avg. Norm. Score (\%)} \\",
        r"\midrule",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            " & ".join(
                [
                    str(int(row.table_rank)),
                    r"\texttt{" + latex_escape(row.feature) + "}",
                    f"{float(row.mean_rank):.2f}",
                    f"{float(row.median_rank):.2f}",
                    f"{float(row.mean_positive_share_pct):.2f}",
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def write_top10(input_name: str, output_name: str, caption: str, label: str) -> None:
    df = pd.read_csv(INPUT_DIR / input_name).head(10).copy()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / output_name).write_text(render_table(df, caption, label), encoding="utf-8")


def main() -> None:
    write_top10(
        "increase_feature_consensus.csv",
        "rq2_increase_top10.tex",
        "Top 10 features associated with CI duration increase shifts.",
        "tab:rq2-increase-top10",
    )
    write_top10(
        "decrease_feature_consensus.csv",
        "rq2_decrease_top10.tex",
        "Top 10 features associated with CI duration decrease shifts",
        "tab:rq2-decrease-top10",
    )
    print(f"[RQ2-Tables] Wrote top-10 tables to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
