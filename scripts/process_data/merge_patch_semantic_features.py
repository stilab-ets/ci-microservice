from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "data" / "intermediate" / "cleaned_commit_enriched"
FEATURES_DIR = ROOT / "data" / "collected_patch_semantic_features"
OUTPUT_DIR = ROOT / "data" / "intermediate" / "cleaned_commit_patch_semantic_enriched"


def project_key(csv_path: Path) -> str:
    return csv_path.name.replace("_fixed.csv", "")


def fill_defaults(merged: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    for column in feature_df.columns:
        if column in {"sha", "repo_commit", "author_date"} or column not in merged.columns:
            continue
        if pd.api.types.is_numeric_dtype(feature_df[column]):
            merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0)
        else:
            merged[column] = merged[column].fillna("")
    return merged


def merge_project(enriched_csv: Path, features_dir: Path, output_dir: Path) -> dict[str, object]:
    key = project_key(enriched_csv)
    feature_csv = features_dir / f"{key}_patch_semantic_features.csv"
    if not feature_csv.exists():
        raise FileNotFoundError(f"Missing patch-semantic CSV for {key}: {feature_csv}")

    enriched_df = pd.read_csv(enriched_csv)
    feature_df = pd.read_csv(feature_csv)
    feature_df = feature_df.drop(columns=["repo_commit", "author_date"], errors="ignore")

    merged = enriched_df.merge(feature_df, how="left", left_on="commit_sha", right_on="sha")
    merged.drop(columns=["sha"], inplace=True, errors="ignore")
    merged = fill_defaults(merged, feature_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / enriched_csv.name
    merged.to_csv(output_path, index=False)
    return {
        "project": key,
        "input_rows": int(len(enriched_df)),
        "feature_rows": int(len(feature_df)),
        "output_rows": int(len(merged)),
        "output_csv": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge pre-computed patch-semantic features into the commit-enriched workflow histories."
    )
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--features-dir", type=Path, default=FEATURES_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--project", action="append", dest="projects")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.input_dir.glob("*_fixed.csv"))
    if args.projects:
        wanted = set(args.projects)
        paths = [path for path in paths if project_key(path) in wanted]
    if not paths:
        raise FileNotFoundError(f"No commit-enriched CSV files found under {args.input_dir}")

    summaries = [merge_project(path, args.features_dir, args.output_dir) for path in paths]
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(args.output_dir / "merge_patch_semantic_summary.csv", index=False)
    print(f"[Patch-Merge] Wrote {len(summary_df)} enriched workflow CSVs to {args.output_dir}")


if __name__ == "__main__":
    main()
