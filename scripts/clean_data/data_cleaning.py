from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SHARED_DIR = Path(__file__).resolve().parents[1] / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from conference_data import KEEP_EVENTS, MAX_DURATION_SEC

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
CLEANED_DIR = ROOT / "data" / "intermediate" / "cleaned"


def clean_runs(
    file_path: Path,
    output_path: Path | None = None,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    for col in ("created_at", "updated_at", "gh_first_commit_created_at"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if verbose:
        print(f"[CLEAN] {file_path.name}: before filtering -> {len(df):,} rows")

    if "conclusion" in df.columns:
        df = df[df["conclusion"] == "success"]

    if "workflow_event_trigger" in df.columns:
        df = df[df["workflow_event_trigger"].isin(KEEP_EVENTS)]

    before_cap = len(df)
    if "build_duration" in df.columns:
        df = df[df["build_duration"] <= MAX_DURATION_SEC].copy()

    if "created_at" in df.columns:
        df = df.sort_values("created_at").reset_index(drop=True)

    if verbose:
        print(
            f"[CLEAN] {file_path.name}: success/push-pr/<=399d -> {before_cap:,} to {len(df):,} rows"
        )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        if verbose:
            print(f"[CLEAN] saved -> {output_path}")

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create cleaned workflow histories.")
    parser.add_argument("--input-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=CLEANED_DIR)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.input_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found under {args.input_dir}")

    for path in paths:
        clean_runs(path, args.output_dir / path.name, verbose=not args.quiet)


if __name__ == "__main__":
    main()
