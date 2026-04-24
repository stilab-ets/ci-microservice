from __future__ import annotations

import argparse
import sys
from pathlib import Path

SHARED_DIR = Path(__file__).resolve().parents[1] / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from feature_engineering import MODELING_DIR, engineer_features
from predictive_feature_config import DEFAULT_MODELING_INPUT_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare modeling datasets for RQ1/RQ2 from the enriched commit+patch source data."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_MODELING_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=MODELING_DIR)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.input_dir.glob("*_fixed.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found under {args.input_dir}")

    for path in paths:
        engineer_features(path, args.output_dir / path.name, verbose=not args.quiet)


if __name__ == "__main__":
    main()
