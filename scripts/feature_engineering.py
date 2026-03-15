from __future__ import annotations

import argparse
from pathlib import Path

from src.data_loader import load_csv, write_csv
from src.preprocessing import text_length_feature


def run(input_path: str | Path, output_path: str | Path, field: str) -> Path:
    rows = load_csv(input_path)
    engineered = text_length_feature(rows, field)
    return write_csv(output_path, engineered)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate simple text features.")
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("field")
    args = parser.parse_args()

    run(args.input_path, args.output_path, field=args.field)


if __name__ == "__main__":
    main()
