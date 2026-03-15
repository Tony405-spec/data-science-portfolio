from __future__ import annotations

import argparse
from pathlib import Path

from src.data_loader import load_csv, write_csv
from src.preprocessing import clean_rows


def run(input_path: str | Path, output_path: str | Path, required_fields: list[str] | None = None) -> Path:
    rows = load_csv(input_path)
    cleaned = clean_rows(rows, required_fields=required_fields)
    return write_csv(output_path, cleaned, fieldnames=rows[0].keys() if rows else None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean CSV data.")
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("--required-field", action="append", dest="required_fields")
    args = parser.parse_args()

    run(args.input_path, args.output_path, required_fields=args.required_fields)


if __name__ == "__main__":
    main()
