from __future__ import annotations

import argparse
from pathlib import Path

from src.data_loader import load_csv


def train_baseline_model(rows: list[dict[str, str]], target_field: str) -> str:
    if not rows:
        raise ValueError("Training data is empty.")

    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(target_field, "")
        counts[value] = counts.get(value, 0) + 1

    return max(counts, key=counts.get)


def run(input_path: str | Path, target_field: str) -> str:
    rows = load_csv(input_path)
    return train_baseline_model(rows, target_field)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a baseline frequency model.")
    parser.add_argument("input_path")
    parser.add_argument("target_field")
    args = parser.parse_args()

    prediction = run(args.input_path, target_field=args.target_field)
    print(f"Most frequent {args.target_field}: {prediction}")


if __name__ == "__main__":
    main()
