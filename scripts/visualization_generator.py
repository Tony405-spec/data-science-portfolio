from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data_loader import load_csv
from src.visualization import summarize_categories


def run(input_path: str | Path, field: str) -> dict[str, list[str] | list[int]]:
    rows = load_csv(input_path)
    labels, values = summarize_categories(rows, field)
    return {"labels": labels, "values": values}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate simple plot data.")
    parser.add_argument("input_path")
    parser.add_argument("field")
    parser.add_argument("--output", dest="output_path")
    args = parser.parse_args()

    payload = run(args.input_path, args.field)

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
