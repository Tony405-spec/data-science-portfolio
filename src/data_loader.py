from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


def load_csv(path: str | Path) -> list[dict[str, str]]:
    """Load a CSV file into a list of dictionaries."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def write_csv(
    path: str | Path, rows: Iterable[dict[str, str]], fieldnames: Iterable[str] | None = None
) -> Path:
    """Write a list of dictionaries to a CSV file."""
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows_list = list(rows)

    if rows_list:
        resolved_fields = list(fieldnames or rows_list[0].keys())
    else:
        resolved_fields = list(fieldnames or [])

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=resolved_fields)
        if resolved_fields:
            writer.writeheader()
        writer.writerows(rows_list)

    return csv_path
