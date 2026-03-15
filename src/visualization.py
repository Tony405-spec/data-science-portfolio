from __future__ import annotations

from collections.abc import Iterable


def summarize_categories(rows: Iterable[dict[str, str]], field: str) -> tuple[list[str], list[int]]:
    """Summarize categorical counts for lightweight plotting."""
    counts: dict[str, int] = {}

    for row in rows:
        value = row.get(field)
        if value is None:
            continue
        key = str(value)
        counts[key] = counts.get(key, 0) + 1

    labels = sorted(counts)
    values = [counts[label] for label in labels]

    return labels, values
