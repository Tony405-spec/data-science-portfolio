from __future__ import annotations

from collections.abc import Iterable


def text_length_feature(rows: Iterable[dict[str, str]], field: str) -> list[dict[str, int]]:
    """Create a simple length-based feature for a text field."""
    features: list[dict[str, int]] = []
    feature_key = f"{field}_length"

    for row in rows:
        value = row.get(field, "")
        features.append({feature_key: len(str(value))})

    return features
