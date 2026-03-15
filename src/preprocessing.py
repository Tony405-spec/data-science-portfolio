from __future__ import annotations

from collections.abc import Iterable


def clean_rows(
    rows: Iterable[dict[str, str]],
    required_fields: Iterable[str] | None = None,
    strip_whitespace: bool = True,
) -> list[dict[str, str]]:
    """Clean rows by stripping whitespace and filtering required fields."""
    required = set(required_fields or [])
    cleaned_rows: list[dict[str, str]] = []

    for row in rows:
        cleaned_row: dict[str, str] = {}
        for key, value in row.items():
            if strip_whitespace and isinstance(value, str):
                cleaned_row[key] = value.strip()
            else:
                cleaned_row[key] = value

        if required and any(not cleaned_row.get(field) for field in required):
            continue

        cleaned_rows.append(cleaned_row)

    return cleaned_rows


def text_length_feature(rows: Iterable[dict[str, str]], field: str) -> list[dict[str, int]]:
    """Create a simple length-based feature for a text field."""
    features: list[dict[str, int]] = []
    feature_key = f"{field}_length"

    for row in rows:
        value = row.get(field, "")
        features.append({feature_key: len(str(value))})

    return features
