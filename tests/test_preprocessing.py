from src.preprocessing import clean_rows


def test_clean_rows_strips_and_filters():
    rows = [
        {"name": " Alice ", "score": "1"},
        {"name": "", "score": "2"},
    ]

    cleaned = clean_rows(rows, required_fields=["name"])

    assert cleaned == [{"name": "Alice", "score": "1"}]


def test_clean_rows_preserves_whitespace_when_disabled():
    rows = [{"name": " Alice ", "score": "1"}]

    cleaned = clean_rows(rows, strip_whitespace=False)

    assert cleaned == rows
