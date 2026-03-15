from src.preprocessing import clean_rows


def test_clean_rows_strips_and_filters():
    rows = [
        {"name": " Alice ", "score": "1"},
        {"name": "", "score": "2"},
    ]

    cleaned = clean_rows(rows, required_fields=["name"])

    assert cleaned == [{"name": "Alice", "score": "1"}]
