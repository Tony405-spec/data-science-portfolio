from src.data_loader import load_csv, write_csv


def test_load_csv_round_trip(tmp_path):
    rows = [
        {"name": "Alice", "score": "1"},
        {"name": "Bob", "score": "2"},
    ]
    file_path = tmp_path / "sample.csv"

    write_csv(file_path, rows)
    loaded = load_csv(file_path)

    assert loaded == rows
