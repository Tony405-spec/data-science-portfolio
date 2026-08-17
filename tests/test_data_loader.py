import pytest

from src.data_loader import load_csv


def test_load_csv_validates_required_fields(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("name,score\nAda,10\n", encoding="utf-8")

    rows = load_csv(csv_path, required_fields=["name", "score"])

    assert rows == [{"name": "Ada", "score": "10"}]


def test_load_csv_reports_missing_required_fields(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("name\nAda\n", encoding="utf-8")

    with pytest.raises(ValueError, match="score"):
        load_csv(csv_path, required_fields=["name", "score"])


def test_load_csv_reports_missing_file(tmp_path):
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="CSV file not found"):
        load_csv(missing_path)
