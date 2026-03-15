from src.preprocessing import text_length_feature


def test_text_length_feature():
    rows = [{"name": "Amy"}, {"name": "Bobby"}]

    features = text_length_feature(rows, "name")

    assert features == [{"name_length": 3}, {"name_length": 5}]
