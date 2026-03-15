from src.features import text_length_feature


def test_text_length_feature():
    rows = [{"name": "Amy"}, {"name": "Bobby"}]

    features = text_length_feature(rows, "name")

    assert features == [{"name_length": 3}, {"name_length": 5}]


def test_text_length_feature_with_missing_field():
    rows = [{"name": "Amy"}, {}]

    features = text_length_feature(rows, "name")

    assert features == [{"name_length": 3}, {"name_length": 0}]
