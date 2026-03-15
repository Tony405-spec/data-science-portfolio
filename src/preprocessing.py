def text_length_feature(rows, key):
    """
    Computes the length of the specified key in each row.
    Returns a list of dictionaries containing the length under the new key.
    """
    return [{f"{key}_length": len(row[key])} for row in rows]
