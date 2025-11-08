# label_map.py
# If dataset uses numeric labels, map them to text. Adjust to your dataset.
DEFAULT_LABEL_MAP = {
    0: "NEGATIVE",
    1: "POSITIVE",
    2: "NEUTRAL"
}

def map_label(label):
    # If label already string, return as-is
    if isinstance(label, str):
        return label
    return DEFAULT_LABEL_MAP.get(int(label), str(label))
