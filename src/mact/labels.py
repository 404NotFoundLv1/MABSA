"""Lossless five- or seven-label encoding of aspect/sentiment tuples."""

TAGS = ("O", "B-POS", "B-NEU", "B-NEG", "I")
TAGS_SEVEN = ("O", "B-POS", "B-NEU", "B-NEG", "I-POS", "I-NEU", "I-NEG")
POLARITY_TO_BEGIN = {0: 3, 1: 2, 2: 1}
BEGIN_TO_POLARITY = {v: k for k, v in POLARITY_TO_BEGIN.items()}


def get_tags(label_scheme="five"):
    if not isinstance(label_scheme, str) or label_scheme not in ("five", "seven"):
        raise ValueError("label_scheme must be 'five' or 'seven'")
    return TAGS if label_scheme == "five" else TAGS_SEVEN


def aspects_to_tags(n_words, aspects, label_scheme="five"):
    get_tags(label_scheme)
    if type(n_words) is not int or n_words < 1:
        raise ValueError("word sequences must be nonempty")
    tags = [0] * n_words
    for start, end, polarity in sorted(aspects):
        if (any(type(v) is not int for v in (start, end, polarity))
                or not 0 <= start < end <= n_words or polarity not in POLARITY_TO_BEGIN):
            raise ValueError("invalid aspect tuple")
        if any(tags[start:end]):
            raise ValueError("overlapping or duplicate aspect spans")
        tags[start] = POLARITY_TO_BEGIN[polarity]
        inside = 4 if label_scheme == "five" else tags[start] + 3
        tags[start + 1:end] = [inside] * (end - start - 1)
    return tags


def tags_to_aspects(tags, label_scheme="five"):
    num_tags = len(get_tags(label_scheme))
    result = []
    start, polarity = None, None
    for i, tag in enumerate(tags):
        if type(tag) is not int or tag not in range(num_tags):
            raise ValueError("invalid sequence tag")
        if tag >= 4:
            if start is None:
                raise ValueError("orphan I tag")
            if label_scheme == "seven" and tag != POLARITY_TO_BEGIN[polarity] + 3:
                raise ValueError("I tag polarity must match its B tag")
            continue
        if start is not None:
            result.append((start, i, polarity))
            start = None
        if tag in BEGIN_TO_POLARITY:
            start, polarity = i, BEGIN_TO_POLARITY[tag]
    if start is not None:
        result.append((start, len(tags), polarity))
    return result
