"""Five-label, lossless encoding of non-overlapping aspect/sentiment tuples."""

TAGS = ("O", "B-POS", "B-NEU", "B-NEG", "I")
POLARITY_TO_BEGIN = {0: 3, 1: 2, 2: 1}
BEGIN_TO_POLARITY = {v: k for k, v in POLARITY_TO_BEGIN.items()}


def aspects_to_tags(n_words, aspects):
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
        tags[start + 1:end] = [4] * (end - start - 1)
    return tags


def tags_to_aspects(tags):
    result = []
    start, polarity = None, None
    for i, tag in enumerate(tags):
        if type(tag) is not int or tag not in range(5):
            raise ValueError("invalid sequence tag")
        if tag == 4:
            if start is None:
                raise ValueError("orphan I tag")
            continue
        if start is not None:
            result.append((start, i, polarity))
            start = None
        if tag in BEGIN_TO_POLARITY:
            start, polarity = i, BEGIN_TO_POLARITY[tag]
    if start is not None:
        result.append((start, len(tags), polarity))
    return result
