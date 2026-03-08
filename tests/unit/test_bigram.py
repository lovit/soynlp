import pytest

from soynlp.word import BigramExtractor


@pytest.mark.parametrize(
    ("method", "threshold"),
    [
        ("frequency", 1),
        ("pmi", 1.0),
        ("mikolov", 0.1),
    ],
)
def test_bigram_extractor(method, threshold):
    train_data = ["a b c d e", "a b c a b c", "a d e b", "a b"]
    bigram_extractor = BigramExtractor(min_frequency=1, score=method, verbose=False)
    bigrams = bigram_extractor.extract(train_data, threshold=threshold)
    assert "a - b" in bigrams
