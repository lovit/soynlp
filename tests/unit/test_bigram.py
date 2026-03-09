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


_BIGRAM_SENTS = [
    "아이오아이가 평가단에게 높은 점수를 받았습니다",
    "아이오아이는 아이돌 그룹입니다",
    "자연어처리는 어렵고 재미있는 분야입니다",
    "자연어처리를 열심히 공부합니다",
] * 200


def test_bigram_extractor_multi_equals_single():
    """n_workers=4 결과의 unigram/bigram 카운터가 단일 프로세스와 동일하다."""
    ext_single = BigramExtractor(min_frequency=1, verbose=False)
    ext_single.extract(_BIGRAM_SENTS, n_workers=1)

    ext_multi = BigramExtractor(min_frequency=1, verbose=False)
    ext_multi.extract(_BIGRAM_SENTS, n_workers=4)

    assert ext_single.unigrams == ext_multi.unigrams
    assert ext_single.bigrams == ext_multi.bigrams


def test_bigram_extractor_auto_workers():
    """n_workers=-1이면 CPU 코어 수를 자동 사용한다."""
    ext = BigramExtractor(min_frequency=1, verbose=False)
    result = ext.extract(_BIGRAM_SENTS, n_workers=-1)
    assert len(result) > 0
