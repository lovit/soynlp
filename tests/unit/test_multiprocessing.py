"""Tests for multiprocessing support in LRNounExtractor."""

from soynlp.core import LRGraph, corpus_to_lrgraph
from soynlp.noun import LRNounExtractor

# 테스트용 샘플 문장 (충분한 양)
SAMPLE_SENTS = [
    "아이오아이가 평가단에게 높은 점수를 받았습니다",
    "아이오아이는 아이돌 그룹입니다",
    "자연어처리는 어렵습니다",
    "자연어처리를 공부합니다",
    "명사추출은 중요한 작업입니다",
] * 100  # 충분한 반복


def test_corpus_to_lrgraph_single():
    """단일 프로세스로 LRGraph를 구축한다."""
    lrgraph = corpus_to_lrgraph(SAMPLE_SENTS, n_workers=1)
    assert isinstance(lrgraph, LRGraph)
    assert len(lrgraph._lr) > 0


def test_corpus_to_lrgraph_multi():
    """멀티프로세스로 구축한 LRGraph가 단일 프로세스 결과와 동일하다."""
    lrgraph_single = corpus_to_lrgraph(SAMPLE_SENTS, n_workers=1)
    lrgraph_multi = corpus_to_lrgraph(SAMPLE_SENTS, n_workers=4)
    assert lrgraph_single._lr == lrgraph_multi._lr


def test_noun_extractor_single():
    """단일 프로세스로 명사를 추출한다."""
    extractor = LRNounExtractor(verbose=False)
    nouns = extractor.extract(SAMPLE_SENTS, n_workers=1)
    assert len(nouns) > 0


def test_noun_extractor_multi():
    """n_workers=4로 추출한 명사 결과가 단일 프로세스와 동일하다."""
    extractor_single = LRNounExtractor(verbose=False)
    nouns_single = extractor_single.extract(SAMPLE_SENTS, n_workers=1)

    extractor_multi = LRNounExtractor(verbose=False)
    nouns_multi = extractor_multi.extract(SAMPLE_SENTS, n_workers=4)

    assert set(nouns_single.keys()) == set(nouns_multi.keys())


def test_noun_extractor_auto_workers():
    """n_workers=-1이면 CPU 코어 수를 자동으로 사용한다."""
    extractor = LRNounExtractor(verbose=False)
    nouns = extractor.extract(SAMPLE_SENTS, n_workers=-1)
    assert len(nouns) > 0
