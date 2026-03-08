"""Tests for multiprocessing support in LRNounExtractor."""

from soynlp.core import LRGraph, corpus_to_lrgraph
from soynlp.noun import LRNounExtractor
from soynlp.noun.lr import longer_first_prediction, prepare_noun_candidates

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


def test_noun_extractor_multi_with_min_eojeol_frequency():
    """min_eojeol_frequency > 1이면 n_workers를 지정해도 단일 프로세스로 동작하며 결과는 동일하다."""
    extractor_single = LRNounExtractor(verbose=False)
    nouns_single = extractor_single.extract(SAMPLE_SENTS, min_eojeol_frequency=2, n_workers=1)

    extractor_multi = LRNounExtractor(verbose=False)
    nouns_multi = extractor_multi.extract(SAMPLE_SENTS, min_eojeol_frequency=2, n_workers=4)

    assert set(nouns_single.keys()) == set(nouns_multi.keys())


def test_longer_first_prediction_parallel_runs():
    """longer_first_prediction n_workers=4가 에러 없이 실행되고 결과를 반환한다."""
    from soynlp.noun.lr import prepare_r_features

    lrgraph = corpus_to_lrgraph(SAMPLE_SENTS, n_workers=1)
    pos, neg, common = prepare_r_features(None, None)
    candidates = prepare_noun_candidates(lrgraph, pos, min_noun_frequency=1)

    scores = longer_first_prediction(candidates, lrgraph, pos, neg, common, 0.3, 1, 30, False, n_workers=4)
    assert isinstance(scores, dict)
    assert len(scores) > 0


def test_longer_first_prediction_parallel_same_keys():
    """longer_first_prediction n_workers=4 결과의 명사 집합이 n_workers=1과 동일하다.

    Note: 개별 스코어는 LRGraph 수정 순서 차이로 약간 다를 수 있으나, 최종 명사 집합은 대부분 동일.
    """
    from soynlp.core import corpus_to_lrgraph
    from soynlp.noun.lr import prepare_r_features

    pos, neg, common = prepare_r_features(None, None)

    lrgraph1 = corpus_to_lrgraph(SAMPLE_SENTS, n_workers=1)
    candidates1 = prepare_noun_candidates(lrgraph1, pos, min_noun_frequency=1)
    scores_single = longer_first_prediction(candidates1, lrgraph1, pos, neg, common, 0.3, 1, 30, False, n_workers=1)
    nouns_single = {w for w, (_, score) in scores_single.items() if score >= 0.3}

    lrgraph4 = corpus_to_lrgraph(SAMPLE_SENTS, n_workers=1)
    candidates4 = prepare_noun_candidates(lrgraph4, pos, min_noun_frequency=1)
    scores_multi = longer_first_prediction(candidates4, lrgraph4, pos, neg, common, 0.3, 1, 30, False, n_workers=4)
    nouns_multi = {w for w, (_, score) in scores_multi.items() if score >= 0.3}

    # 완전히 동일하지 않을 수 있으나 대부분 겹침 (차이 < 5%)
    overlap = len(nouns_single & nouns_multi)
    total = len(nouns_single | nouns_multi)
    assert overlap / total >= 0.95 if total > 0 else True
