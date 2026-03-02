"""Integration test: Corpus normalization."""

import json

from soynlp.normalizer import TextNormalizer, only_hangle, repeat_normalize
from soynlp.utils import CorpusLoader

from .conftest import NEWS_DATA, REVIEW_DATA, read_answer_lines


def _normalize_news() -> list[dict[str, str]]:
    loader = CorpusLoader(NEWS_DATA, format="jsonl")
    normalizer = TextNormalizer.build_normalizer(
        alphabet=True,
        hangle=True,
        number=True,
        remove_repeatchar=2,
        remove_longspace=True,
    )
    results = []
    for i, item in enumerate(loader):
        if i >= 100:
            break
        text = item["text"]
        normalized = normalizer(text)
        results.append({"original": text[:200], "normalized": normalized[:200]})
    return results


def _normalize_reviews() -> list[dict[str, str]]:
    loader = CorpusLoader(REVIEW_DATA, format="jsonl")
    results = []
    for i, item in enumerate(loader):
        if i >= 100:
            break
        text = item["text"]
        normed = repeat_normalize(text, num_repeats=2)
        normed_hangle = only_hangle(text)
        results.append(
            {
                "original": text[:200],
                "repeat_normalized": normed[:200],
                "hangle_only": normed_hangle[:200],
            }
        )
    return results


def test_news_normalization():
    expected_lines = read_answer_lines("normalize_corpus", "news_normalized.jsonl")
    actual = _normalize_news()
    assert len(actual) == len(expected_lines)
    for result, expected_line in zip(actual, expected_lines):
        expected_obj = json.loads(expected_line)
        assert result == expected_obj


def test_review_normalization():
    expected_lines = read_answer_lines("normalize_corpus", "reviews_normalized.jsonl")
    actual = _normalize_reviews()
    assert len(actual) == len(expected_lines)
    for result, expected_line in zip(actual, expected_lines):
        expected_obj = json.loads(expected_line)
        assert result == expected_obj
