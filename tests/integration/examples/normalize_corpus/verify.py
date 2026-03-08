"""Verify corpus normalization results."""

import json
import os

from soynlp.normalizer import TextNormalizer, only_hangle, repeat_normalize
from soynlp.utils import CorpusLoader

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
NEWS_DATA = os.path.join(ROOT_DIR, "tests/integration/data/news-text/2016-10-20.jsonl")
REVIEW_DATA = os.path.join(ROOT_DIR, "tests/integration/data/movie-review-score/91031.jsonl")


def _read_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line for line in f.read().splitlines() if line.strip()]


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


def verify(answers_dir: str) -> None:
    # News normalization
    expected_news = _read_lines(f"{answers_dir}/news_normalized.jsonl")
    actual_news = _normalize_news()
    assert len(actual_news) == len(expected_news)
    for result, expected_line in zip(actual_news, expected_news):
        expected_obj = json.loads(expected_line)
        assert result == expected_obj

    # Review normalization
    expected_reviews = _read_lines(f"{answers_dir}/reviews_normalized.jsonl")
    actual_reviews = _normalize_reviews()
    assert len(actual_reviews) == len(expected_reviews)
    for result, expected_line in zip(actual_reviews, expected_reviews):
        expected_obj = json.loads(expected_line)
        assert result == expected_obj
