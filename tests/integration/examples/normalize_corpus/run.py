"""Normalize news and movie review corpus using TextNormalizer.

Tests both function-based and class-based normalizer APIs on real data.

Usage:
    uv run python tests/integration/examples/normalize_corpus/run.py
"""

import json
import os

from soynlp.normalizer import (
    TextNormalizer,
    only_hangle,  # type: ignore[attr-defined]
    repeat_normalize,  # type: ignore[attr-defined]
    text_normalizer,
)
from soynlp.utils import CorpusLoader

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
NEWS_PATH = os.path.join(ROOT_DIR, "tests/integration/data/news-text/2016-10-20.jsonl")
REVIEW_PATH = os.path.join(ROOT_DIR, "tests/integration/data/movie-review-score/91031.jsonl")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def normalize_news():
    """Normalize news corpus with class-based API."""
    print("=== News corpus normalization ===")
    loader = CorpusLoader(NEWS_PATH, format="jsonl")

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

    output_path = os.path.join(OUTPUT_DIR, "news_normalized.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  Normalized {len(results)} news documents → {output_path}")
    print(f"  Sample: {results[0]['normalized'][:100]}...")
    return results


def normalize_reviews():
    """Normalize movie reviews with function-based API."""
    print("\n=== Movie review normalization ===")
    loader = CorpusLoader(REVIEW_PATH, format="jsonl")

    results = []
    for i, item in enumerate(loader):
        if i >= 100:
            break
        text = item["text"]
        # Apply multiple normalization functions
        normed = repeat_normalize(text, num_repeats=2)
        normed_hangle = only_hangle(text)
        results.append(
            {
                "original": text[:200],
                "repeat_normalized": normed[:200],
                "hangle_only": normed_hangle[:200],
            }
        )

    output_path = os.path.join(OUTPUT_DIR, "reviews_normalized.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  Normalized {len(results)} reviews → {output_path}")
    print(f"  Sample original:   {results[0]['original'][:80]}...")
    print(f"  Sample normalized: {results[0]['repeat_normalized'][:80]}...")
    print(f"  Sample hangle:     {results[0]['hangle_only'][:80]}...")
    return results


def test_default_normalizer():
    """Test the pre-built default normalizer."""
    print("\n=== Default text_normalizer test ===")
    test_sentences = [
        "아아아아아 진짜좋다ㅋㅋㅋㅋㅋㅋㅋㅋ",
        "너무너무너무 좋아요!!!!!!!",
        "ㅎㅎㅎㅎㅎ 재밌었다 ㅋㅋㅋㅋㅋ",
        "이   영화   정말    좋아요",
    ]
    for sent in test_sentences:
        normed = text_normalizer(sent)
        print(f"  '{sent}' → '{normed}'")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    normalize_news()
    normalize_reviews()
    test_default_normalizer()
    print("\nDone.")


if __name__ == "__main__":
    main()
