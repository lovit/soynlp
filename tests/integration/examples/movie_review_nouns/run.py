"""Extract nouns from movie review corpus and compare with news domain.

Tests LRNounExtractor on a different domain (movie reviews with scores).

Usage:
    uv run python tests/integration/examples/movie_review_nouns/run.py
"""

import os

from soynlp.noun import LRNounExtractor
from soynlp.utils import CorpusLoader

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
REVIEW_PATH = os.path.join(ROOT_DIR, "tests/integration/data/movie-review-score/91031.jsonl")
NEWS_PATH = os.path.join(ROOT_DIR, "tests/integration/data/news-text/2016-10-20.jsonl")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def extract_nouns(data_path, label):
    """Extract nouns from a corpus."""
    loader = CorpusLoader(data_path, format="jsonl", verbose=True)
    sents = [item["text"] for item in loader]
    print(f"[{label}] Loaded {len(sents)} documents")

    extractor = LRNounExtractor(verbose=True)
    nouns = extractor.extract(sents, min_noun_frequency=10)
    print(f"[{label}] Extracted {len(nouns)} nouns")
    return nouns


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Extract nouns from movie reviews
    print("=== Movie Review Nouns ===")
    review_nouns = extract_nouns(REVIEW_PATH, "Review")

    # Extract nouns from news (for comparison)
    print("\n=== News Nouns ===")
    news_nouns = extract_nouns(NEWS_PATH, "News")

    # Save review nouns
    top_review = sorted(review_nouns.items(), key=lambda x: -x[1].frequency)[:100]
    review_path = os.path.join(OUTPUT_DIR, "review_top_nouns.txt")
    with open(review_path, "w", encoding="utf-8") as f:
        for noun, score in top_review:
            f.write(f"{noun}\t{score.frequency}\t{score.score:.4f}\n")
    print(f"\nReview top-100 nouns saved to {review_path}")

    # Domain comparison
    review_set = set(review_nouns.keys())
    news_set = set(news_nouns.keys())
    common = review_set & news_set
    review_only = review_set - news_set
    news_only = news_set - review_set

    comparison_path = os.path.join(OUTPUT_DIR, "domain_comparison.txt")
    with open(comparison_path, "w", encoding="utf-8") as f:
        f.write(f"Review nouns: {len(review_set)}\n")
        f.write(f"News nouns: {len(news_set)}\n")
        f.write(f"Common: {len(common)}\n")
        f.write(f"Review-only: {len(review_only)}\n")
        f.write(f"News-only: {len(news_only)}\n\n")

        # Top review-only nouns (domain-specific)
        review_only_sorted = sorted(
            [(n, review_nouns[n]) for n in review_only],
            key=lambda x: -x[1].frequency,
        )[:30]
        f.write("Top review-only nouns (domain-specific):\n")
        for noun, score in review_only_sorted:
            f.write(f"  {noun}\t{score.frequency}\n")

    print(f"Domain comparison saved to {comparison_path}")
    print(f"\nCommon nouns: {len(common)}")
    print(f"Review-only nouns: {len(review_only)}")
    print(f"News-only nouns: {len(news_only)}")

    print("\nTop-10 review-specific nouns:")
    review_only_sorted = sorted(
        [(n, review_nouns[n]) for n in review_only],
        key=lambda x: -x[1].frequency,
    )[:10]
    for noun, score in review_only_sorted:
        print(f"  {noun}: frequency={score.frequency}")


if __name__ == "__main__":
    main()
