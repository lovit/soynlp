"""Integration test: Movie review noun extraction and domain comparison."""

from soynlp.noun import LRNounExtractor
from soynlp.utils import CorpusLoader

from .conftest import REVIEW_DATA, read_answer, read_answer_lines


def test_movie_review_nouns(news_nouns):
    loader = CorpusLoader(REVIEW_DATA, format="jsonl", verbose=False)
    review_sents = [item["text"] for item in loader]
    review_extractor = LRNounExtractor(verbose=False)
    review_nouns = review_extractor.extract(review_sents, min_noun_frequency=10)

    # Test review top nouns
    top_review = sorted(review_nouns.items(), key=lambda x: (-x[1].frequency, x[0]))[:100]
    actual_top = [f"{noun}\t{score.frequency}\t{score.score:.4f}" for noun, score in top_review]
    expected_top = read_answer_lines("movie_review_nouns", "review_top_nouns.txt")
    assert actual_top == expected_top

    # Test domain comparison
    review_set = set(review_nouns.keys())
    news_set = set(news_nouns.keys())
    common = review_set & news_set
    review_only = review_set - news_set
    news_only = news_set - review_set

    lines = [
        f"Review nouns: {len(review_set)}",
        f"News nouns: {len(news_set)}",
        f"Common: {len(common)}",
        f"Review-only: {len(review_only)}",
        f"News-only: {len(news_only)}",
        "",
        "Top review-only nouns (domain-specific):",
    ]
    review_only_sorted = sorted(
        [(n, review_nouns[n]) for n in review_only],
        key=lambda x: (-x[1].frequency, x[0]),
    )[:30]
    for noun, score in review_only_sorted:
        lines.append(f"  {noun}\t{score.frequency}")
    actual_comparison = "\n".join(lines) + "\n"
    expected_comparison = read_answer("movie_review_nouns", "domain_comparison.txt")
    assert actual_comparison == expected_comparison
