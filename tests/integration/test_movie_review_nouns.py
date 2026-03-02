"""Integration test: Movie review noun extraction and domain comparison."""

from soynlp.noun import LRNounExtractor
from soynlp.utils import CorpusLoader

from .conftest import NEWS_DATA, REVIEW_DATA, read_answer, read_answer_lines


def _extract_nouns(data_path: str) -> dict:
    loader = CorpusLoader(data_path, format="jsonl", verbose=False)
    sents = [item["text"] for item in loader]
    extractor = LRNounExtractor(verbose=False)
    return extractor.extract(sents, min_noun_frequency=10)


def _build_review_top_nouns(review_nouns: dict) -> list[str]:
    top_review = sorted(review_nouns.items(), key=lambda x: (-x[1].frequency, x[0]))[:100]
    return [f"{noun}\t{score.frequency}\t{score.score:.4f}" for noun, score in top_review]


def _build_domain_comparison(review_nouns: dict, news_nouns: dict) -> str:
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
    return "\n".join(lines) + "\n"


def test_movie_review_nouns():
    review_nouns = _extract_nouns(REVIEW_DATA)
    news_nouns = _extract_nouns(NEWS_DATA)

    # Test review top nouns
    actual_top = _build_review_top_nouns(review_nouns)
    expected_top = read_answer_lines("movie_review_nouns", "review_top_nouns.txt")
    assert actual_top == expected_top

    # Test domain comparison
    actual_comparison = _build_domain_comparison(review_nouns, news_nouns)
    expected_comparison = read_answer("movie_review_nouns", "domain_comparison.txt")
    assert actual_comparison == expected_comparison
