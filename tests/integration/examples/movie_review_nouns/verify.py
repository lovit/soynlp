"""Verify movie review noun extraction and domain comparison."""

import os

from soynlp.noun import LRNounExtractor
from soynlp.utils import CorpusLoader

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
NEWS_DATA = os.path.join(ROOT_DIR, "tests/integration/data/news-text/2016-10-20.jsonl")
REVIEW_DATA = os.path.join(ROOT_DIR, "tests/integration/data/movie-review-score/91031.jsonl")


def _read_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line for line in f.read().splitlines() if line.strip()]


def _read_answer(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def verify(answers_dir: str) -> None:
    # Extract news nouns
    news_loader = CorpusLoader(NEWS_DATA, format="jsonl", verbose=False)
    news_sents = [item["text"] for item in news_loader]
    news_extractor = LRNounExtractor(verbose=False)
    news_nouns = news_extractor.extract(news_sents, min_noun_frequency=10)

    # Extract review nouns
    review_loader = CorpusLoader(REVIEW_DATA, format="jsonl", verbose=False)
    review_sents = [item["text"] for item in review_loader]
    review_extractor = LRNounExtractor(verbose=False)
    review_nouns = review_extractor.extract(review_sents, min_noun_frequency=10)

    # Test review top nouns
    top_review = sorted(review_nouns.items(), key=lambda x: (-x[1].frequency, x[0]))[:100]
    actual_top = [f"{noun}\t{score.frequency}\t{score.score:.4f}" for noun, score in top_review]
    expected_top = _read_lines(f"{answers_dir}/review_top_nouns.txt")
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
    expected_comparison = _read_answer(f"{answers_dir}/domain_comparison.txt")
    assert actual_comparison == expected_comparison

    # 멀티프로세싱 결과 검증 (n_workers=4 결과가 단일 프로세스와 동일한지 확인)
    news_extractor_multi = LRNounExtractor(verbose=False)
    news_nouns_multi = news_extractor_multi.extract(news_sents, min_noun_frequency=10, n_workers=4)
    assert set(news_nouns.keys()) == set(news_nouns_multi.keys()), (
        f"뉴스 멀티프로세싱 결과 불일치: single={len(news_nouns)}, multi={len(news_nouns_multi)}"
    )

    review_extractor_multi = LRNounExtractor(verbose=False)
    review_nouns_multi = review_extractor_multi.extract(review_sents, min_noun_frequency=10, n_workers=4)
    assert set(review_nouns.keys()) == set(review_nouns_multi.keys()), (
        f"리뷰 멀티프로세싱 결과 불일치: single={len(review_nouns)}, multi={len(review_nouns_multi)}"
    )
