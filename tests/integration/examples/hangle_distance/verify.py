"""Verify hangle distance calculations and similar word search."""

import os

from soynlp.hangle import (
    character_is_complete_korean,
    cosine_distance,
    jaccard_distance,
    jamo_levenshtein,
    levenshtein,
)
from soynlp.utils import CorpusLoader

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
NEWS_DATA = os.path.join(ROOT_DIR, "tests/integration/data/news-text/2016-10-20.jsonl")


def _read_answer(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def verify(answers_dir: str) -> None:
    # Test deterministic distance calculations
    expected = _read_answer(f"{answers_dir}/distance_results.txt")
    expected_table = expected.split("\n# Similar words")[0].strip()

    word_pairs = [
        ("대통령", "대통렬"),
        ("삼성전자", "삼성전기"),
        ("경찰", "경철"),
        ("연합뉴스", "연합뉴우스"),
        ("서울", "서울시"),
        ("대한민국", "대한민극"),
        ("아이오아이", "아이오에이"),
        ("국회의원", "국회위원"),
    ]
    lines = ["# Distance Calculations"]
    lines.append(f"{'word1':<15} {'word2':<15} {'levenshtein':>12} {'jamo_lev':>10} {'cosine':>10} {'jaccard':>10}")
    for w1, w2 in word_pairs:
        lev = levenshtein(w1, w2)
        jlev = jamo_levenshtein(w1, w2)
        cos = cosine_distance(w1, w2)
        jac = jaccard_distance(w1, w2)
        lines.append(f"{w1:<15} {w2:<15} {lev:>12.2f} {jlev:>10.2f} {cos:>10.4f} {jac:>10.4f}")

    actual_table = "\n".join(lines)
    assert actual_table == expected_table

    # Test similar word search structure (non-deterministic ordering)
    loader = CorpusLoader(NEWS_DATA, format="jsonl")
    words: set[str] = set()
    for i, item in enumerate(loader):
        if i >= 1000:
            break
        for eojeol in item["text"].split():
            if 2 <= len(eojeol) <= 5 and all(character_is_complete_korean(c) for c in eojeol):
                words.add(eojeol)

    queries = ["대통령", "경찰", "서울"]
    for query in queries:
        assert query in words
        distances = [(w, jamo_levenshtein(query, w)) for w in words if w != query]
        distances.sort(key=lambda x: x[1])
        top5 = distances[:5]
        assert len(top5) == 5
        for i in range(len(top5) - 1):
            assert top5[i][1] <= top5[i + 1][1]
