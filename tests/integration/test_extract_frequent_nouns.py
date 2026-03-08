"""Integration test: Frequent noun extraction."""

from .conftest import read_answer_lines


def test_top_nouns(news_nouns):
    top_nouns = sorted(news_nouns.items(), key=lambda x: -x[1].frequency)[:100]
    lines = []
    for noun, score in top_nouns:
        lines.append(f"{noun}\t{score.frequency}\t{score.score:.4f}")

    expected = read_answer_lines("extract_frequent_nouns", "top_nouns.txt")
    assert lines == expected
