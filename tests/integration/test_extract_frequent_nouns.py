"""Integration test: Frequent noun extraction."""

from soynlp.noun import LRNounExtractor
from soynlp.utils import CorpusLoader

from .conftest import NEWS_DATA, read_answer_lines


def _extract_top_nouns() -> list[str]:
    corpus = CorpusLoader(NEWS_DATA, format="jsonl", verbose=False)
    noun_extractor = LRNounExtractor(verbose=False)
    nouns = noun_extractor.extract(corpus, min_noun_frequency=10)
    top_nouns = sorted(nouns.items(), key=lambda x: -x[1].frequency)[:100]
    lines = []
    for noun, score in top_nouns:
        lines.append(f"{noun}\t{score.frequency}\t{score.score:.4f}")
    return lines


def test_top_nouns():
    expected = read_answer_lines("extract_frequent_nouns", "top_nouns.txt")
    actual = _extract_top_nouns()
    assert actual == expected
