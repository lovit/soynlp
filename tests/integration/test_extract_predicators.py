"""Integration test: Predicator (verb/adjective) extraction."""

from soynlp.noun import LRNounExtractor
from soynlp.predicator import PredicatorExtractor
from soynlp.utils import CorpusLoader

from .conftest import NEWS_DATA, read_answer_lines


def _extract_predicators() -> tuple[list[str], list[str]]:
    loader = CorpusLoader(NEWS_DATA, format="jsonl", verbose=False)
    sents = [item["text"] for item in loader]

    noun_extractor = LRNounExtractor(verbose=False)
    nouns = noun_extractor.extract(sents, min_noun_frequency=10)
    noun_set = set(nouns.keys())

    predicator_extractor = PredicatorExtractor(nouns=noun_set, verbose=False)
    adjectives, verbs = predicator_extractor.train_extract(
        sents,
        min_eojeol_frequency=2,
        min_predicator_frequency=5,
    )

    adj_lines = []
    for word, pred in sorted(adjectives.items(), key=lambda x: (-x[1].frequency, x[0]))[:50]:
        lemmas = ", ".join(f"{s}+{e}" for s, e in sorted(pred.lemma)[:3])
        adj_lines.append(f"{word}\t{pred.frequency}\t{lemmas}")

    verb_lines = []
    for word, pred in sorted(verbs.items(), key=lambda x: (-x[1].frequency, x[0]))[:50]:
        lemmas = ", ".join(f"{s}+{e}" for s, e in sorted(pred.lemma)[:3])
        verb_lines.append(f"{word}\t{pred.frequency}\t{lemmas}")

    return adj_lines, verb_lines


def test_predicator_extraction():
    adj_lines, verb_lines = _extract_predicators()
    expected_adj = read_answer_lines("extract_predicators", "top_adjectives.txt")
    expected_verbs = read_answer_lines("extract_predicators", "top_verbs.txt")
    assert adj_lines == expected_adj
    assert verb_lines == expected_verbs
