"""Verify predicator (verb/adjective) extraction results."""

import os

from soynlp.noun import LRNounExtractor
from soynlp.predicator import PredicatorExtractor
from soynlp.utils import CorpusLoader

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
NEWS_DATA = os.path.join(ROOT_DIR, "tests/integration/data/news-text/2016-10-20.jsonl")


def _read_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line for line in f.read().splitlines() if line.strip()]


def verify(answers_dir: str) -> None:
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

    expected_adj = _read_lines(f"{answers_dir}/top_adjectives.txt")
    expected_verbs = _read_lines(f"{answers_dir}/top_verbs.txt")
    assert adj_lines == expected_adj
    assert verb_lines == expected_verbs
