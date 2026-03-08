"""Verify frequent noun extraction results."""

import os

from soynlp.noun import LRNounExtractor
from soynlp.utils import CorpusLoader

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
DATA_PATH = os.path.join(ROOT_DIR, "tests/integration/data/news-text/2016-10-20.jsonl")


def _read_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line for line in f.read().splitlines() if line.strip()]


def verify(parameters: dict, answers_dir: str) -> None:
    # 단일 프로세스 결과 검증 (pipeline 실행 결과)
    nouns = parameters["nouns"]
    top_nouns = sorted(nouns.items(), key=lambda x: -x[1].frequency)[:100]
    lines = [f"{noun}\t{score.frequency}\t{score.score:.4f}" for noun, score in top_nouns]
    expected = _read_lines(f"{answers_dir}/top_nouns.txt")
    assert lines == expected

    # 멀티프로세싱 결과 검증 (n_workers=4 결과가 단일 프로세스와 동일한지 확인)
    corpus = CorpusLoader(DATA_PATH, format="jsonl", verbose=False)
    sents = [item["text"] for item in corpus]
    extractor_multi = LRNounExtractor(verbose=False)
    nouns_multi = extractor_multi.extract(sents, min_noun_frequency=10, n_workers=4)
    assert set(nouns.keys()) == set(nouns_multi.keys()), (
        f"멀티프로세싱 결과 불일치: single={len(nouns)}, multi={len(nouns_multi)}"
    )
