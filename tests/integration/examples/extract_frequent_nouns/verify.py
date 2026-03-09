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

    # 멀티프로세싱 결과 검증 (n_workers=4 결과가 단일 프로세스와 95% 이상 겹치는지 확인)
    # 병렬 버전은 frozen LRGraph snapshot을 사용하므로 소수의 명사가 다를 수 있음
    corpus = CorpusLoader(DATA_PATH, format="jsonl", verbose=False)
    sents = [item["text"] for item in corpus]
    extractor_multi = LRNounExtractor(verbose=False)
    nouns_multi = extractor_multi.extract(sents, min_noun_frequency=10, n_workers=4)
    single_set = set(nouns.keys())
    multi_set = set(nouns_multi.keys())
    overlap = len(single_set & multi_set)
    total = len(single_set | multi_set)
    assert total == 0 or overlap / total >= 0.95, (
        f"멀티프로세싱 결과 겹침 부족: single={len(single_set)}, multi={len(multi_set)}, overlap={overlap / total:.3f}"
    )
