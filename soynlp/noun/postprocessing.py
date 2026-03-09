import os

from soynlp.core.lrgraph import LRGraph

filepath = os.path.dirname(os.path.realpath(__file__))
josapath = os.path.join(filepath, "frequent_enrolled_josa.txt")
suffixpath = os.path.join(filepath, "frequent_noun_suffix.txt")


def load_lines_as_set(path: str) -> set[str]:
    with open(path, encoding="utf-8") as f:
        return {word.strip() for word in f if word.strip()}


josaset = load_lines_as_set(josapath)
suffixset = load_lines_as_set(suffixpath)


def subtract(base: dict[str, tuple[int, float]], removals: set[str]) -> dict[str, tuple[int, float]]:
    return {word: score for word, score in base.items() if (word not in removals)}


def detaching_features(
    nouns: dict[str, tuple[int, float]], features: set[str]
) -> tuple[dict[str, tuple[int, float]], set[str]]:
    removals: set[str] = set()
    for word in nouns:
        if len(word) <= 2:
            continue
        for e in range(2, len(word)):
            l, r = word[:e], word[e:]  # noqa: E741
            # Skip a syllable word such as 고양이, 이력서
            if len(r) <= 1:
                continue
            if (l in nouns) and (r in features):
                removals.add(word)
                break
    nouns = subtract(nouns, removals)
    return nouns, removals


def ignore_features(nouns: dict[str, tuple[int, float]], features: set[str]) -> tuple[dict[str, tuple[int, float]], set[str]]:
    removals: set[str] = set()
    for word in nouns:
        if word in features:
            removals.add(word)
    nouns = subtract(nouns, removals)
    return nouns, removals


def check_N_is_NJ(
    nouns: dict[str, tuple[int, float]], lrgraph: LRGraph, min_num_of_josa: int = 5
) -> tuple[dict[str, tuple[int, float]], set[str]]:
    removals: set[str] = set()
    for word, score in nouns.items():
        n = len(word)
        if n <= 2:
            continue
        for i in range(2, n):
            l, r = word[:i], word[i:]  # noqa: E741
            if (
                r not in josaset  # R 이 조사가 아니거나
                or (r in suffixset)  # -서, -장, -이 처럼 suffix 이거나
                or l not in nouns  # L 이 명사가 아니거나
                or score[0] >= nouns[l][0]  # L 의 명사 빈도수가 더 작으면
            ):
                continue
            features = lrgraph._lr_origin.get(l, {})
            features = [r for r in features if r in josaset]
            n_josa = len(features)
            if n_josa >= min_num_of_josa:
                removals.add(word)
    nouns = subtract(nouns, removals)
    return nouns, removals
