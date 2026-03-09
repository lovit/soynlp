import os

from soynlp.core.lrgraph import LRGraph

filepath = os.path.dirname(os.path.realpath(__file__))
josapath = filepath + "/frequent_enrolled_josa.txt"
suffixpath = filepath + "/frequent_noun_suffix.txt"


def load_lines_as_set(path: str) -> set[str]:
    with open(path, encoding="utf-8") as f:
        return {word.strip() for word in f if word.strip()}


josaset = load_lines_as_set(josapath)
suffixset = load_lines_as_set(suffixpath)

# 파생 명사 접미사: 생산성 높음/중간
# 주의: "가"는 josaset에도 포함된 조사이므로 제외 (check_N_is_NJ 후처리와 충돌 가능)
_HIGH_MEDIUM_SUFFIXES: tuple[str, ...] = (
    "화",
    "성",
    "적",
    "자",
    "들",
    "상",
    "기",
    "학",
    "론",
    "계",
    "형",
    "주의",
    "권",
    "력",
    "감",
    "관",
    "제",
)
# 파생 명사 접미사: 생산성 낮음 (Noun 길이 >= 2 조건 적용)
_LOW_SUFFIXES: tuple[str, ...] = ("꾼", "쟁이", "질")


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


def expand_suffix_nouns(
    nouns: dict[str, tuple[int, float]],
    lrgraph: LRGraph,
    min_noun_frequency: int = 1,
) -> dict[str, tuple[int, float]]:
    """추출된 명사에 파생 접미사를 붙인 형태가 코퍼스에 존재할 경우 명사로 추가한다.

    - 생산성 높음/중간 접미사: 이미 추출된 명사가 아닌 경우에만 추가
    - 생산성 낮음 접미사 (꾼, 쟁이, 질): Noun 길이 >= 2 조건 추가 적용
    - 추가된 명사의 score 는 1.0 으로 설정
    """
    added: dict[str, tuple[int, float]] = {}
    for noun in list(nouns.keys()):
        for suffix in _HIGH_MEDIUM_SUFFIXES:
            candidate = noun + suffix
            if candidate in nouns or candidate in added:
                continue
            freq = sum(lrgraph._lr_origin.get(candidate, {}).values())
            if freq >= min_noun_frequency:
                added[candidate] = (freq, 1.0)
        if len(noun) >= 2:
            for suffix in _LOW_SUFFIXES:
                candidate = noun + suffix
                if candidate in nouns or candidate in added:
                    continue
                freq = sum(lrgraph._lr_origin.get(candidate, {}).values())
                if freq >= min_noun_frequency:
                    added[candidate] = (freq, 1.0)
    return {**nouns, **added}


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
