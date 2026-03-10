import logging

from soynlp.hangle import compose
from soynlp.hangle import decompose as _raw_decompose

logger = logging.getLogger(__name__)

positive_moum = set("ㅏㅑㅗㅛ")
negative_moum = set("ㅓㅕㅜㅠ")
neuter_moum = set("ㅡㅣ")
pos_to_neg: dict[str, str] = {"ㅏ": "ㅓ", "ㅑ": "ㅕ", "ㅗ": "ㅜ", "ㅛ": "ㅠ"}
neg_to_pos: dict[str, str] = {"ㅓ": "ㅏ", "ㅕ": "ㅑ", "ㅜ": "ㅗ", "ㅠ": "ㅛ"}


def decompose(c: str) -> tuple[str, str, str]:
    """Decompose a Korean character, raising ValueError if not Korean."""
    result = _raw_decompose(c)
    if result is None:
        raise ValueError(f"Cannot decompose non-Korean character: {c!r}")
    return result


def conjugate_chat(stem: str, ending: str, enforce_moum_harmoney: bool = False, debug: bool = False) -> set[str]:
    if not ending:
        return {stem}

    candidates = conjugate(stem, ending, enforce_moum_harmoney, debug)

    l_last = list(decompose(stem[-1]))
    r_first = list(decompose(ending[0]))

    # 어미의 첫글자가 종성일 경우 (-ㄴ, -ㄹ, -ㅂ, -ㅆ)
    # 이 + ㅂ니다 -> 입니다
    if r_first[1] == " " and r_first[0] != " ":
        l = stem[:-1] + compose(l_last[0], l_last[1], r_first[0])
        r = ending[1:]
        surface = l + r
        candidates.add(surface)
        if r_first[1] != " ":
            candidates.add(stem + ending)
        logger.debug("어미의 첫 글자가 자음인 경우: %s", surface)

    return candidates


def conjugate(stem: str, ending: str, enforce_moum_harmoney: bool = False, debug: bool = False) -> set[str]:
    if not ending:
        raise ValueError("ending must be provided")

    l_len = len(stem)
    l_last = list(decompose(stem[-1]))
    l_last_ = stem[-1]
    r_first = list(decompose(ending[0]))

    if enforce_moum_harmoney:
        if (l_last[2] != "ㅂ" and l_last[1] in positive_moum) and (r_first[0] == "ㅇ" and r_first[1] in negative_moum):
            r_first[1] = neg_to_pos[r_first[1]]
            ending = compose(*r_first) + ending[1:]
        if (l_last[2] != "ㅂ" and l_last[1] in negative_moum) and (r_first[0] == "ㅇ" and r_first[1] in positive_moum):
            r_first[1] = pos_to_neg[r_first[1]]
            ending = compose(*r_first) + ending[1:]
        if (l_last[1] in neuter_moum) and (r_first[1] in positive_moum):
            r_first[1] = pos_to_neg[r_first[1]]
            ending = compose(*r_first) + ending[1:]

    r_first_ = compose(r_first[0], r_first[1], " ") if r_first[1] != " " else ending[0]

    candidates: set[str] = set()

    logger.debug("l_last = %s", l_last)
    logger.debug("r_first = %s", r_first)

    if ending[0] == "다":
        surface = stem + ending
        candidates.add(surface)
        logger.debug("'다'로 시작하는 어미: %s", surface)

    # ㄷ 불규칙 활용: 깨달 + 아 -> 깨달아
    if l_last[2] == "ㄷ" and r_first[0] == "ㅇ":
        l = stem[:-1] + compose(l_last[0], l_last[1], "ㄹ")
        surface = l + ending
        candidates.add(surface)
        candidates.add(stem + ending)  # 받 + 았다 -> 받았다
        logger.debug("ㄷ 불규칙: %s", surface)

    # 르 불규칙 활용: 구르 + 어 -> 굴러
    if (l_last_ == "르" and stem[-2:] != "푸르") and (r_first_ == "아" or r_first_ == "어") and l_len >= 2:
        c0, c1, c2 = decompose(stem[-2])
        l = stem[:-2] + compose(c0, c1, "ㄹ")
        r = compose("ㄹ", r_first[1], r_first[2]) + ending[1:]
        surface = l + r
        candidates.add(surface)
        logger.debug("르 불규칙: %s", surface)

    # ㅂ 불규칙 활용
    if l_last[2] == "ㅂ":
        l = stem[:-1] + compose(l_last[0], l_last[1], " ")
        if r_first_ == "어" or r_first_ == "아":
            if l_len >= 2 and (l_last_ == "답" or l_last_ == "곱" or l_last_ == "깝" or l_last_ == "롭"):
                c1 = "ㅝ"
            elif r_first[1] == "ㅗ":
                c1 = "ㅘ"
            elif r_first[1] == "ㅜ":
                c1 = "ㅝ"
            elif r_first_ == "어":
                c1 = "ㅝ"
            else:
                c1 = "ㅘ"
            r = compose("ㅇ", c1, r_first[2]) + ending[1:]
            surface = l + r
            candidates.add(surface)
            logger.debug("ㅂ 불규칙: %s", surface)
            # 워/와 생략 대화체 추가 (간지러워 → 간지러)
            if not ending[1:]:
                candidates.add(l)
                logger.debug("ㅂ 불규칙 워/와 생략: %s", l)
        elif r_first[0] == "ㅇ":
            surface = l + ending
            candidates.add(surface)
            logger.debug("ㅂ 불규칙: %s", surface)

    # 어미의 첫글자가 종성일 경우
    if r_first[1] == " " and r_first[0] in ("ㄴ", "ㄹ", "ㅁ", "ㅂ", "ㅆ"):
        l = stem[:-1] + compose(l_last[0], l_last[1], r_first[0])
        r = ending[1:]
        surface = l + r
        candidates.add(surface)
        if r_first[1] != " ":
            candidates.add(stem + ending)
        logger.debug("어미의 첫 글자가 -ㄴ, -ㄹ, -ㅁ-, -ㅂ, -ㅆ 인 경우: %s", surface)

    # ㅅ 불규칙 활용: 붓 + 어 -> 부어
    if (l_last[2] == "ㅅ") and (r_first[0] == "ㅇ"):
        if stem[-1] == "벗":
            l = stem
        else:
            l = stem[:-1] + compose(l_last[0], l_last[1], " ")
        surface = l + ending
        candidates.add(surface)
        logger.debug("ㅅ 불규칙: %s", surface)

    # 우 불규칙 활용: 푸 + 어 -> 퍼 / 주 + 어 -> 줘
    if l_last[1] == "ㅜ" and l_last[2] == " " and r_first[0] == "ㅇ" and r_first[1] == "ㅓ":
        if l_last_ == "푸":
            l = stem[:-1] + "퍼"
        else:
            l = stem[:-1] + compose(l_last[0], "ㅝ", r_first[2])
        r = ending[1:]
        surface = l + r
        candidates.add(surface)
        logger.debug("우 불규칙: %s", surface)

    # 오 활용: 오 + 았어 -> 왔어
    if l_last[1] == "ㅗ" and l_last[2] == " " and r_first[0] == "ㅇ" and r_first[1] == "ㅏ":
        l = stem[:-1] + compose(l_last[0], "ㅘ", r_first[2])
        r = ending[1:]
        surface = l + r
        candidates.add(surface)
        logger.debug("오 활용: %s", surface)

    # ㅡ 탈락 불규칙 활용
    if (l_last[1] == "ㅡ") and (l_last[2] == " ") and (r_first[0] == "ㅇ"):
        if l_last[0] == "ㅇ" and len(stem) > 1:
            surface = stem[:-1] + ending
        elif l_last[0] != "ㄹ":
            surface = stem[:-1] + compose(l_last[0], r_first[1], r_first[2]) + ending[1:]
        else:
            # 르 ㅡ탈락: 치르다, 따르다, 들르다, 다다르다, 우러르다 등
            # 르의 ㅡ가 탈락하고 ㄹ이 다음 음절 초성이 됨 (치르 + 어 → 치러)
            surface = stem[:-1] + compose("ㄹ", r_first[1], r_first[2]) + ending[1:]
        if surface is not None:
            candidates.add(surface)
        if surface is not None:
            logger.debug("ㅡ 탈락 불규칙: %s", surface)

    # 거라, 너라 불규칙 활용
    if ending[:2] == "어라" or ending[:2] == "아라":
        if stem[-1] == "오":
            l = stem[:-1]
            r = "와" + ending[1:]
        elif stem[-1] == "우":
            l = stem[:-1]
            r = "워" + ending[1:]
        elif stem[-1] == "가":
            l = stem
            r = ending[1:]
        else:
            if l_last[1] in negative_moum:
                l = stem
                r = "어" + ending[1:]
            else:
                l = stem
                r = "아" + ending[1:]
        surface = l + r
        candidates.add(surface)
        logger.debug("거라/너라 불규칙: %s", surface)

    # 러 불규칙 활용: 이르 + 어 -> 이르러
    if (l_last_ == "르" and stem[-2:] != "구르") and (r_first[0] == "ㅇ" and r_first[1] == "ㅓ"):
        r = compose("ㄹ", r_first[1], r_first[2]) + ending[1:]
        surface = stem + r
        candidates.add(surface)
        logger.debug("러 불규칙: %s", surface)

    # 여 불규칙 활용
    if l_last_ == "하" and r_first[0] == "ㅇ" and (r_first[1] == "ㅏ" or r_first[1] == "ㅓ"):
        r = compose(r_first[0], "ㅕ", r_first[2]) + ending[1:]
        surface0 = stem + r
        candidates.add(surface0)
        l = stem[:-1] + compose("ㅎ", "ㅐ", r_first[2])
        r = ending[1:]
        surface1 = l + r
        candidates.add(surface1)
        logger.debug("여 불규칙: %s, %s", surface0, surface1)

    # ㅎ (탈락) 불규칙 활용
    if l_last[2] == "ㅎ" and r_first[1] != " ":
        if l_last_ == "좋" or l_last_ == "놓":
            l = stem
        else:
            l = stem[:-1] + compose(l_last[0], l_last[1], " ")
        r = ending
        surface = l + r
        candidates.add(surface)
        logger.debug("ㅎ 탈락 불규칙: %s", surface)

    # ㅎ (축약) 불규칙 활용
    if (l_last[2] == "ㅎ" and l_last_ != "좋") and (r_first[0] == "ㅇ" and (r_first[1] == "ㅏ" or r_first[1] == "ㅓ")):
        l = stem[:-1] + compose(l_last[0], "ㅐ" if r_first[1] == "ㅏ" else "ㅔ", r_first[2])
        r = ending[1:]
        surface = l + r
        candidates.add(surface)
        logger.debug("ㅎ 축약 불규칙: %s", surface)

    # ㅎ + 네 불규칙 활용
    if l_last[2] == "ㅎ" and r_first[0] == "ㄴ" and r_first[1] != " ":
        surface = stem + ending
        candidates.add(surface)
        logger.debug("ㅎ + 네 불규칙: %s", surface)

    # 이 + 어 -> 여 규칙활용
    if r_first_ == "어" and l_last[1] == "ㅣ" and l_last[2] == " ":
        surface = stem[:-1] + compose(l_last[0], "ㅕ", r_first[2]) + ending[1:]
        candidates.add(surface)
        surface = stem + ending
        candidates.add(surface)
        logger.debug("이 + 어 -> 여 규칙: %s", surface)

    if not candidates and r_first[1] != " ":
        if (l_last[2] == " ") and (r_first[0] == "ㅇ") and (r_first[1] == l_last[1]):
            l = stem[:-1] + compose(l_last[0], l_last[1], r_first[2])
            r = ending[1:]
            surface = l + r
            candidates.add(surface)
        else:
            surface = stem + ending
            candidates.add(surface)
        logger.debug("L + R 규칙 결합: %s", surface)

    return candidates


def _conjugate_stem(stem: str, debug: bool = False) -> set[str]:
    l_last = decompose(stem[-1])
    l_last_ = stem[-1]
    l_front = stem[:-1]

    candidates = {stem}

    # ㄷ 불규칙 활용
    if l_last[2] == "ㄷ":
        l = l_front + compose(l_last[0], l_last[1], "ㄹ")
        candidates.add(l)
        logger.debug("ㄷ 불규칙")

    # 르 불규칙 활용
    if (l_last_ == "르") and len(stem) >= 2:
        c0, c1, c2 = decompose(stem[-2])
        l = stem[:-2] + compose(c0, c1, "ㄹ")
        candidates.add(l)
        logger.debug("르 불규칙")

    # ㅂ 불규칙 활용
    if l_last[2] == "ㅂ":
        l = l_front + compose(l_last[0], l_last[1], " ")
        candidates.add(l)
        logger.debug("ㅂ 불규칙")

    # 어미의 첫글자가 종성일 경우
    if l_last[2] == " ":
        candidates.add(l_front + compose(l_last[0], l_last[1], "ㄴ"))
        candidates.add(l_front + compose(l_last[0], l_last[1], "ㄹ"))
        candidates.add(l_front + compose(l_last[0], l_last[1], "ㅂ"))
        candidates.add(l_front + compose(l_last[0], l_last[1], "ㅆ"))
        logger.debug("어미의 첫 글자가 -ㄴ, -ㄹ, -ㅂ, -ㅆ 일 경우")

    # ㅅ 불규칙 활용
    if (l_last[2] == "ㅅ") and stem[-1] != "벗":
        candidates.add(l_front + compose(l_last[0], l_last[1], " "))
        logger.debug("ㅅ 불규칙")

    # 우 불규칙 활용
    if l_last[1] == "ㅜ" and l_last[2] == " ":
        if l_last_ != "푸":
            candidates.add(l_front + compose(l_last[0], "ㅝ", " "))
            candidates.add(l_front + compose(l_last[0], "ㅝ", "ㅆ"))
            logger.debug("우 불규칙")

    # 오 활용
    if l_last[1] == "ㅗ" and l_last[2] == " ":
        candidates.add(l_front + compose(l_last[0], "ㅘ", " "))
        candidates.add(l_front + compose(l_last[0], "ㅘ", "ㅆ"))
        logger.debug("오 + 았어 -> 왔어 규칙")

    # ㅡ 탈락 불규칙 활용
    if l_last[1] == "ㅡ" and l_last[2] == " ":
        candidates.add(l_front + compose(l_last[0], "ㅓ", " "))
        candidates.add(l_front + compose(l_last[0], "ㅓ", "ㅆ"))
        if l_last[0] == "ㅇ" and len(stem) > 1:
            candidates.add(l_front)
        logger.debug("ㅡ 탈락 불규칙")

    # 여 불규칙 활용
    if l_last_ == "하":
        candidates.add(l_front + "해")
        candidates.add(l_front + "했")
        logger.debug("하 -> 해, 했 활용")

    # ㅎ (탈락) 불규칙 활용
    if l_last[2] == "ㅎ" and l_last_ != "좋":
        candidates.add(l_front + compose(l_last[0], l_last[1], " "))
        candidates.add(l_front + compose(l_last[0], l_last[1], "ㄴ"))
        candidates.add(l_front + compose(l_last[0], l_last[1], "ㄹ"))
        candidates.add(l_front + compose(l_last[0], l_last[1], "ㅆ"))
        logger.debug("ㅎ 탈락 불규칙")

    # ㅎ (축약) 불규칙 활용
    if l_last[2] == "ㅎ" and l_last_ != "좋":
        candidates.add(l_front + compose(l_last[0], "ㅐ", "ㅆ"))
        logger.debug("ㅎ 축약 불규칙")

    # 이었 -> 였 규칙활용
    if l_last[1] == "ㅣ" and l_last[2] == " ":
        candidates.add(l_front + compose(l_last[0], "ㅕ", "ㅆ"))
        logger.debug("이었 -> 였 규칙")

    return candidates
