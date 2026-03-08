import logging

from soynlp.hangle import compose

from ._conjugation import conjugate, decompose

logger = logging.getLogger(__name__)


class Lemmatizer:
    def __init__(
        self,
        stems: set[str],
        endings: set[str],
        predefined: dict | None = None,
    ) -> None:
        self._stems = stems
        self._endings = endings
        self._initialize()
        if predefined:
            self._predefined.update(predefined)

    def _initialize(self) -> None:
        self._predefined: dict = {
            "불어": ("붇다", "불다"),
            "그래": ("그렇다",),
        }

    def lemmatize(self, word: str, check_only_stem: bool = False) -> set[tuple[str, str]]:
        candidates: set[tuple[str, str]] = set()
        for i in range(1, len(word) + 1):
            l, r = word[:i], word[i:]
            for stem, ending in lemma_candidate(l, r, self._predefined):
                if stem in self._stems:
                    if check_only_stem:
                        candidates.add((stem, ending))
                    elif ending in self._endings:
                        candidates.add((stem, ending))
        return candidates

    def candidates(self, word: str) -> set[tuple[str, str]]:
        candidates: set[tuple[str, str]] = set()
        for i in range(1, len(word) + 1):
            l = word[:i]
            r = word[i:]
            candidates.update(lemma_candidate(l, r, self._predefined))
        return candidates


def lemma_candidate_chat(
    l: str,
    r: str,
    predefined: dict[tuple[str, str], tuple[str, ...]] | None = None,
    debug: bool = False,
) -> set[tuple[str, str]]:
    def character_is_emoticon(c: str) -> bool:
        return c in set("ㄷㅂㅅㅇㅋㅎ")

    candidates = lemma_candidate(l, r, predefined, debug)
    l_last = decompose(l[-1])

    if not r and character_is_emoticon(l_last[2]):
        l_ = l[:-1] + compose(l_last[0], l_last[1], " ")
        logger.debug("마지막 종성이 이모티콘으로 의심되는 경우: %s + ()", l_)
        candidates.update(lemma_candidate(l_, r, predefined, debug))

    return candidates


def lemma_candidate(
    l: str,
    r: str,
    predefined: dict | None = None,
    debug: bool = False,
) -> set[tuple[str, str]]:
    def add_lemma(stem: str, ending: str) -> None:
        candidates.add((stem, ending))

    def debug_message(message: str, left: str, right: str) -> None:
        logger.debug("%s: %s + %s", message, left, right)

    candidates: set[tuple[str, str]] = {(l, r)}
    word = l + r

    l_last = decompose(l[-1])
    l_last_ = compose(l_last[0], l_last[1], " ")
    l_front = l[:-1]
    r_first = decompose(r[0]) if r else ("", "", "")
    r_first_ = compose(r_first[0], r_first[1], " ") if r else " "
    r_end = r[1:]

    # ㄷ 불규칙 활용: 깨달 + 아 -> 깨닫 + 아
    if l_last[2] == "ㄹ" and r_first[0] == "ㅇ":
        l_stem = l_front + compose(l_last[0], l_last[1], "ㄷ")
        add_lemma(l_stem, r)
        if debug:
            debug_message("ㄷ 불규칙 활용", l_stem, r)

    # 르 불규칙 활용: 굴 + 러 -> 구르 + 어
    if (l_last[2] == "ㄹ") and (r_first_ == "러" or r_first_ == "라"):
        l_stem = l_front + compose(l_last[0], l_last[1], " ") + "르"
        r_canon = compose("ㅇ", r_first[1], r_first[2]) + r_end
        add_lemma(l_stem, r_canon)
        if debug:
            debug_message("르 불규칙 활용", l_stem, r_canon)

    # ㅂ 불규칙 활용: 더러 + 워서 -> 더럽 + 어서
    if l_last[2] == " ":
        l_stem = l_front + compose(l_last[0], l_last[1], "ㅂ")
        if r_first_ == "워" or r_first_ == "와":
            r_canon = compose("ㅇ", "ㅏ" if r_first_ == "와" else "ㅓ", r_first[2] if r_first[2] else " ") + r_end
        elif r_end and r_end[0] == "려":
            r_canon = compose("ㅇ", "ㅜ", r_first[2] if r_first[2] else " ") + r_end
        else:
            r_canon = r
        add_lemma(l_stem, r_canon)
        if debug:
            debug_message("ㅂ 불규칙 활용", l_stem, r_canon)

    # 어미의 첫글자가 종성일 경우 (-ㄴ, -ㄹ, -ㅁ-, -ㅂ, -ㅆ)
    if l_last[2] in ("ㄴ", "ㄹ", "ㅁ", "ㅂ", "ㅆ"):
        for jongsung in " ㄹㅂㅎ":
            if l_last[2] == jongsung:
                continue
            l_stem = l_front + compose(l_last[0], l_last[1], jongsung)
            r_canon = l_last[2] + r
            add_lemma(l_stem, r_canon)
            if debug:
                debug_message("어미의 첫글자가 종성일 경우 (%s)" % jongsung, l_stem, r_canon)

    # ㅅ 불규칙 활용: 부 + 어 -> 붓 + 어
    if (l_last[2] == " " and l[-1] != "벗") and (r_first[0] == "ㅇ"):
        l_stem = l_front + compose(l_last[0], l_last[1], "ㅅ")
        add_lemma(l_stem, r)
        if debug:
            debug_message("ㅅ 불규칙 활용", l_stem, r)

    # 우 불규칙 활용: 똥퍼 + '' -> 똥푸 + 어
    if l_last_ == "퍼":
        l_stem = l_front + "푸"
        r_canon = compose("ㅇ", l_last[1], l_last[2]) + r
        add_lemma(l_stem, r_canon)
        if debug:
            debug_message("우 불규칙 활용 (퍼)", l_stem, r_canon)

    # 우 불규칙 활용: 줬 + 어 -> 주 + 었어
    if l_last[1] == "ㅝ":
        l_stem = l_front + compose(l_last[0], "ㅜ", " ")
        r_canon = compose("ㅇ", "ㅓ", l_last[2]) + r
        add_lemma(l_stem, r_canon)
        if debug:
            debug_message("우 불규칙 활용", l_stem, r_canon)

    # 오 불규칙 활용: 왔 + 어 -> 오 + 았어
    if l_last[1] == "ㅘ":
        l_stem = l_front + compose(l_last[0], "ㅗ", " ")
        r_canon = compose("ㅇ", "ㅏ", l_last[2]) + r
        add_lemma(l_stem, r_canon)
        if debug:
            debug_message("오 불규칙 활용", l_stem, r_canon)

    # ㅡ 탈락 불규칙 활용: 꺼 + '' -> 끄 + 어
    if l_last[1] == "ㅓ" or l_last[1] == "ㅏ":
        l_stem = l_front + compose(l_last[0], "ㅡ", " ")
        r_canon = compose("ㅇ", l_last[1], l_last[2]) + r
        add_lemma(l_stem, r_canon)
        if debug:
            debug_message("ㅡ 탈락 불규칙 활용 (꺼)", l_stem, r_canon)

    # ㅡ 탈락 불규칙 활용: 모 + 았다 -> 모으 + 았다
    if l_last[2] == " " and r_first[0] == "ㅇ" and (r_first[1] == "ㅏ" or r_first[1] == "ㅓ"):
        l_stem = l + "으"
        r_canon = r
        add_lemma(l_stem, r_canon)
        if debug:
            debug_message("ㅡ 탈락 불규칙 활용 (모으)", l_stem, r_canon)

    # 여 불규칙 활용 (2)
    if l_last[0] == "ㅎ" and l_last[1] == "ㅐ":
        l_stem = l_front + "하"
        r_canon = compose("ㅇ", "ㅏ", l_last[2]) + r
        add_lemma(l_stem, r_canon)
        if debug:
            debug_message("여 불규칙 활용", l_stem, r_canon)

    # ㅎ (탈락) 불규칙 활용
    if l_last[2] in (" ", "ㄴ", "ㄹ", "ㅂ", "ㅆ"):
        if l_last[1] == "ㅏ" or l_last[1] == "ㅓ":
            l_stem = l_front + compose(l_last[0], l_last[1], "ㅎ")
            r_canon = r if l_last[2] == " " else l_last[2] + r
            add_lemma(l_stem, r_canon)
            if debug:
                debug_message("ㅎ 탈락 불규칙 활용", l_stem, r_canon)
        if l_last[1] == "ㅐ" or l_last[1] == "ㅔ":
            if len(l) >= 2 and l[-2] == "그" and l_last[0] == "ㄹ":
                l_stem = l_front + "렇"
            else:
                l_stem = l_front + compose(l_last[0], "ㅓ" if l_last[1] == "ㅔ" else "ㅏ", "ㅎ")
            r_canon = compose("ㅇ", "ㅓ" if l_last[1] == "ㅔ" else "ㅏ", l_last[2]) + r
            add_lemma(l_stem, r_canon)
            if debug:
                debug_message("ㅎ 축약 불규칙 활용", l_stem, r_canon)

    # 이었 -> 였 규칙활용
    if (l_last[2] in ("ㅆ", "ㅅ", " ")) and (l_last[1] == "ㅕ"):
        if ((l_last[0] == "ㅇ") and (l_last[1] == "ㅕ")) or not (l_last[0] == "ㅇ"):
            l_stem = l_front + compose(l_last[0], "ㅣ", " ")
            r_canon = compose("ㅇ", "ㅓ", l_last[2]) + r
            add_lemma(l_stem, r_canon)
            if debug:
                debug_message("이었 -> 였 규칙 활용", l_stem, r_canon)

    # Pre-defined set
    if predefined and (l, r) in predefined:
        for stem in predefined[(l, r)]:
            candidates.add(stem)
            logger.debug("Predefined: %s", stem)

    # check whether lemma is conjugatable
    candidates_ = set()
    for item in candidates:
        if isinstance(item, tuple) and len(item) == 2:
            stem, eomi = item
        else:
            continue
        if not eomi:
            continue
        if decompose(eomi[0])[2] == "ㅎ":
            continue
        surfaces = conjugate(stem, eomi)
        if word in surfaces:
            candidates_.add((stem, eomi))
    return candidates_
