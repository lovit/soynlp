from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LR:
    """LR decomposition result. b: begin, m: middle, e: end."""

    l: str
    l_tag: str | None
    r: str
    r_tag: str | None
    b: int
    m: int
    e: int


class BaseTemplateMatcher:
    def generate(self, token: str) -> list[list[LR]]:
        raise NotImplementedError


class EojeolTemplateMatcher(BaseTemplateMatcher):
    def __init__(
        self,
        dictionary,
        single_tags: list[str] | None = None,
        lr_templates: list[tuple[str, str]] | None = None,
    ) -> None:
        if not single_tags:
            single_tags = ["Noun", "Verb", "Adjective", "Adverb", "Exclamation"]
        if not lr_templates:
            lr_templates = [("Noun", "Verb"), ("Noun", "Adjective"), ("Noun", "Josa")]
        self.dictionary = dictionary
        self.single_tags = single_tags
        self.lr_templates = lr_templates

    def generate(self, eojeol: str) -> list[list[LR]]:
        n = len(eojeol)
        candidates: list[list[LR]] = []
        for tag in self.dictionary.get_pos(eojeol):
            if tag in self.single_tags:
                candidates.append([LR(eojeol, tag, "", None, 0, n, n)])
        if not candidates:
            candidates.append([LR(eojeol, None, "", None, 0, n, n)])

        for b in range(1, n):
            l, r = eojeol[:b], eojeol[b:]
            for l_tag, r_tag in self.lr_templates:
                if self.dictionary.word_is_tag(l, l_tag) and self.dictionary.word_is_tag(r, r_tag):
                    candidates.append([LR(l, l_tag, r, r_tag, 0, b, n)])

        compound_noun = self._decompose_compound(eojeol, "Noun")
        if compound_noun:
            candidates.append(compound_noun)

        compound_adverb = self._decompose_compound(eojeol, "Adverb")
        if compound_adverb:
            candidates.append(compound_adverb)

        return candidates

    def _decompose_compound(self, eojeol: str, tag: str) -> list[LR]:
        n, b = len(eojeol), 0
        words: list[LR] = []
        while b < n:
            next_round = False
            for i in range(b + 1, n + 1):
                if next_round or (b == 0 and i == n):
                    break
                subword = eojeol[b:i]
                if self.dictionary.word_is_tag(subword, tag):
                    words.append(LR(subword, tag, "", None, b, i, i))
                    b = i
                    next_round = True
                    break
            if not next_round:
                return []
        return words


class LRTemplateMatcher(BaseTemplateMatcher):
    def __init__(
        self,
        dictionary,
        ltags: set[str] | None = None,
        templates: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        if not ltags:
            ltags = {"Noun", "Adjective", "Verb", "Adverb", "Exclamation"}
        if not templates:
            templates = {"Noun": ("Josa", "Verb", "Adjective")}

        self.dictionary = dictionary
        self.ltags = ltags
        self.rtags = {tag for tags in templates.values() for tag in tags}
        self.templates = templates

    def generate(self, token: str) -> list[LR]:
        candidates = self._initialize_L(token)
        candidates = self._expand_R(token, candidates)
        return candidates

    def _pos_L(self, word: str) -> set[str]:
        poses = self.dictionary.get_pos(word)
        poses = {pos for pos in poses if pos in self.ltags}
        return poses

    def _initialize_L(self, t: str) -> list[list]:
        n = len(t)
        candidates: list[list] = []

        for b in range(n):
            for e in range(b + 2, min(n, b + self.dictionary.max_length) + 1):
                l = t[b:e]
                l_tags = self._pos_L(l)

                if not l_tags:
                    continue

                for l_tag in l_tags:
                    candidates.append([l, l_tag, b, e])

        return sorted(candidates, key=lambda x: x[2])

    def _expand_R(self, t: str, candidates: list[list]) -> list[LR]:
        n = len(t)
        expanded: list[LR] = []

        for l, l_tag, b, e1 in candidates:
            last = min(self.dictionary.max_length + e1, n)

            for e2 in range(e1, last + 1):
                r = t[e1:e2]

                if not r:
                    expanded.append(LR(l, l_tag, r, None, b, e1, e2))
                else:
                    for r_tag in self.templates.get(l_tag, []):
                        if not self.dictionary.word_is_tag(r, r_tag):
                            continue
                        expanded.append(LR(l, l_tag, r, r_tag, b, e1, e2))

        return sorted(expanded, key=lambda x: x.b)
