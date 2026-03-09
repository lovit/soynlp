import json
from dataclasses import dataclass

from .dictionary import DictionaryProtocol


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
    dictionary: DictionaryProtocol

    def generate(self, token: str) -> list[list[LR]]:
        raise NotImplementedError


class EojeolTemplateMatcher(BaseTemplateMatcher):
    """어절 단위 template 매처.

    어절 전체를 단일 품사로 분석하거나, L+R 조합으로 분석하는 후보를 생성한다.

    Template 파일 포맷 (JSON):
        {
            "single_tags": ["Noun", "Verb", "Adjective", "Adverb", "Exclamation"],
            "lr_templates": [["Noun", "Verb"], ["Noun", "Adjective"], ["Noun", "Josa"]]
        }

        - single_tags: 어절 전체를 하나의 품사로 인정할 태그 목록
        - lr_templates: 허용할 (L_tag, R_tag) 조합 목록

    Example:
        >>> # 파일에서 로드
        >>> matcher = EojeolTemplateMatcher.from_file("my_template.json", dictionary)

        >>> # 파라미터로 직접 지정
        >>> matcher = EojeolTemplateMatcher(
        ...     dictionary,
        ...     single_tags=["Noun", "Verb"],
        ...     lr_templates=[("Noun", "Josa")],
        ... )
    """

    def __init__(
        self,
        dictionary: DictionaryProtocol,
        single_tags: list[str] | None = None,
        lr_templates: list[tuple[str, str]] | None = None,
        template_path: str | None = None,
    ) -> None:
        if template_path is not None:
            loaded = self._load_template(template_path)
            single_tags = loaded["single_tags"]
            lr_templates = [tuple(pair) for pair in loaded["lr_templates"]]  # type: ignore[misc]
        if not single_tags:
            single_tags = ["Noun", "Verb", "Adjective", "Adverb", "Exclamation"]
        if not lr_templates:
            lr_templates = [("Noun", "Verb"), ("Noun", "Adjective"), ("Noun", "Josa")]
        self.dictionary = dictionary
        self.single_tags = single_tags
        self.lr_templates = lr_templates

    @classmethod
    def from_file(cls, template_path: str, dictionary: DictionaryProtocol) -> "EojeolTemplateMatcher":
        """JSON 파일에서 template을 로드하여 인스턴스를 생성한다.

        Args:
            template_path: JSON template 파일 경로.
            dictionary: 사용할 사전 객체.

        Returns:
            파일에서 로드된 template이 적용된 EojeolTemplateMatcher 인스턴스.
        """
        return cls(dictionary, template_path=template_path)

    def save(self, path: str) -> None:
        """현재 template 설정을 JSON 파일로 저장한다.

        Args:
            path: 저장할 파일 경로.
        """
        data = {
            "single_tags": self.single_tags,
            "lr_templates": [list(pair) for pair in self.lr_templates],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _load_template(path: str) -> dict:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

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
    """LR 분리 기반 template 매처.

    어절을 L(체언/용언 어간)과 R(조사/어미) 조합으로 분석하는 후보를 생성한다.

    Template 파일 포맷 (JSON):
        {
            "ltags": ["Noun", "Adjective", "Verb", "Adverb", "Exclamation"],
            "templates": {
                "Noun": ["Josa", "Verb", "Adjective"]
            }
        }

        - ltags: L 위치에 허용할 품사 태그 목록
        - templates: L_tag → 허용 R_tag 목록 매핑

    Example:
        >>> # 파일에서 로드
        >>> matcher = LRTemplateMatcher.from_file("my_template.json", dictionary)

        >>> # 파라미터로 직접 지정
        >>> matcher = LRTemplateMatcher(
        ...     dictionary,
        ...     ltags={"Noun"},
        ...     templates={"Noun": ("Josa",)},
        ... )
    """

    def __init__(
        self,
        dictionary: DictionaryProtocol,
        ltags: set[str] | None = None,
        templates: dict[str, tuple[str, ...]] | None = None,
        template_path: str | None = None,
    ) -> None:
        if template_path is not None:
            loaded = self._load_template(template_path)
            ltags = set(loaded["ltags"])
            templates = {k: tuple(v) for k, v in loaded["templates"].items()}
        if not ltags:
            ltags = {"Noun", "Adjective", "Verb", "Adverb", "Exclamation"}
        if not templates:
            templates = {"Noun": ("Josa", "Verb", "Adjective")}

        self.dictionary = dictionary
        self.ltags = ltags
        self.rtags = {tag for tags in templates.values() for tag in tags}
        self.templates = templates

    @classmethod
    def from_file(cls, template_path: str, dictionary: DictionaryProtocol) -> "LRTemplateMatcher":
        """JSON 파일에서 template을 로드하여 인스턴스를 생성한다.

        Args:
            template_path: JSON template 파일 경로.
            dictionary: 사용할 사전 객체.

        Returns:
            파일에서 로드된 template이 적용된 LRTemplateMatcher 인스턴스.
        """
        return cls(dictionary, template_path=template_path)

    def save(self, path: str) -> None:
        """현재 template 설정을 JSON 파일로 저장한다.

        Args:
            path: 저장할 파일 경로.
        """
        data = {
            "ltags": list(self.ltags),
            "templates": {k: list(v) for k, v in self.templates.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _load_template(path: str) -> dict:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

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
