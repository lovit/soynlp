from pathlib import Path
from typing import cast

from .dictionary import Dictionary
from .evaluator import SimpleEojeolEvaluator
from .tagger import MorphTag, SimpleTagger
from .template import EojeolTemplateMatcher

_BASE_DICT_DIR = Path(__file__).parent / "dictionary" / "pos"


def _load_base_dictionary(min_frequency: int = 100) -> Dictionary:
    """내장 사전 데이터(dictionary/pos/)에서 Dictionary를 로드한다.

    Args:
        min_frequency: 로드할 단어의 최소 빈도.

    Returns:
        내장 사전이 적용된 Dictionary 인스턴스.
    """
    pos_dict: dict[str, set[str]] = {}
    for tag_dir in _BASE_DICT_DIR.iterdir():
        if not tag_dir.is_dir():
            continue
        tag = tag_dir.name
        words: set[str] = set()
        for txt_file in tag_dir.glob("*.txt"):
            with open(txt_file, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if not parts or not parts[0]:
                        continue
                    if len(parts) >= 2:
                        try:
                            if int(parts[1]) >= min_frequency:
                                words.add(parts[0])
                        except ValueError:
                            words.add(parts[0])
                    else:
                        words.add(parts[0])
        if words:
            pos_dict[tag] = words
    return Dictionary(pos_dict)


class KoreanPOSTagger:
    """한국어 형태소 분석기.

    LRNounExtractor와 동일한 패턴으로 기본값만으로 즉시 사용 가능하다.
    내장 사전(dictionary/pos/)을 기반으로 EojeolTemplateMatcher + LREvaluator를
    자동으로 구성하여 사전 조립 과정 없이 바로 분석을 수행할 수 있다.

    Example:
        >>> # 기본 사용 — LRNounExtractor 패턴과 동일
        >>> tagger = KoreanPOSTagger.default()
        >>> result = tagger.tag("나는 학교에 갔다")
        >>> result[0].surface, result[0].tag
        ('나는', 'Noun')

        >>> # train()으로 추가 어휘 등록
        >>> tagger.train(sentences, extra_nouns={"ChatGPT", "딥러닝"})

        >>> # 고급 사용 — 커스텀 컴포넌트
        >>> custom_dict = Dictionary({"Noun": {"사과", "배"}, "Josa": {"를", "이"}})
        >>> tagger = KoreanPOSTagger(dictionary=custom_dict)
    """

    def __init__(
        self,
        dictionary: Dictionary | None = None,
        evaluator: SimpleEojeolEvaluator | None = None,
        min_frequency: int = 100,
    ) -> None:
        self._dictionary = dictionary if dictionary is not None else _load_base_dictionary(min_frequency)
        self._evaluator = evaluator if evaluator is not None else SimpleEojeolEvaluator()
        self._matcher = EojeolTemplateMatcher(self._dictionary)
        self._tagger = SimpleTagger(self._matcher, self._evaluator)

    @classmethod
    def default(cls) -> "KoreanPOSTagger":
        """검증된 기본 파라미터(내장 사전, 기본 Evaluator)로 인스턴스를 생성한다.

        Returns:
            기본 파라미터가 적용된 KoreanPOSTagger 인스턴스.
        """
        return cls()

    @property
    def is_trained(self) -> bool:
        """내장 사전이 로드되면 True를 반환한다. 사전 기반 분석기는 항상 True."""
        return bool(self._dictionary.pos_dict)

    def __repr__(self) -> str:
        num_words = sum(len(w) for w in self._dictionary.pos_dict.values())
        return f"KoreanPOSTagger(trained={self.is_trained}, vocab_size={num_words})"

    def train(
        self,
        sentences: list[str] | None = None,
        extra_nouns: set[str] | None = None,
        extra_words: dict[str, set[str]] | None = None,
    ) -> None:
        """추가 어휘를 사전에 등록한다.

        사전 기반 분석기이므로 sentences로부터 통계 학습을 하지 않는다.
        도메인 어휘를 추가하려면 extra_nouns 또는 extra_words를 사용한다.

        Args:
            sentences: 사용하지 않음 (LRNounExtractor 패턴 호환용).
            extra_nouns: 명사 사전에 추가할 단어 집합.
            extra_words: 태그별 추가 단어 dict. 예: {"Noun": {"ChatGPT"}, "Josa": {"ㅇ"}}
        """
        if extra_nouns:
            self._dictionary.add_words("Noun", extra_nouns, force=False)
        if extra_words:
            for tag, words in extra_words.items():
                self._dictionary.add_words(tag, words, force=True)

    def tag(self, text: str) -> list[MorphTag]:
        """문장의 형태소를 분석한다.

        Args:
            text: 분석할 문장.

        Returns:
            형태소 분석 결과 MorphTag 리스트.
        """
        return cast(list[MorphTag], self._tagger.tag(text))
