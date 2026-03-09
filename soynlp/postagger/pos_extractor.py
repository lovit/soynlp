import logging
from typing import Protocol, runtime_checkable

from soynlp.noun import LRNounExtractor
from soynlp.predicator import PredicatorExtractor

logger = logging.getLogger(__name__)


@runtime_checkable
class ExtractorStepProtocol(Protocol):
    """POSExtractor 파이프라인에 삽입 가능한 추출 단계 인터페이스.

    각 단계는 문장(또는 lrgraph)과 이전 단계 결과를 담은 context를 받아,
    자신의 추출 결과를 dict로 반환한다. 반환된 dict는 다음 단계의 context에 병합된다.

    context 주요 키:
        - "nouns" (dict): 명사 추출 결과
        - "predicators" (dict): 용언 추출 결과
        - "lrgraph": LR 그래프 (LRNounExtractor가 설정)

    Example:
        >>> class MyDomainExtractor:
        ...     def extract(self, sentences, context: dict) -> dict:
        ...         nouns = context.get("nouns", {})
        ...         # 도메인 용어를 nouns에 추가
        ...         nouns.update({"항체": 1.0, "세포": 1.0})
        ...         return {"nouns": nouns}
        ...
        >>> extractor = POSExtractor(extra_steps=[MyDomainExtractor()])
        >>> nouns, predicators = extractor.extract(sentences)
    """

    def extract(self, sentences, context: dict) -> dict:
        """추출을 수행하고 결과를 반환한다.

        Args:
            sentences: 입력 문장 리스트 또는 lrgraph.
            context: 이전 단계 결과 및 공유 상태.

        Returns:
            이번 단계 추출 결과. context에 병합된다.
        """
        ...


class _NounExtractorStep:
    """LRNounExtractor를 ExtractorStepProtocol로 감싸는 기본 명사 추출 단계."""

    def __init__(self, l_max_length: int, r_max_length: int, verbose: bool) -> None:
        self.l_max_length = l_max_length
        self.r_max_length = r_max_length
        self.verbose = verbose

    def extract(self, sentences, context: dict) -> dict:
        noun_extractor = LRNounExtractor(
            max_l_length=self.l_max_length,
            max_r_length=self.r_max_length,
            verbose=self.verbose,
        )
        nouns = noun_extractor.extract(sentences, min_noun_score=0.4, min_noun_frequency=10)
        context["_noun_extractor"] = noun_extractor
        return {"nouns": nouns, "lrgraph": noun_extractor.lrgraph}


class _PredicatorExtractorStep:
    """PredicatorExtractor를 ExtractorStepProtocol로 감싸는 기본 용언 추출 단계."""

    def __init__(self, verbose: bool) -> None:
        self.verbose = verbose

    def extract(self, sentences, context: dict) -> dict:
        nouns = context.get("nouns")
        lrgraph = context.get("lrgraph", sentences)
        predicator_extractor = PredicatorExtractor(
            nouns,  # type: ignore[arg-type]
            extract_eomi=True,
            extract_stem=True,
            verbose=self.verbose,
        )
        predicator_extractor.train(lrgraph, min_eojeol_frequency=2)
        predicators = predicator_extractor.extract(candidates=None, min_predicator_frequency=10)
        context["_predicator_extractor"] = predicator_extractor
        return {"predicators": predicators}


class POSExtractor:
    """명사·용언 추출 파이프라인.

    기본 동작은 LRNounExtractor → PredicatorExtractor 순서로 추출을 수행한다.
    extra_steps를 통해 도메인 특화 단계를 중간에 삽입할 수 있다.

    Args:
        l_max_length: LR 그래프 L 최대 길이.
        r_max_length: LR 그래프 R 최대 길이.
        verbose: 진행 상황 출력 여부.
        logpath: 로그 파일 경로.
        extra_steps: 명사 추출 직후, 용언 추출 직전에 실행할 추가 단계 목록.
            각 단계는 ExtractorStepProtocol을 구현해야 한다.

    Example:
        >>> # 기본 사용
        >>> extractor = POSExtractor()
        >>> nouns, predicators = extractor.extract(sentences)

        >>> # 도메인 특화 단계 삽입
        >>> class MedicalTermStep:
        ...     def extract(self, sentences, context: dict) -> dict:
        ...         nouns = context.get("nouns", {})
        ...         nouns.update({"항체": 1.0})
        ...         return {"nouns": nouns}
        ...
        >>> extractor = POSExtractor(extra_steps=[MedicalTermStep()])
        >>> nouns, predicators = extractor.extract(sentences)
    """

    def __init__(
        self,
        l_max_length: int = 10,
        r_max_length: int = 8,
        verbose: bool = True,
        logpath: str | None = None,
        extra_steps: list[ExtractorStepProtocol] | None = None,
    ):
        self.l_max_length = l_max_length
        self.r_max_length = r_max_length
        self.verbose = verbose
        self.logpath = logpath
        self.extra_steps: list[ExtractorStepProtocol] = extra_steps or []

    @property
    def is_trained(self) -> bool:
        """extract()가 완료된 경우 True를 반환한다."""
        return hasattr(self, "noun_extractor") and hasattr(self, "predicator_extractor")

    def __repr__(self) -> str:
        return f"POSExtractor(trained={self.is_trained}, l_max_length={self.l_max_length}, r_max_length={self.r_max_length})"

    def extract(self, sentences) -> tuple[dict, dict]:
        """명사와 용언을 추출한다.

        Args:
            sentences: 학습에 사용할 문장 리스트.

        Returns:
            (nouns, predicators) 튜플.
        """
        context: dict = {}

        # 1단계: 명사 추출
        noun_step = _NounExtractorStep(self.l_max_length, self.r_max_length, self.verbose)
        context.update(noun_step.extract(sentences, context))
        self.noun_extractor = context.pop("_noun_extractor")
        self._log_coverage(context, "noun extraction")

        # 2단계: 사용자 정의 추가 단계 (명사 추출 후, 용언 추출 전)
        for step in self.extra_steps:
            context.update(step.extract(sentences, context))

        # 3단계: 용언 추출
        pred_step = _PredicatorExtractorStep(self.verbose)
        context.update(pred_step.extract(sentences, context))
        self.predicator_extractor = context.pop("_predicator_extractor")

        nouns = context.get("nouns", {})
        predicators = context.get("predicators", {})

        logger.info("%d nouns, %d predicators were extracted", len(nouns), len(predicators))
        return nouns, predicators

    def _log_coverage(self, context: dict, stage: str) -> None:
        noun_extractor = self.noun_extractor
        num_eojeols = getattr(noun_extractor, "_num_of_eojeols", 0)
        num_covered = getattr(noun_extractor, "_num_of_covered_eojeols", 0)
        if num_eojeols > 0:
            pct = 100 * num_covered / num_eojeols
            logger.info("%s was done. %.2f %% eojeols are covered", stage, pct)
