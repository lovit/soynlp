from dataclasses import dataclass

from .evaluator import BaseEvaluator
from .template import LR, BaseTemplateMatcher


@dataclass(frozen=True, slots=True)
class MorphTag:
    """형태소 분석 결과 단위.

    Attributes:
        surface: 표층형 (실제 텍스트)
        tag: 품사 태그. 미등록어인 경우 None.

    Example:
        >>> MorphTag(surface="사과", tag="Noun")
        MorphTag(surface='사과', tag='Noun')
        >>> MorphTag(surface="를", tag="Josa")
        MorphTag(surface='를', tag='Josa')
    """

    surface: str
    tag: str | None


class BaseTagger:
    def __init__(
        self, generator: BaseTemplateMatcher, evaluator: BaseEvaluator, postprocessor: "BasePostprocessor | None" = None
    ) -> None:
        self.evaluator = evaluator
        self.generator = generator
        self.dictionary = generator.dictionary
        self.postprocessor = postprocessor

    def tag(self, sentence: str, flatten: bool = True, debug: bool = False) -> list | tuple[list, list]:
        raise NotImplementedError


class SimpleTagger(BaseTagger):
    def tag(
        self, sentence: str, flatten: bool = True, debug: bool = False
    ) -> list[MorphTag] | list[list[MorphTag]] | tuple[list[MorphTag], list] | tuple[list[list[MorphTag]], list]:
        """문장을 형태소 분석한다.

        Args:
            sentence: 분석할 문장.
            flatten: True이면 모든 어절의 결과를 하나의 리스트로 합쳐 반환.
                     False이면 어절 단위 리스트의 리스트로 반환.
            debug: True이면 (결과, 디버그 정보) 튜플로 반환.

        Returns:
            flatten=True, debug=False: list[MorphTag]
            flatten=False, debug=False: list[list[MorphTag]]
            debug=True: 위 결과와 디버그 정보의 tuple
        """
        sent_: list[list[MorphTag]] = []
        debug_: list[list] = []
        eojeols = sentence.split()

        for eojeol in eojeols:
            candidates = self.generator.generate(eojeol)
            best = self.evaluator.select_best(candidates) or []

            if self.postprocessor:
                postprocessed = self.postprocessor.postprocess(eojeol, best)
            else:
                postprocessed = best

            postprocessed_: list[MorphTag] = []
            for word in postprocessed:
                if word.l:
                    postprocessed_.append(MorphTag(word.l, word.l_tag))
                if word.r:
                    postprocessed_.append(MorphTag(word.r, word.r_tag))

            sent_.append(postprocessed_)

            if debug:
                scored_candidates = [(c, self.evaluator.evaluate(c)) for c in candidates]
                scored_candidates = sorted(scored_candidates, key=lambda x: (x[0][0].b if x[0] else 0, x[1]))
                debug_.append(scored_candidates)

        if flatten:
            flat = [word for words in sent_ for word in words]
            if not debug:
                return flat
            return flat, debug_

        if not debug:
            return sent_
        return sent_, debug_


class BasePostprocessor:
    def postprocess(self, token: str, best_wordstream: list[LR]) -> list[LR]:
        return best_wordstream


class UnknownLRPostprocessor(BasePostprocessor):
    def postprocess(self, token: str, words: list[LR]) -> list[LR]:
        n = len(token)
        adds: list[LR] = []
        if words and words[0].b > 0:
            adds.append(self._add_first_subword(token, words))
        if words and words[-1].e < n:
            adds.append(self._add_last_subword(token, words, n))
        adds += self._add_inter_subwords(token, words)
        post = words + adds
        return sorted(post, key=lambda x: x.b)

    def _add_last_subword(self, token: str, words: list[LR], n: int) -> LR:
        b = words[-1].e
        subword = token[b:]
        return LR(subword, None, "", None, b, n, n)

    def _add_first_subword(self, token: str, words: list[LR]) -> LR:
        e = words[0].b
        subword = token[0:e]
        return LR(subword, None, "", None, 0, e, e)

    def _add_inter_subwords(self, token: str, words: list[LR]) -> list[LR]:
        adds: list[LR] = []
        for i, base in enumerate(words[:-1]):
            if base.e == words[i + 1].b:
                continue
            b = base.e
            e = words[i + 1].b
            subword = token[b:e]
            adds.append(LR(subword, None, "", None, b, e, e))
        return adds
