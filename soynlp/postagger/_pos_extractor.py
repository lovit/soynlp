from soynlp.noun import LRNounExtractor
from soynlp.predicator import PredicatorExtractor


class POSExtractor:
    """Unified POS extractor combining noun and predicator extraction.

    Note: This class references LRNounExtractor_v2 in the original code,
    which is not yet available. Currently uses LRNounExtractor as a fallback.
    """

    def __init__(
        self,
        l_max_length: int = 10,
        r_max_length: int = 8,
        verbose: bool = True,
        logpath: str | None = None,
    ):
        self.l_max_length = l_max_length
        self.r_max_length = r_max_length
        self.verbose = verbose
        self.logpath = logpath

    def _print(self, message: str, replace: bool = False, newline: bool = True):
        header = "[POS Extractor]"
        if replace:
            print(f"\r{header} {message}", end="\n" if newline else "", flush=True)
        else:
            print(f"{header} {message}", end="\n" if newline else "", flush=True)

    def extract(self, sentences):
        self._num_of_eojeols = 0
        self._num_of_covered_eojeols = 0

        nouns = self._extract_nouns(sentences)
        predicators = self._extract_predicators(self._lrgraph, nouns)

        del self._lrgraph

        if self.verbose:
            self._print(
                f"{len(nouns)} nouns, {len(predicators)} predicators were extracted",
                replace=True,
                newline=True,
            )

        return nouns, predicators

    def _extract_nouns(self, sentences):
        noun_extractor = LRNounExtractor(
            max_l_length=self.l_max_length,
            max_r_length=self.r_max_length,
            verbose=self.verbose,
        )

        nouns = noun_extractor.extract(sentences, min_noun_score=0.4, min_noun_frequency=10)

        self._lrgraph = noun_extractor.lrgraph
        self._num_of_eojeols = getattr(noun_extractor, "_num_of_eojeols", 0)
        self._num_of_covered_eojeols = getattr(noun_extractor, "_num_of_covered_eojeols", 0)
        self.noun_extractor = noun_extractor

        if self.verbose and self._num_of_eojeols > 0:
            pct = 100 * self._num_of_covered_eojeols / self._num_of_eojeols
            self._print(f"noun extraction was done. {pct:.2f} % eojeols are covered", replace=True, newline=True)

        return nouns

    def _extract_predicators(self, sentences_or_lrgraph, nouns=None):
        predicator_extractor = PredicatorExtractor(
            nouns,  # type: ignore[arg-type]
            extract_eomi=True,
            extract_stem=True,
            verbose=self.verbose,
        )

        predicator_extractor.train(sentences_or_lrgraph, min_eojeol_frequency=2)
        predicators = predicator_extractor.extract(candidates=None, min_predicator_frequency=10)

        self._num_of_covered_eojeols += predicator_extractor._num_of_covered_eojeols
        self.predicator_extractor = predicator_extractor

        if self.verbose and self._num_of_eojeols > 0:
            pct = 100 * self._num_of_covered_eojeols / self._num_of_eojeols
            self._print(
                f"predicator extraction was done. {pct:.2f} % eojeols are covered (cum)",
                replace=True,
                newline=True,
            )

        return predicators
