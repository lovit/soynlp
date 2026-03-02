from collections import namedtuple

from soynlp.lemmatizer import _conjugate_stem, lemma_candidate

EomiScore = namedtuple("EomiScore", "frequency score")


class EomiExtractor:
    def __init__(
        self,
        lrgraph,
        stems: set[str],
        nouns: set[str],
        min_num_of_features: int = 5,
        verbose: bool = True,
        logpath: str | None = None,
    ):
        self.lrgraph = lrgraph
        self._stems = stems
        self._nouns = nouns
        self.min_num_of_features = min_num_of_features
        self.verbose = verbose
        self.logpath = logpath
        self._eomis: dict[str, EomiScore] | None = None

    @property
    def is_trained(self) -> bool:
        return self._eomis is not None

    def _print(self, message: str, replace: bool = False, newline: bool = True):
        header = "[Eomi Extractor]"
        if replace:
            print(f"\r{header} {message}", end="\n" if newline else "", flush=True)
        else:
            print(f"{header} {message}", end="\n" if newline else "", flush=True)

    def extract(
        self,
        condition: str | None = None,
        min_eomi_score: float = 0.3,
        min_eomi_frequency: int = 1,
        reset_lrgraph: bool = True,
    ) -> dict[str, EomiScore]:
        self._num_of_covered_eojeols = 0
        self._eomis = {}
        self._stem_surfaces = {l for stem in self._stems for l in _conjugate_stem(stem)}

        candidates = self._candidates_from_stem_surfaces(condition)
        prediction_scores = self._batch_prediction(candidates, min_eomi_score, self.min_num_of_features)

        eomi_surfaces = {eomi: score for eomi, score in prediction_scores.items() if score[1] >= min_eomi_score}

        if self.verbose:
            self._print(f"eomi lemmatization with {len(eomi_surfaces)} candidates")

        self.lrgraph.reset_lrgraph()
        lemmas = self._eomi_lemmatize(eomi_surfaces)
        lemmas = {
            eomi: score for eomi, score in lemmas.items() if (score[0] >= min_eomi_frequency) and (score[1] >= min_eomi_score)
        }

        if self.logpath:
            with open(self.logpath + "_eomi_prediction_score.log", "w", encoding="utf-8") as f:
                f.write("eomi frequency score\n")
                for word, score in sorted(prediction_scores.items(), key=lambda x: -x[1][1]):
                    f.write(f"{word} {score[0]} {score[1]}\n")

        if self.verbose:
            self._print(
                f"{len(lemmas)} eomis extracted with min frequency = {min_eomi_frequency}, min score = {min_eomi_score}"
            )

        self._check_covered_eojeols(lemmas)
        self._eomis = lemmas  # type: ignore[assignment]

        if reset_lrgraph:
            self.lrgraph.reset_lrgraph()

        del self._stem_surfaces

        return {eomi: EomiScore(score[0], score[1]) for eomi, score in lemmas.items()}

    def predict(
        self, r: str, min_eomi_score: float = 0.3, min_num_of_features: int = 5, debug: bool = False
    ) -> tuple[int, float]:
        features = self.lrgraph.get_l(r, -1)
        pos, neg, unk = self._predict(features, r)

        base = pos + neg
        score = 0 if base == 0 else (pos - neg) / base
        support = pos + unk if score >= min_eomi_score else neg + unk

        features_ = self._refine_features(features, r)
        n_features_ = len(features_)

        if debug:
            print(f"pos={pos}, neg={neg}, unk={unk}, n_features_={n_features_}")

        if n_features_ >= min_num_of_features:
            return support, score
        else:
            return (0, 0)

    def _predict(self, features: list, r: str) -> tuple[int, int, int]:
        pos, neg, unk = 0, 0, 0
        for l, freq in features:
            if (l + r) in self._nouns:
                continue
            if self._exist_longer_pos(l, r):
                continue
            if l in self._stem_surfaces:
                pos += freq
            elif self._is_a_noun_verb(l):
                pos += freq
            elif self._has_stem_at_last(l):
                unk += freq
            else:
                neg += freq
        return pos, neg, unk

    def _exist_longer_pos(self, l: str, r: str) -> bool:
        for i in range(1, len(r) + 1):
            if (l + r[:i]) in self._stem_surfaces:
                return True
        return False

    def _is_a_noun_verb(self, l: str) -> bool:
        return (l[0] in self._nouns) and (l[1:] in self._stem_surfaces)

    def _has_stem_at_last(self, l: str) -> bool:
        for i in range(1, len(l)):
            if l[-i:] in self._stem_surfaces:
                return True
        return False

    def _refine_features(self, features: list, r: str) -> list:
        return [(l, count) for l, count in features if (l in self._stem_surfaces) and (not self._exist_longer_pos(l, r))]

    def _candidates_from_stem_surfaces(self, condition: str | None = None) -> dict[str, int]:
        R_from_L: dict[str, int] = {}
        for l in self._stem_surfaces:
            for r, c in self.lrgraph.get_r(l, -1):
                if condition is None or r[-len(condition) :] == condition:
                    R_from_L[r] = R_from_L.get(r, 0) + c
        return R_from_L

    def _batch_prediction(
        self, eomi_candidates: dict[str, int], min_eomi_score: float = 0.3, min_num_of_features: int = 5
    ) -> dict[str, tuple[int, float]]:
        prediction_scores: dict[str, tuple[int, float]] = {}
        n = len(eomi_candidates)

        for i, r in enumerate(sorted(eomi_candidates, key=lambda x: -len(x))):
            if self.verbose and i % 10000 == 9999:
                percentage = f"{100 * (i + 1) / n:.2f}"
                self._print(f"  -- batch prediction {percentage} % of {n} words", replace=True, newline=False)

            support, score = self.predict(r, min_eomi_score, min_num_of_features)
            prediction_scores[r] = (support, score)

            if score >= min_eomi_score:
                for l, count in self.lrgraph.get_l(r, -1):
                    if (l in self._stem_surfaces) or self._is_a_noun_verb(l):
                        self.lrgraph.discount_eojeol(l + r, count)

        self.lrgraph.reset_lrgraph()

        if self.verbose:
            self._print(f"batch prediction was completed for {n} words", replace=True, newline=True)

        return prediction_scores

    def _eomi_lemmatize(self, eomis: dict[str, tuple[int, float]]) -> dict[str, tuple[int, float]]:
        def merge_score(freq0: int, score0: float, freq1: int, score1: float) -> tuple[int, float]:
            return (freq0 + freq1, (score0 * freq0 + score1 * freq1) / (freq0 + freq1))

        eomis_: dict[str, tuple[int, float]] = {}
        for eomi, (_, score0) in eomis.items():
            for stem_surface, count in self.lrgraph.get_l(eomi, -1):
                try:
                    for stem_, eomi_ in lemma_candidate(stem_surface, eomi):
                        if stem_ not in self._stems:
                            continue
                        eomis_[eomi_] = merge_score(count, score0, *eomis_.get(eomi_, (0, 0)))
                except Exception:
                    continue
        return eomis_

    def _check_covered_eojeols(self, eomis: dict) -> None:
        # TODO
        pass
