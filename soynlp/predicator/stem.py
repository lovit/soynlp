import logging
import math
from typing import cast

from soynlp.lemmatizer import conjugate, lemma_candidate

logger = logging.getLogger(__name__)


class StemExtractor:
    def __init__(
        self,
        lrgraph,
        stems: set[str],
        eomis: set[str],
        min_num_of_unique_R_char: int = 10,
        min_entropy_of_R_char: float = 0.5,
        min_entropy_of_R: float = 1.5,
        verbose: bool = True,
    ) -> None:
        self.lrgraph = lrgraph
        self.stems = stems
        self.eomis = eomis
        self.min_num_of_unique_R_char = min_num_of_unique_R_char
        self.min_entropy_of_R_char = min_entropy_of_R_char
        self.min_entropy_of_R = min_entropy_of_R
        self.verbose = verbose

        self.L, self.R = self._conjugate_stem_and_eomi(lrgraph, stems, eomis)
        self._josa = {"거나", "게", "게는", "게도", "고", "고도", "고만", "는", "다", "다가", "서는", "아", "은"}

    def _conjugate_stem_and_eomi(self, lrgraph, stems: set[str], eomis: set[str]) -> tuple[set[str], set[str]]:
        eojeol_counter = lrgraph.to_EojeolCounter()

        stem_surfaces: set[str] = set()
        eomi_surfaces: set[str] = set()

        n_stems = len(stems)
        n_eomis = len(eomis)
        for i, stem in enumerate(stems):
            if i % 100 == 0:
                logger.info("Checking combination of %d / %d stems + %d eomis", i, n_stems, n_eomis)

            stem_len = len(stem)
            for eomi in eomis:
                try:
                    for word in conjugate(stem, eomi):
                        if (eojeol_counter[word] == 0) or (len(word) <= stem_len):
                            continue
                        l, r = word[:stem_len], word[stem_len:]
                        stem_surfaces.add(l)
                        eomi_surfaces.add(r)
                except Exception:
                    continue

        logger.info("Initializing was done with %d stems and %d eomis", len(stems), len(eomis))

        del eojeol_counter
        return stem_surfaces, eomi_surfaces

    def extract(
        self,
        L_ignore: set[str] | None = None,
        min_stem_score: float = 0.7,
        min_stem_frequency: int = 100,
    ) -> dict[str, tuple[float, float]]:
        if L_ignore is None:
            L_ignore = set()

        candidates: dict[str, int] = {}
        for r in self.R:
            for l, count in self.lrgraph.get_l(r, -1):
                if (l in self.L) or (l in L_ignore):
                    continue
                candidates[l] = candidates.get(l, 0) + count

        candidates = {l: count for l, count in candidates.items() if count >= min_stem_frequency}

        logger.info("batch prediction for %d candidates", len(candidates))

        stem_surfaces = self._batch_prediction(candidates, min_stem_score, min_stem_frequency)
        self.stem_surfaces, self.removals = self._post_processing(stem_surfaces)
        self.stems = self._to_stem(self.stem_surfaces)

        logger.info(
            "%d stems, %d surfacial stems, %d removals",
            len(self.stems),
            len(self.stem_surfaces),
            len(self.removals),
        )

        return self.stems

    def _batch_prediction(
        self,
        candidates: dict[str, int],
        min_stem_score: float,
        min_frequency: int,
    ) -> dict[str, tuple[float, int]]:
        extracted: dict[str, tuple[float, int] | None] = {l: None for l in self.L}

        for l in sorted(candidates, key=lambda x: -len(x)):
            if (l in self.L) or (l in self.R) or (len(l) == 1) or (l[-1] == "다") or (l in extracted):
                continue

            score, freq = self.predict(l, min_stem_score, min_frequency)

            if (score < min_stem_score) or (freq < min_frequency):
                continue

            extracted[l] = (score, freq)

        return cast(dict[str, tuple[float, int]], {l: score for l, score in extracted.items() if l not in self.L})

    def predict(self, l: str, min_stem_score: float = 0.7, min_frequency: int = 1, debug: bool = False) -> tuple[float, int]:
        features = self.lrgraph.get_r(l, -1)
        char_count = self._count_first_chars(features)

        unique_of_char = len(char_count)
        entropy_of_char = self._entropy(tuple(self._select_pos_features(l, features).values()))

        pos, neg, unk = self._predict(l, features)
        score = (pos - neg) / (pos + neg) if (pos + neg) > 0 else 0
        freq = pos if score >= min_stem_score else neg + unk

        logger.info(
            "pos=%d, neg=%d, unk=%d, n_features_=%d, n_char=%d, entropy_r=%s",
            pos,
            neg,
            unk,
            len(features),
            unique_of_char,
            entropy_of_char,
        )

        if (unique_of_char < self.min_num_of_unique_R_char) or (entropy_of_char < self.min_entropy_of_R_char):
            return (0, 0)

        if freq < min_frequency:
            return (0, freq)
        else:
            return (score, freq)

    def _predict(self, l: str, features: list) -> tuple[int, int, int]:
        pos, neg, unk = 0, 0, 0
        for r, freq in features:
            if r in self._josa:
                continue
            if not r:
                neg += freq
            elif r in self.R:
                pos += freq
            elif self._r_is_predicator(r):
                neg += freq
            elif self._exist_longer_eomi(l, r):
                neg += freq
            else:
                unk += freq
        return pos, neg, unk

    def _count_first_chars(self, features: list) -> dict[str, int]:
        counter: dict[str, int] = {}
        for r, count in features:
            if r:
                counter[r[0]] = counter.get(r[0], 0) + count
        return counter

    def _entropy(self, counts: tuple[int, ...]) -> float:
        if len(counts) <= 1:
            return 0
        sum_ = sum(counts)
        return -1 * sum(p / sum_ * math.log(p / sum_) for p in counts)

    def _select_pos_features(self, l: str, features: list) -> dict[str, int]:
        def is_pos(r: str) -> bool:
            if (r in self._josa) or self._r_is_predicator(r) or self._exist_longer_eomi(l, r):
                return False
            return r in self.R

        return {r: count for r, count in features if is_pos(r)}

    def _r_is_predicator(self, r: str) -> bool:
        n = len(r)
        for i in range(1, n):
            if (r[:i] in self.L) and (r[i:] in self.R):
                return True
        return False

    def _exist_longer_eomi(self, l: str, r: str) -> bool:
        for i in range(1, len(l) + 1):
            if (l[-i:] + r) in self.R:
                return True
        return False

    def _post_processing(self, extracted: dict[str, tuple[float, int]]) -> tuple[dict[str, tuple[float, int]], set[str]]:
        def is_stem_and_eomi(l: str) -> bool:
            n = len(l)
            for i in range(1, n):
                if (l[:i] not in self.L) and (l[:i] not in extracted):
                    continue
                for j in range(i + 1, n + 1):
                    if l[i:j] in self.R:
                        return True
            return False

        def exist_subword(l: str) -> bool:
            for i in range(2, len(l)):
                if l[:i] in extracted:
                    return True
            return False

        removals = set()
        for l in sorted(extracted, key=lambda x: len(x)):
            if is_stem_and_eomi(l) or exist_subword(l):
                removals.add(l)
        extracted = {l: score for l, score in extracted.items() if l not in removals}
        return extracted, removals

    def _to_stem(self, surfaces: dict[str, tuple[float, int]]) -> dict[str, tuple[float, float]]:
        def merge_score(freq0: float, score0: float, freq1: float, score1: float) -> tuple[float, float]:
            return (freq0 + freq1, (score0 * freq0 + score1 * freq1) / (freq0 + freq1))

        stems: dict[str, tuple[float, float]] = {}
        for l, (freq0, score0) in surfaces.items():
            for r, count in self.lrgraph.get_r(l, -1):
                try:
                    for stem, eomi in lemma_candidate(l, r):
                        if eomi in self.eomis:
                            continue
                        stems[stem] = merge_score(freq0, score0, *stems.get(stem, (0, 0)))
                except Exception:
                    continue
        return stems
