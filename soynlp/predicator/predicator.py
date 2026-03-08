"""Korean predicator (adjective/verb) extraction using LR-Graph.

TERM DEFINITION:
    (l, r) : L and R position subwords
    stem : stem of Adjective and Verb
    ending : suffix, canonical form of ending
    stems : set of stem including Adjectives and Verbs
    composable_stems : stems that can be compounded with other prefix
        e.g. [] + 하다 : 덕질+하다, 냐옹+하다
    endings : set of ending
    pos_l_features : canonical form set of stems (L subwords)
    lrgraph : L-R graph including [stem + Ending], Adverbs,
              and maybe some Noun + Josa
"""

import logging
from collections import defaultdict
from dataclasses import dataclass

from soynlp.hangle import character_is_complete_korean
from soynlp.lemmatizer import _conjugate_stem, conjugate, lemma_candidate
from soynlp.normalizer import normalize_sent_for_lrgraph
from soynlp.utils import EojeolCounter, LRGraph, get_process_memory
from soynlp.utils.utils import installpath

from .adjective_vs_verb import conjugate_as_imperative, conjugate_as_pleasure, conjugate_as_present, rule_classify
from .eomi import EomiExtractor
from .stem import StemExtractor

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Predicator:
    frequency: int
    lemma: list[tuple[str, str]] | set[tuple[str, str]]


class PredicatorExtractor:
    def __init__(
        self,
        nouns: set[str] | dict,
        josas: set[str] | None = None,
        adjectives: set[str] | None = None,
        verbs: set[str] | None = None,
        eomis: set[str] | None = None,
        extract_eomi: bool = False,
        extract_stem: bool = False,
        verbose: bool = True,
        ensure_normalized: bool = False,
    ):
        if not josas:
            josas = self._load_default_josa()
        if (adjectives is None) or (verbs is None):
            adjectives, verbs = self._load_default_stems()
        if eomis is None:
            eomis = self._load_default_eomis()

        self._josas = josas
        self._adjective_stems = adjectives
        self._verb_stems = verbs
        self._stems = set(adjectives) | set(verbs)
        self._eomis = eomis
        self.verbose = verbose
        self.extract_eomi = extract_eomi
        self.extract_stem = extract_stem
        self.ensure_normalized = ensure_normalized

        self._stem_surfaces = self._transform_stem_as_surfaces()
        self.eojeol_counter: EojeolCounter | None = None
        self.lrgraph = None

        nouns = self._remove_stem_prefix(nouns)
        self._nouns = nouns
        self._eomis_ = set(self._eomis)

    def _load_default_josa(self) -> set[str]:
        path = f"{installpath}/postagger/dictionary/default/Josa/josa_chat.txt"
        with open(path, encoding="utf-8") as f:
            return {word.strip() for word in f}

    def _load_default_stems(self, min_frequency: int = 2) -> tuple[set[str], set[str]]:
        def load(path: str) -> set[str]:
            stems: set[str] = set()
            with open(path, encoding="utf-8") as f:
                for line in f:
                    word, frequency = line.split()
                    if int(frequency) < min_frequency:
                        continue
                    stems.add(word)
            return stems

        dirs = f"{installpath}/lemmatizer/dictionary/default/Stem"
        adjectives = load(f"{dirs}/Adjective.txt")
        verbs = load(f"{dirs}/Verb.txt")
        return adjectives, verbs

    def _load_default_eomis(self, min_frequency: int = 20) -> set[str]:
        path = f"{installpath}/lemmatizer/dictionary/default/Eomi/Eomi.txt"
        eomis: set[str] = set()
        with open(path, encoding="utf-8") as f:
            for line in f:
                word, frequency = line.split()
                if int(frequency) < min_frequency:
                    continue
                eomis.add(word)
        return eomis

    def _remove_stem_prefix(self, nouns: set[str] | dict) -> set[str]:
        def parse_noun(stem: str) -> str | None:
            if stem[:-1] in nouns:
                return stem[:-1]
            if stem[:-2] in nouns:
                return stem[:-2]
            return None

        removals = {parse_noun(stem) for stem in self._stems if parse_noun(stem) is not None}
        return {noun for noun in nouns if noun not in removals}

    def _transform_stem_as_surfaces(self) -> set[str]:
        surfaces: set[str] = set()
        for stem in self._stems:
            try:
                for l in _conjugate_stem(stem):
                    surfaces.add(l)
            except Exception as e:
                logger.warning("Exception stem = %s, %s", stem, e)
                continue
        return surfaces

    @property
    def is_trained(self) -> bool:
        return self.lrgraph is not None

    def train_extract(
        self,
        inputs,
        min_eojeol_frequency: int = 2,
        filtering_checkpoint: int = 100000,
        candidates=None,
        min_predicator_frequency: int = 1,
        reset_lrgraph: bool = True,
        min_num_of_features: int = 5,
        min_eomi_score: float = 0.3,
        min_eomi_frequency: int = 1,
        min_num_of_unique_R_char: int = 10,
        min_entropy_of_R_char: float = 0.5,
        min_entropy_of_R: float = 1.5,
        min_stem_score: float = 0.7,
        min_stem_frequency: int = 100,
    ):
        self.train(
            inputs,
            min_eojeol_frequency,
            filtering_checkpoint,
            min_num_of_features,
            min_eomi_score,
            min_eomi_frequency,
            min_num_of_unique_R_char,
            min_entropy_of_R_char,
            min_entropy_of_R,
            min_stem_score,
            min_stem_frequency,
        )
        return self.extract(candidates, min_predicator_frequency)

    def train(
        self,
        inputs,
        min_eojeol_frequency: int = 2,
        filtering_checkpoint: int = 100000,
        min_num_of_features: int = 5,
        min_eomi_score: float = 0.3,
        min_eomi_frequency: int = 1,
        min_num_of_unique_R_char: int = 10,
        min_entropy_of_R_char: float = 0.5,
        min_entropy_of_R: float = 1.5,
        min_stem_score: float = 0.7,
        min_stem_frequency: int = 100,
    ):
        if isinstance(inputs, LRGraph):
            self._train_with_eojeol_counter(inputs.to_EojeolCounter(), min_eojeol_frequency)  # type: ignore[union-attr]
        elif isinstance(inputs, EojeolCounter):
            self._train_with_eojeol_counter(inputs, min_eojeol_frequency)
        else:
            self._train_with_sentences(inputs, min_eojeol_frequency, filtering_checkpoint)

        if self.extract_eomi or self.extract_stem:
            lrgraph = self._prepare_predicator_lrgraph()

        if self.extract_eomi:
            self._extract_eomi(lrgraph, min_num_of_features, min_eomi_score, min_eomi_frequency)

        if self.extract_stem:
            if self.extract_eomi:
                lrgraph.reset_lrgraph()
            self._extract_stem(
                lrgraph,
                min_num_of_unique_R_char,
                min_entropy_of_R_char,
                min_entropy_of_R,
                min_stem_score,
                min_stem_frequency,
            )

        logger.info("has been trained")

    def _train_with_sentences(self, sentences, min_eojeol_frequency: int = 2, filtering_checkpoint: int = 100000):
        logger.info("counting eojeols ...")

        preprocess = (lambda x: x) if self.ensure_normalized else normalize_sent_for_lrgraph

        eojeol_counter = EojeolCounter(
            sentences,
            min_count=min_eojeol_frequency,
            verbose=self.verbose,
            preprocess=preprocess,
        )
        self._train_with_eojeol_counter(eojeol_counter)

    def _train_with_eojeol_counter(self, eojeol_counter: EojeolCounter, min_eojeol_frequency: int = 2):
        eojeol_counter._counter = {
            eojeol: count for eojeol, count in eojeol_counter._counter.items() if count >= min_eojeol_frequency
        }
        eojeol_counter._set_count_sum()
        self._num_of_eojeols = len(eojeol_counter)
        self._num_of_covered_eojeols = 0
        self._count_of_eojeols = eojeol_counter._count_sum
        self._count_of_covered_eojeols = 0

        self.eojeol_counter = eojeol_counter

        logger.info("#eojeols=%d, mem=%.3f Gb", self._num_of_eojeols, get_process_memory())

    def extract(self, candidates=None, min_predicator_frequency: int = 1):
        """Extract predicators. candidates is EojeolCounter or dict format."""
        self._num_of_covered_eojeols = 0
        predicators = self._extract_predicator(candidates, min_predicator_frequency)
        adjectives, verbs = self._separate_adjective_verb(predicators)
        return adjectives, verbs

    def _prepare_predicator_lrgraph(self):
        def contains_noun(eojeol: str) -> bool:
            n = len(eojeol)
            for e in range(2, n + 1):
                if eojeol[:e] in self._nouns:
                    return True
            return False

        eojeols = dict(self.eojeol_counter._counter)  # type: ignore[union-attr]
        eojeols = {eojeol: count for eojeol, count in eojeols.items() if len(eojeol) > 1 and not contains_noun(eojeol)}
        return EojeolCounter()._to_lrgraph(eojeols)

    def _extract_eomi(
        self,
        lrgraph,
        min_num_of_features: int = 5,
        min_eomi_score: float = 0.3,
        min_eomi_frequency: int = 1,
    ):
        eomi_extractor = EomiExtractor(
            lrgraph=lrgraph,
            stems=self._stems,
            nouns=self._nouns,
            min_num_of_features=min_num_of_features,
            verbose=self.verbose,
        )
        extracted_eomis = eomi_extractor.extract(
            condition=None,
            min_eomi_score=min_eomi_score,
            min_eomi_frequency=min_eomi_frequency,
            reset_lrgraph=True,
        )
        extracted_eomis = {eomi for eomi in extracted_eomis if eomi not in self._eomis}

        n_before = len(self._eomis)
        self._eomis.update(extracted_eomis)
        n_after = len(self._eomis)

        logger.info("eomis: %d -> %d", n_before, n_after)

    def _extract_stem(
        self,
        lrgraph,
        min_num_of_unique_R_char: int = 10,
        min_entropy_of_R_char: float = 0.5,
        min_entropy_of_R: float = 1.5,
        min_stem_score: float = 0.7,
        min_stem_frequency: int = 100,
    ):
        stem_extractor = StemExtractor(
            lrgraph=lrgraph,
            stems=self._stems,
            eomis=self._eomis,
            min_num_of_unique_R_char=min_num_of_unique_R_char,
            min_entropy_of_R_char=min_entropy_of_R_char,
            min_entropy_of_R=min_entropy_of_R,
        )
        extracted_stems = stem_extractor.extract(
            L_ignore=None,
            min_stem_score=min_stem_score,
            min_stem_frequency=min_stem_frequency,
        )
        extracted_stems = {stem for stem in extracted_stems if stem not in self._stems}

        n_before = len(self._stems)
        self._stems.update(extracted_stems)
        n_after = len(self._stems)

        logger.info("stems: %d -> %d", n_before, n_after)

    def _extract_predicator(self, eojeol_counter=None, min_frequency: int = 1) -> dict:
        def all_characters_are_complete_korean(s: str) -> bool:
            return all(character_is_complete_korean(c) for c in s)

        if (eojeol_counter is None) or (not eojeol_counter):
            eojeol_counter = {
                eojeol: count
                for eojeol, count in self.eojeol_counter.items()  # type: ignore[union-attr]
                if (count > min_frequency) and all_characters_are_complete_korean(eojeol)
            }

        lemmas = self._as_lemma_candidates(eojeol_counter)

        logger.info("%d predicators are extracted", len(lemmas))

        return lemmas

    def _as_lemma_candidates(self, eojeol_counter=None) -> dict:
        def is_noun_josa(eojeol: str) -> bool:
            for i in range(1, len(eojeol)):
                if (eojeol[:i] in self._nouns) and (eojeol[i:] in self._josas):
                    return True
            return False

        self._num_of_covered_eojeols = 0
        self._count_of_covered_eojeols = 0

        lemmas: dict = {}
        eomi_to_word_count: dict[str, list] = defaultdict(list)
        num_eojeol = len(eojeol_counter)  # type: ignore[arg-type]

        for i, (eojeol, count) in enumerate(eojeol_counter.items()):  # type: ignore[union-attr]
            if i % 5000 == 4999:
                logger.info("lemmatizing %d / %d words", i + 1, num_eojeol)
            if is_noun_josa(eojeol):
                continue

            n = len(eojeol)
            lemma_candidates: set[tuple[str, str]] = set()

            for j in range(1, n + 1):
                l, r = eojeol[:j], eojeol[j:]
                for stem, eomi in lemma_candidate(l, r):
                    if (stem in self._stems) and (eomi in self._eomis):
                        lemma_candidates.add((stem, eomi))

            lemma_candidates_ = set()
            for stem, eomi in lemma_candidates:
                if eojeol in conjugate(stem, eomi):
                    lemma_candidates_.add((stem, eomi))

            if lemma_candidates_:
                lemmas[eojeol] = Predicator(count, lemma_candidates_)
                for stem, eomi in lemma_candidates_:
                    eomi_to_word_count[eomi].append((eojeol, count))
                self._num_of_covered_eojeols += 1
                self._count_of_covered_eojeols += count

        lemmas = self._remove_wrong_eomis(lemmas, eomi_to_word_count)
        logger.info("lemma candidating was done")

        return lemmas

    def _remove_wrong_eomis(self, lemmas: dict, eomi_to_word_count: dict) -> dict:
        def noun_proportion(word_count: list) -> tuple[float, float]:
            sum_ = sum(1 for w, v in word_count if len(w) == 2)
            prop = sum(1 for w, v in word_count if (w in self._nouns) and (len(w) == 2))
            prop_len2 = 0.0
            if sum_ > 0:
                prop /= sum_
                prop_len2 = sum_ / sum(1 for w, v in word_count)
            return prop, prop_len2

        remove_eomis: set[str] = set()
        remove_morphs: dict[str, tuple[str, ...]] = {}

        for eomi, word_count in eomi_to_word_count.items():
            if len(eomi) >= 3:
                continue

            prop, prop_len2 = noun_proportion(word_count)
            if prop < 0.5:
                continue

            if prop_len2 == 1:
                remove_eomis.add(eomi)
                remove_words = tuple(word for word, _ in word_count if len(word) == 2)
            else:
                remove_words = tuple(
                    word
                    for word, _ in word_count
                    if (len(word) == 2 and word in self._nouns) or (len(word) == 3 and len(eomi) == 2)
                )
            remove_morphs[eomi] = remove_words

        self._eomis = {eomi for eomi in self._eomis if eomi not in remove_eomis}
        words = {word for words in remove_morphs.values() for word in words}
        logger.info("%d eomis are removed, %d words are modified.", len(remove_eomis), len(words))

        for eomi, words in remove_morphs.items():
            for word in words:
                if word not in lemmas:
                    continue
                predicator = lemmas[word]
                if len(predicator.lemma) == 1:
                    lemmas.pop(word)
                else:
                    lemmas[word] = Predicator(predicator.frequency, {lemma for lemma in predicator.lemma if lemma[1] != eomi})

        return lemmas

    def _separate_adjective_verb(self, predicators: dict) -> tuple[dict, dict]:
        adjectives: dict = {}
        verbs: dict = {}

        for word, predicator in predicators.items():
            frequency = predicator.frequency
            lemma_set = predicator.lemma
            adj: set = set()
            v: set = set()
            for lemma in lemma_set:
                if lemma[0] in self._verb_stems:
                    v.add(lemma)
                    continue
                if lemma[0] in self._adjective_stems:
                    adj.add(lemma)
                    continue

                answer = rule_classify(lemma[0])
                if answer == "Verb":
                    v.add(lemma)
                    continue
                if answer == "Adjective":
                    adj.add(lemma)
                    continue

                surfaces = conjugate_as_present(lemma[0])
                surfaces.update(conjugate_as_imperative(lemma[0]))
                surfaces.update(conjugate_as_pleasure(lemma[0]))
                surfaces = {surface for surface in surfaces if surface in predicators}

                if len(surfaces) <= 1:
                    adj.add(lemma)
                else:
                    v.add(lemma)

            if adj:
                adjectives[word] = Predicator(frequency, adj)
            if v:
                verbs[word] = Predicator(frequency, v)

        return adjectives, verbs
