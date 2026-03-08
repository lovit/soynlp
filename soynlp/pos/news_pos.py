"""News domain POS extractor.

Note: The original code uses LRNounExtractor_v2 which is not yet available.
Currently uses LRNounExtractor as a fallback. Some features (extract_pos_feature,
extract_compound, etc.) may not be available.
"""

import logging

from soynlp.hangle import decompose
from soynlp.lemmatizer import lemma_candidate
from soynlp.noun import LRNounExtractor
from soynlp.predicator import Predicator, PredicatorExtractor
from soynlp.tokenizer import MaxScoreTokenizer
from soynlp.utils import CorpusLoader, EojeolCounter, LRGraph

from .adverb import load_default_adverbs, stem_to_adverb

logger = logging.getLogger(__name__)


class NewsPOSExtractor:
    def __init__(self, verbose: bool = True, ensure_normalized: bool = True, extract_eomi: bool = True):
        self._verbose = verbose
        self._ensure_normalized = ensure_normalized
        self._extract_eomi = extract_eomi

    def train_extract(
        self,
        sents: str | list[str] | CorpusLoader | EojeolCounter | LRGraph,
        min_num_of_noun_features: int = 1,
        max_frequency_when_noun_is_eojeol: int = 30,
        min_noun_score: float = 0.3,
        min_noun_frequency: int = 1,
        min_eojeol_frequency: int = 1,
        min_predicator_frequency: int = 1,
        min_num_of_eomi_features: int = 5,
        min_eomi_score: float = 0.3,
        min_eomi_frequency: int = 1,
        debug: bool = False,
    ) -> dict:
        self.train(
            sents,
            min_num_of_noun_features,
            max_frequency_when_noun_is_eojeol,
            min_noun_score,
            min_noun_frequency,
            min_eojeol_frequency,
            min_predicator_frequency,
            min_num_of_eomi_features,
            min_eomi_score,
            min_eomi_frequency,
        )
        return self.extract()

    def train(
        self,
        sents: str | list[str] | CorpusLoader | EojeolCounter | LRGraph,
        min_num_of_noun_features: int = 1,
        max_frequency_when_noun_is_eojeol: int = 30,
        min_noun_score: float = 0.3,
        min_noun_frequency: int = 1,
        min_eojeol_frequency: int = 1,
        min_predicator_frequency: int = 1,
        min_num_of_eomi_features: int = 5,
        min_eomi_score: float = 0.3,
        min_eomi_frequency: int = 1,
        debug: bool = False,
    ) -> None:
        self.nouns = self._train_noun_extractor(
            sents,
            min_num_of_noun_features,
            max_frequency_when_noun_is_eojeol,
            min_noun_score,
            min_noun_frequency,
            min_eojeol_frequency,
        )

        self.adjectives, self.verbs = self._train_predicator_extractor(
            sents,
            min_predicator_frequency,
            min_eojeol_frequency,
            min_num_of_eomi_features,
            min_eomi_score,
            min_eomi_frequency,
        )

        self.adjective_stems = self.predicator_extractor._adjective_stems
        self.verb_stems = self.predicator_extractor._verb_stems
        self.eomis = self.predicator_extractor._eomis
        self.josas = self.predicator_extractor._josas
        self.adverbs = load_default_adverbs()
        self.adverbs.update(stem_to_adverb(self.adjective_stems))
        self.adverbs.update(stem_to_adverb(self.verb_stems))

        self.eojeols = self.noun_extractor.lrgraph.to_EojeolCounter(reset_lrgraph=True)  # type: ignore[union-attr]
        self.eojeols = self.eojeols._counter

    def extract(self) -> dict:
        nouns, adjectives, verbs, adverbs, josas, irrecognized, confused_nouns = self._count_matched_patterns()

        adjectives = self._as_predicator(adjectives, self.adjectives, self.adjective_stems, self.eomis)
        verbs = self._as_predicator(verbs, self.verbs, self.verb_stems, self.eomis)

        return {
            "Noun": nouns,
            "Eomi": self.eomis,
            "Adjective": adjectives,
            "AdjectiveStem": self.adjective_stems,
            "Verb": verbs,
            "VerbStem": self.verb_stems,
            "Adverb": adverbs,
            "Josa": self.josas,
            "Irrecognizable": irrecognized,
            "ConfusedNouns": confused_nouns,
        }

    def _train_noun_extractor(
        self,
        sents: str | list[str] | CorpusLoader | EojeolCounter | LRGraph,
        min_num_of_features: int = 1,
        max_frequency_when_noun_is_eojeol: int = 30,
        min_noun_score: float = 0.3,
        min_noun_frequency: int = 1,
        min_eojeol_frequency: int = 1,
    ) -> dict:
        self.noun_extractor = LRNounExtractor(
            verbose=self._verbose,
        )

        nouns = self.noun_extractor.extract(
            sents,
            min_noun_score=min_noun_score,
            min_noun_frequency=min_noun_frequency,
        )
        return nouns

    def _train_predicator_extractor(
        self,
        sents: str | list[str] | CorpusLoader | EojeolCounter | LRGraph,
        min_predicator_frequency: int = 1,
        min_eojeol_frequency: int = 2,
        min_num_of_eomi_features: int = 5,
        min_eomi_score: float = 0.3,
        min_eomi_frequency: int = 1,
    ) -> tuple[dict, dict]:
        self.predicator_extractor = PredicatorExtractor(
            self.nouns,
            extract_eomi=self._extract_eomi,
            extract_stem=False,
            verbose=self._verbose,
        )

        adjectives, verbs = self.predicator_extractor.train_extract(
            sents,
            min_eojeol_frequency,
            100000,
            None,
            min_predicator_frequency,
            True,
            min_num_of_eomi_features,
            min_eomi_score,
            min_eomi_frequency,
        )
        return adjectives, verbs

    def _count_matched_patterns(self) -> tuple[dict, dict, dict, dict, dict, dict, dict]:
        eojeols = self.eojeols
        total_frequency = sum(eojeols.values())

        eojeols, nouns, adjectives, verbs, adverbs = self._match_word(eojeols)

        eojeols, nouns, adjectives, verbs, josas = self._match_noun_and_word(
            eojeols,
            nouns,
            adjectives,
            verbs,
            self.josas,
        )

        eojeols, adjectives, verbs = self._match_predicator_compounds(eojeols, adjectives, verbs)
        eojeols, adjectives, verbs = self._lemmatizing_predicators(eojeols, adjectives, verbs)

        eojeols, nouns, adjectives, verbs, josas = self._match_syllable_noun_and_r(
            eojeols,
            nouns,
            adjectives,
            verbs,
            josas,
        )

        eojeols = self._remove_irregular_words(eojeols)
        eojeols, nouns = self._match_compound_nouns(eojeols, nouns, adjectives, verbs, josas)

        confused_nouns = {
            word: count for word, count in nouns.items() if (word in adjectives) or (word in verbs) or (word in adverbs)
        }
        nouns = {word: count for word, count in nouns.items() if word not in confused_nouns}

        self._print_stats(total_frequency, nouns, adjectives, verbs, adverbs, josas, eojeols)

        return nouns, adjectives, verbs, adverbs, josas, eojeols, confused_nouns

    def _as_predicator(self, counter: dict, lemma_dict: dict, stems: set, eomis: set) -> dict:
        predicators = {}
        for word, count in counter.items():
            if word in lemma_dict:
                lemma = lemma_dict[word].lemma
            else:
                lemma = self._lemmatize(word, stems, eomis)
            if lemma is None:
                continue
            predicators[word] = Predicator(count, lemma)
        return predicators

    def _separate_lr(self, word: str, lset: set | dict, rset: set | dict, begin: int = 2) -> tuple[str, str] | None:
        for i in range(begin, len(word)):
            l, r = word[:i], word[i:]
            if (l in lset) and (r in rset):
                return l, r
        return None

    def _cumulate_counter(self, counter: dict, word_count) -> dict:
        for word, count in word_count:
            counter[word] = counter.get(word, 0) + count
        return counter

    def _remove_recognized(self, eojeols: dict, removals) -> dict:
        return {word: count for word, count in eojeols.items() if word not in removals}

    def _lemmatize(self, word: str, stems: set, eomis: set) -> list[tuple[str, str]] | None:
        def only_knowns(lemmas):
            return [lemma for lemma in lemmas if (lemma[0] in stems) and (lemma[1] in eomis)]

        for i in range(len(word) + 1, 0, -1):
            try:
                lemmas = lemma_candidate(word[:i], word[i:])
            except Exception:
                continue
            lemmas = only_knowns(lemmas)
            if lemmas:
                return lemmas
        return None

    def _parse_predicator_compounds(
        self, eojeols: dict, predicators: set, base: dict
    ) -> tuple[dict, set[str], dict[str, int]]:
        def check_suffix_prefix(stem: str, eomi: str) -> bool:
            l = decompose(stem[-1])  # type: ignore[assignment]
            r = decompose(eomi[0])  # type: ignore[assignment]
            jongcho_l = {"ㄹ", "ㅂ"}
            jongcho_r = {"ㄴ", "ㄹ", "ㅁ", "ㅂ"}
            if (l[2] in jongcho_l) and (r[0] in jongcho_r):  # type: ignore[index]
                return False
            if l[1] == "ㅡ" and l[2] == " " and r[0] == "ㅇ" and r[1] in ("ㅓ", "ㅏ"):  # type: ignore[index]
                return False
            return True

        stems: set[str] = set()
        predicator_compounds: dict = {}
        counter: dict[str, int] = {}
        for word, count in eojeols.items():
            lr = self._separate_lr(word, predicators, base)
            if lr is None:
                continue
            lemmas = base[lr[1]].lemma
            lemmas = {(lr[0] + stem, eomi) for stem, eomi in lemmas}
            lemmas = {(stem, eomi) for stem, eomi in lemmas if check_suffix_prefix(stem, eomi)}
            lemmas = {
                (stem, eomi) for stem, eomi in lemmas if stem not in self.verb_stems and stem not in self.adjective_stems
            }
            if not lemmas:
                continue
            predicator_compounds[word] = Predicator(count, lemmas)
            stems.update({stem for stem, _ in lemmas})
            counter[word] = count
        return predicator_compounds, stems, counter

    def _match_word(self, eojeols: dict) -> tuple[dict, dict, dict, dict, dict]:
        logger.info('[POS Extractor] matching "Noun, Adjective, Verb, and Adverb" from %d eojeols', len(eojeols))

        nouns = {word: count for word, count in eojeols.items() if word in self.nouns}
        adjectives = {word: count for word, count in eojeols.items() if word in self.adjectives}
        verbs = {word: count for word, count in eojeols.items() if word in self.verbs}
        adverbs = {word: count for word, count in eojeols.items() if word in self.adverbs}

        for removals in (nouns, adjectives, verbs, adverbs):
            eojeols = self._remove_recognized(eojeols, removals)

        return eojeols, nouns, adjectives, verbs, adverbs

    def _match_noun_and_word(
        self, eojeols: dict, nouns: dict, adjectives: dict, verbs: dict, josas: set
    ) -> tuple[dict, dict, dict, dict, dict]:
        logger.info('[POS Extractor] matching "Noun + Josa/Adjective/Verb" from %d eojeols', len(eojeols))

        def match_process(eojeols, nouns, rset, rcounter, removals):
            noun_r = [(self._separate_lr(word, nouns, rset), count) for word, count in eojeols.items()]
            noun_r = [(word, count) for word, count in noun_r if word is not None]
            nouns = self._cumulate_counter(nouns, [(word[0], count) for word, count in noun_r])
            rcounter = self._cumulate_counter(rcounter, [(word[1], count) for word, count in noun_r])
            removals.update({"".join(word) for word, _ in noun_r})
            return nouns, rcounter, removals

        nouns, josas_counter, removals = match_process(eojeols, nouns, josas, {}, set())
        nouns, adjectives, removals = match_process(eojeols, nouns, adjectives, adjectives, removals)
        nouns, verbs, removals = match_process(eojeols, nouns, verbs, verbs, removals)
        eojeols = self._remove_recognized(eojeols, removals)

        return eojeols, nouns, adjectives, verbs, josas_counter

    def _match_predicator_compounds(self, eojeols: dict, adjectives: dict, verbs: dict) -> tuple[dict, dict, dict]:
        logger.info('[POS Extractor] matching "Predicator + Adjective/Verb" from %d eojeols', len(eojeols))

        predicators = set(self.adjectives.keys()) | set(self.verbs.keys())
        before_adj, before_verb = len(self.adjectives), len(self.verbs)

        compounds, stems, counter = self._parse_predicator_compounds(eojeols, predicators, self.adjectives)
        self.adjectives.update(compounds)
        self.adjective_stems.update(stems)
        adjectives = self._cumulate_counter(adjectives, counter.items())

        compounds, stems, counter = self._parse_predicator_compounds(eojeols, predicators, self.verbs)
        self.verbs.update(compounds)
        self.verb_stems.update(stems)
        verbs = self._cumulate_counter(verbs, counter.items())

        removals = set(adjectives) | set(verbs)
        eojeols = self._remove_recognized(eojeols, removals)

        after_adj, after_verb = len(self.adjectives), len(self.verbs)
        logger.info("    adjective: %d -> %d, verb: %d -> %d", before_adj, after_adj, before_verb, after_verb)
        return eojeols, adjectives, verbs

    def _lemmatizing_predicators(self, eojeols: dict, adjectives: dict, verbs: dict) -> tuple[dict, dict, dict]:
        def lemmatize(eojeols, stems, eomis):
            predicator = {}
            n = len(eojeols)
            for i, (word, count) in enumerate(eojeols.items()):
                if i % 1000 == 0:
                    logger.info("    lemmatizing %d / %d", i, n)
                lemmas = self._lemmatize(word, stems, eomis)
                if lemmas is None:
                    continue
                predicator[word] = Predicator(count, lemmas)
            return predicator

        logger.info("    lemmatizing Adjective/Verb from %d eojeols", len(eojeols))

        new_adjectives = lemmatize(eojeols, self.adjective_stems, self.eomis)
        counter_adj = {word: count for word, count in eojeols.items() if word in new_adjectives}
        adjectives = self._cumulate_counter(adjectives, counter_adj.items())
        self.adjectives.update(new_adjectives)

        new_verbs = lemmatize(eojeols, self.verb_stems, self.eomis)
        counter_verb = {word: count for word, count in eojeols.items() if word in new_verbs}
        verbs = self._cumulate_counter(verbs, counter_verb.items())
        self.verbs.update(new_verbs)

        eojeols = self._remove_recognized(eojeols, counter_adj)
        eojeols = self._remove_recognized(eojeols, counter_verb)

        return eojeols, adjectives, verbs

    def _match_syllable_noun_and_r(
        self, eojeols: dict, nouns: dict, adjectives: dict, verbs: dict, josas: dict
    ) -> tuple[dict, dict, dict, dict, dict]:
        logger.info("[POS Extractor] parse 1 syllable Noun + Adj/Verb/Josa from %d eojeols", len(eojeols))

        def syllable_noun_and_r(rset, rcounter, removals):
            for word, count in eojeols.items():
                l, r = word[:1], word[1:]
                if r in rcounter:
                    nouns[l] = nouns.get(l, 0) + count
                    rcounter[r] = rcounter.get(r, 0) + count
                    removals.add(word)
            return removals

        removals = syllable_noun_and_r(self.adjectives, adjectives, set())
        removals = syllable_noun_and_r(self.verbs, verbs, removals)
        removals = syllable_noun_and_r(self.josas, josas, removals)
        eojeols = self._remove_recognized(eojeols, removals)

        return eojeols, nouns, adjectives, verbs, josas

    def _remove_irregular_words(self, eojeols: dict) -> dict:
        def remove(word):
            return (word in self.josas) or (word in self.eomis) or (len(word) == 1)

        return {word: count for word, count in eojeols.items() if not remove(word)}

    def _match_compound_nouns(
        self, eojeols: dict, nouns: dict, adjectives: dict, verbs: dict, josas: dict | set
    ) -> tuple[dict, dict]:
        logger.info("[POS Extractor] extract compound nouns from %d eojeols", len(eojeols))

        suffix = set(nouns) | set(adjectives) | set(verbs) | set(josas)

        compounds, removals = self._extract_compound_nouns(eojeols, nouns, suffix)
        nouns = self._cumulate_counter(nouns, compounds.items())
        eojeols = self._remove_recognized(eojeols, removals)

        return eojeols, nouns

    def _print_stats(
        self, total_frequency: int, nouns: dict, adjectives: dict, verbs: dict, adverbs: dict, josas: dict, eojeols: dict
    ) -> None:
        logger.info("[POS Extractor] ## statistics")

        def as_percent(dic):
            return 100 * sum(dic.values()) / total_frequency

        stats = [
            (len(nouns), as_percent(nouns), "Noun + [Josa/Predicator]"),
            (len(adjectives), as_percent(adjectives), "[Noun] + Adjective"),
            (len(verbs), as_percent(verbs), "[Noun] + Verb"),
            (len(josas), as_percent(josas), "[Noun] + Josa"),
            (len(adverbs), as_percent(adverbs), "Adverb"),
            (len(eojeols), as_percent(eojeols), "Irrecognizable"),
        ]
        for count, pct, label in stats:
            logger.info("[POS Extractor] (%d, %.3f %%) words in %s", count, pct, label)

    def _extract_compound_nouns(self, eojeols: dict, nouns: dict, suffix: set):
        def parse_compound(tokens):
            for token in tokens[:-1]:
                if token.score <= 0:
                    return None
            if len(tokens) >= 3 and (tokens[-1].word in suffix):
                return "".join(t.word for t in tokens[:-1])
            if tokens[-1].score > 0:
                return "".join(t.word for t in tokens)
            return None

        tokenizer = MaxScoreTokenizer(scores={noun: 1 for noun in nouns if len(noun) > 1})

        compounds: dict[str, int] = {}
        removals: set[str] = set()
        for word, count in eojeols.items():
            tokens = tokenizer.tokenize(word, flatten=False)[0]  # type: ignore[call-arg]
            noun = parse_compound(tokens)
            if noun is not None:
                compounds[noun] = compounds.get(noun, 0) + count
                removals.add(word)

        return compounds, removals
