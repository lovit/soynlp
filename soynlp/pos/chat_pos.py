"""Chat domain POS extractor (extends NewsPOSExtractor)."""

import logging
from collections import defaultdict

from soynlp.hangle import decompose, jaum_list
from soynlp.predicator import Predicator
from soynlp.utils.utils import installpath

from .news_pos import NewsPOSExtractor

logger = logging.getLogger(__name__)

# 어간 끝 음절이 이들 중 하나이면 어미 결합 시 불규칙 활용이 발생하여 후보에서 제외
# "업": '업다' 형 불규칙 (예: 없다), "닿": 받침 결합 예외, "땋": 받침 결합 예외
_IRREGULAR_STEM_ENDINGS = frozenset({"업", "닿", "땋"})


class ChatPOSExtractor(NewsPOSExtractor):
    def __init__(self, verbose: bool = True, ensure_normalized: bool = True, extract_eomi: bool = True):
        super().__init__(verbose, ensure_normalized, extract_eomi)

    def _count_matched_patterns(self) -> tuple[dict, dict, dict, dict, dict, dict, dict]:
        eojeols = self.eojeols
        total_frequency = sum(eojeols.values())

        path = f"{installpath}/postagger/dictionary/default/Josa/josa_chat.txt"
        with open(path, encoding="utf-8") as f:
            self.josas = {word.strip() for word in f}

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

        confused_nouns = {
            word: count for word, count in nouns.items() if (word in adjectives) or (word in verbs) or (word in adverbs)
        }
        confused_nouns.update(_find_noun_phrase(nouns, josas, adjectives, verbs))

        nouns = {word: count for word, count in nouns.items() if word not in confused_nouns}

        self._print_stats(total_frequency, nouns, adjectives, verbs, adverbs, josas, eojeols)

        return nouns, adjectives, verbs, adverbs, josas, eojeols, confused_nouns

    def _match_predicator_compounds(self, eojeols: dict, adjectives: dict, verbs: dict) -> tuple[dict, dict, dict]:
        logger.info('[POS Extractor] matching "Predicator + Adjective/Verb" from %d eojeols', len(eojeols))

        predicators = set(self.adjectives.keys()) | set(self.verbs.keys())
        before_adj, before_verb = len(self.adjectives), len(self.verbs)

        compounds, stems, counter = self._parse_predicator_compounds_chat(eojeols, predicators, self.adjectives)
        self.adjectives.update(compounds)
        self.adjective_stems.update(stems)
        wrong_eomis = _find_wrong_eomi(self.adjectives, compounds)
        adjectives = self._cumulate_counter(adjectives, counter.items())

        compounds, stems, counter = self._parse_predicator_compounds_chat(eojeols, predicators, self.verbs)
        self.verbs.update(compounds)
        self.verb_stems.update(stems)
        wrong_eomis.update(_find_wrong_eomi(self.verbs, compounds))
        verbs = self._cumulate_counter(verbs, counter.items())

        removals = set(adjectives) | set(verbs)
        eojeols = self._remove_recognized(eojeols, removals)

        self.eomis = {eomi for eomi in self.eomis if eomi not in wrong_eomis}

        after_adj, after_verb = len(self.adjectives), len(self.verbs)
        logger.info("[POS Extractor] adjective: %d -> %d, verb: %d -> %d", before_adj, after_adj, before_verb, after_verb)
        return eojeols, adjectives, verbs

    def _parse_predicator_compounds_chat(
        self, eojeols: dict, predicators: set, base: dict
    ) -> tuple[dict, set[str], dict[str, int]]:
        def check_suffix_prefix(stem: str, eomi: str) -> bool:
            # "업", "닿", "땋": 어간 끝 음절이 이 경우 어미 결합이 불규칙하여 제외
            if stem[-1] in _IRREGULAR_STEM_ENDINGS:
                return False
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
            if word in base:
                predicator = base[word]
                base_len = max(len(stem) for stem, _ in predicator.lemma)
                lemmas = {(stem, eomi) for stem, eomi in lemmas if len(stem) > base_len}
            if not lemmas:
                continue
            predicator_compounds[word] = Predicator(count, lemmas)
            stems.update({stem for stem, _ in lemmas})
            counter[word] = count

        wrong_stems = _find_wrong_stem(predicator_compounds)
        predicator_compounds = _delete_predicators_having_wrong_stem(predicator_compounds, wrong_stems)
        stems = {stem for stem in stems if stem not in wrong_stems}
        counter = {word: count for word, count in counter.items() if word in predicator_compounds}
        return predicator_compounds, stems, counter


def _find_wrong_stem(compounds: dict[str, Predicator]) -> set[str]:
    def jaum_begin_prop(stem: str) -> float:
        jaum_counter: dict[str, int] = defaultdict(int)
        for r, count in lrgraph.get(stem, {}).items():
            if not r or r[0] not in jaum_list:
                continue
            jaum_counter[r[0]] += count
        sum_ = sum(lrgraph.get(stem, {}).values())
        return 0 if sum_ == 0 else sum(jaum_counter.values()) / sum_

    lrgraph: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for predicator in compounds.values():
        count = predicator.frequency
        for stem, eomi in predicator.lemma:
            lrgraph[stem][eomi] += count
    return {stem for stem in lrgraph if jaum_begin_prop(stem) >= 0.9}


def _find_wrong_eomi(predicators: dict[str, Predicator], compounds: dict[str, Predicator]) -> set[str]:
    candidates: dict[str, int] = defaultdict(int)
    for word, predicator in predicators.items():
        if word not in compounds:
            for _, eomi in predicator.lemma:
                candidates[eomi] -= 1
        else:
            for _, eomi in predicator.lemma:
                candidates[eomi] += 1
    return {eomi for eomi, count in candidates.items() if count > 0}


def _delete_predicators_having_wrong_stem(predicators: dict[str, Predicator], wrong_stems: set[str]) -> dict[str, Predicator]:
    predicators_ = {}
    for word, predicator in predicators.items():
        lemmas = {(stem, eomi) for stem, eomi in predicator.lemma if stem not in wrong_stems}
        if lemmas:
            predicators_[word] = Predicator(predicator.frequency, lemmas)
    return predicators_


def _find_noun_phrase(
    nouns: dict[str, int], josas: set[str] | dict[str, int], adjectives: dict[str, int], verbs: dict[str, int]
) -> dict[str, int]:
    compounds: dict[str, int] = {}
    for noun, count in nouns.items():
        if len(noun) <= 2:
            continue
        for i in range(1, len(noun)):
            _, r = noun[:i], noun[i:]
            if (r in josas) or (r in adjectives) or (r in verbs):
                compounds[noun] = count
    return compounds
