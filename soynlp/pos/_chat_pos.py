from collections import defaultdict

from soynlp.hangle import decompose
from soynlp.hangle._hangle import jaum_list
from soynlp.lemmatizer import conjugate
from soynlp.noun import LRNounExtractor_v2
from soynlp.noun import NounScore
from soynlp.pos import load_default_adverbs
from soynlp.pos import stem_to_adverb
from soynlp.predicator import Predicator
from soynlp.predicator import PredicatorExtractor
from soynlp.tokenizer import MaxScoreTokenizer
from soynlp.utils import LRGraph
from soynlp.utils.utils import installpath
from ._news_pos import NewsPOSExtractor


class ChatPOSExtractor(NewsPOSExtractor):

    def __init__(self, verbose=True, ensure_normalized=True, extract_eomi=True):
        super().__init__(verbose, ensure_normalized, extract_eomi)

    def _count_matched_patterns(self):
        eojeols = self.eojeols
        total_frequency = sum(eojeols.values())

        path = '%s/postagger/dictionary/default/Josa/josa_chat.txt' % installpath
        with open(path, encoding='utf-8') as f:
            self.josas = {word.strip() for word in f}

        eojeols, nouns, adjectives, verbs, adverbs = self._match_word(eojeols)

        eojeols, nouns, adjectives, verbs, josas = self._match_noun_and_word(
            eojeols, nouns, adjectives, verbs, self.josas)

        eojeols, adjectives, verbs = self._match_predicator_compounds(
            eojeols, adjectives, verbs)

        eojeols, adjectives, verbs = self._lemmatizing_predicators(
            eojeols, adjectives, verbs)

        eojeols, nouns, adjectives, verbs, josas = self._match_syllable_noun_and_r(
            eojeols, nouns, adjectives, verbs, josas)

        eojeols = self._remove_irregular_words(eojeols)

        #eojeols, nouns = self._match_compound_nouns(
        #    eojeols, nouns, adjectives, verbs, josas)

        confused_nouns = {word:count for word, count in nouns.items()
            if (word in adjectives) or (word in verbs) or (word in adverbs)}
        confused_nouns.update(find_noun_phrase(nouns, josas, adjectives, verbs))

        nouns = {word:count for word, count in nouns.items()
                 if not (word in confused_nouns)}

        if self._verbose:
            self._print_stats(total_frequency, nouns,
                adjectives, verbs, adverbs, josas, eojeols)

        return nouns, adjectives, verbs, adverbs, josas, eojeols, confused_nouns

    def _match_predicator_compounds(self, eojeols, adjectives, verbs):
        if self._verbose:
            print('[POS Extractor] matching "Predicator + Adjective/Verb" from {} eojeols'.format(len(eojeols)))

        predicators = set(self.adjectives.keys())
        predicators.update(set(self.verbs.keys()))
        before_adj, before_verb = len(self.adjectives), len(self.verbs)

        # adjective compounds
        compounds, stems, counter = self._parse_predicator_compounds(
            eojeols, predicators, self.adjectives)
        self.adjectives.update(compounds)
        self.adjective_stems.update(stems)
        wrong_eomis = find_wrong_eomi(self.adjectives, compounds) # additional code
        adjectives = self._cumulate_counter(adjectives, counter.items())

        ## verb compounds
        compounds, stems, counter = self._parse_predicator_compounds(
            eojeols, predicators, self.verbs)
        self.verbs.update(compounds)
        self.verb_stems.update(stems)
        wrong_eomis.update(find_wrong_eomi(self.verbs, compounds)) # additional code
        verbs = self._cumulate_counter(verbs, counter.items())

        ## remove matched eojeols
        wrong_stems = {word for word in adjectives}
        wrong_stems.update({word for word in verbs})
        eojeols = self._remove_recognized(eojeols, wrong_stems)

        # additional code
        self.eomis = {eomi for eomi in self.eomis if not (eomi in wrong_eomis)}

        if self._verbose:
            after_adj, after_verb = len(self.adjectives), len(self.verbs)
            print('[POS Extractor] adjective: %d -> %d, verb: %d -> %d' % (
                before_adj, after_adj, before_verb, after_verb))
        return eojeols, adjectives, verbs

    def _parse_predicator_compounds(self, eojeols, predicators, base):
        def check_suffix_prefix(stem, eomi):
            if stem[-1] == '업' or stem[-1] == '닿' or stem[-1] == '땋':
                return False
            l = decompose(stem[-1])
            r = decompose(eomi[0])
            jongcho_l = set('ㄹㅂ')
            jongcho_r = set('ㄴㄹㅁㅂ')
            if (l[2] in jongcho_l) and (r[0] in jongcho_r):
                return False
            if (l[1] == 'ㅡ' and l[2] == ' ' and r[0] == 'ㅇ' and (r[1] == 'ㅓ' or r[1] == 'ㅏ')):
                return False
            return True

        stems = set()
        predicator_compounds = {}
        counter = {}
        for word, count in eojeols.items():
            lr = self._separate_lr(word, predicators, base)
            if lr is None:
                continue
            lemmas = base[lr[1]].lemma
            lemmas = {(lr[0]+stem, eomi) for stem, eomi in lemmas}
            lemmas = {(stem, eomi) for stem, eomi in lemmas if check_suffix_prefix(stem, eomi)}
            lemmas = {(stem, eomi) for stem, eomi in lemmas
                if not (stem in self.verb_stems) and not (stem in self.adjective_stems)}
            if word in base:
                predicator = base[word]
                base_len = max(len(stem) for stem, _ in predicator.lemma)
                lemmas = {(stem, eomi) for stem, eomi in lemmas if len(stem) > base_len}
            if not lemmas:
                continue
            predicator_compounds[word] = Predicator(count, lemmas)
            stems.update({stem for stem, _ in lemmas})
            counter[word] = count

        wrong_stems = find_wrong_stem(predicator_compounds)
        predicator_compounds = delete_predicators_having_wrong_stem(
            predicator_compounds, wrong_stems)
        stems = {stem for stem in wrong_stems if not (stem in wrong_stems)}
        counter = {word:count for word, count in counter.items() if word in predicator_compounds}
        return predicator_compounds, stems, counter

def find_wrong_stem(compounds):
    def jaum_begin_prop(stem):
        jaum_counter = defaultdict(int)
        for r, count in lrgraph.get(stem, {}).items():
            if not r or not (r[0] in jaum_list):
                continue
            jaum_counter[r[0]] += count
        sum_ = sum(lrgraph.get(stem, {}).values())
        return 0 if sum_ == 0 else sum(jaum_counter.values()) / sum_

    lrgraph = defaultdict(lambda: defaultdict(int))
    for predicator in compounds.values():
        count = predicator.frequency
        for stem, eomi in predicator.lemma:
            lrgraph[stem][eomi] += count
    return {stem for stem in lrgraph if jaum_begin_prop(stem) >= 0.9}

def find_wrong_eomi(predicators, compounds):
    candidates = defaultdict(int)
    for word, predicator in predicators.items():
        if not (word in compounds):
            continue
        for _, eomi in predicator.lemma:
            candidates[eomi] += 1
    for word, predicator in predicators.items():
        if (word in compounds):
            continue
        for _, eomi in predicator.lemma:
            candidates[eomi] -= 1
    removals = {eomi for eomi, count in candidates.items() if count > 0}
    return removals

def delete_predicators_having_wrong_stem(predicators, wrong_stems):
    predicators_ = {}
    for word, predicator in predicators.items():
        frequency = predicator.frequency
        lemmas = predicator.lemma
        lemmas = {(stem, eomi) for stem, eomi in lemmas if not (stem in wrong_stems)}
        if not lemmas:
            continue
        predicators_[word] = Predicator(frequency, lemmas)
    return predicators_

def find_noun_phrase(nouns, josas, adjectives, verbs):
    compounds = {}
    for noun, count in nouns.items():
        if len(noun) <= 2:
            continue
        for i in range(1, len(noun)):
            l, r = noun[:i], noun[i:]
            if (r in josas) or (r in adjectives) or (r in verbs):
                compounds[noun] = count
    return compounds