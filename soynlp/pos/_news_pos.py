from collections import defaultdict

from soynlp.lemmatizer import _lemma_candidate
from soynlp.lemmatizer import conjugate
from soynlp.noun import LRNounExtractor_v2
from soynlp.predicator import Predicator
from soynlp.predicator import PredicatorExtractor
from soynlp.tokenizer import MaxScoreTokenizer
from soynlp.utils import LRGraph


class NewsPOSExtractor:

    def __init__(self, verbose=True, ensure_normalized=True, extract_eomi=True):
        self._verbose = verbose
        self._ensure_normalized = ensure_normalized
        self._extract_eomi = extract_eomi

    def extract(self, sents,
        # noun init
        min_num_of_noun_features=1, max_frequency_when_noun_is_eojeol=30,
        # noun extraction
        min_noun_score=0.3, min_noun_frequency=1, min_eojeol_frequency=1,
        # predicator train
        min_predicator_frequency=1,
        # Eomi extractor
        min_num_of_eomi_features=5, min_eomi_score=0.3, min_eomi_frequency=1,
        debug=False):

        nouns = self._extract_nouns(sents, min_num_of_noun_features,
            max_frequency_when_noun_is_eojeol, min_noun_score,
            min_noun_frequency, min_eojeol_frequency)

        adjectives, verbs = self._extract_predicators(
            nouns, sents, min_predicator_frequency, min_eojeol_frequency,
            min_num_of_eomi_features, min_eomi_score, min_eomi_frequency)

        adjective_stems = self.predicator_extractor._adjective_stems
        verb_stems = self.predicator_extractor._verb_stems
        eomis = self.predicator_extractor._eomis

        nouns_, confused_nouns, adjectives_, verbs_, not_covered = self._word_match_postprocessing(
            nouns, adjectives, adjective_stems, verbs, verb_stems, eomis)

        adjectives_ = self._value_as_Predicator(
            adjectives_, adjectives, adjective_stems, eomis)
        verbs_ = self._value_as_Predicator(
            verbs_, verbs, verb_stems, eomis)

        wordtags = {
            'Noun': nouns_,
            'Eomi': eomis,
            'Adjective': adjectives_,
            'AdjectiveStem': adjective_stems,
            'Verb': verbs_,
            'VerbStem': verb_stems
        }

        if debug:
            wordtags['ConfusedNoun'] = confused_nouns
            wordtags['NotCovered'] = not_covered

        return wordtags

    def _extract_nouns(self, sents,
        min_num_of_features=1, max_frequency_when_noun_is_eojeol=30, # noun init
        min_noun_score=0.3, min_noun_frequency=1, min_eojeol_frequency=1): # noun extraction

        self.noun_extractor = LRNounExtractor_v2(
            extract_pos_feature = False,
            extract_determiner = False,
            extract_compound = True,
            ensure_normalized = self._ensure_normalized,
            verbose = self._verbose,
            min_num_of_features = min_num_of_features,
            max_frequency_when_noun_is_eojeol = max_frequency_when_noun_is_eojeol
        )

        self.noun_extractor.train(sents, min_eojeol_frequency)
        nouns = self.noun_extractor.extract(min_noun_score,
            min_noun_frequency, reset_lrgraph=False)

        return nouns

    def _extract_predicators(self, nouns, sents,
        # predicator train
        min_predicator_frequency=1, min_eojeol_frequency=2,
        # Eomi extractor
        min_num_of_eomi_features=5, min_eomi_score=0.3, min_eomi_frequency=1):

        # prepare predicator_lrgraph
        predicator_lrgraph = LRGraph(self.noun_extractor.lrgraph._lr)
        noun_pos_features = {r for r in self.noun_extractor._pos_features}
        noun_pos_features.update({r for r in self.noun_extractor._common_features})

        # predicator extraction
        self.predicator_extractor = PredicatorExtractor(
            nouns,
            noun_pos_features,
            extract_eomi = self._extract_eomi,
            extract_stem = False,
            verbose = self._verbose
        )

        adjectives, verbs = self.predicator_extractor.train_extract(
            sents, min_eojeol_frequency, 100000, #filtering_checkpoint
            None, min_predicator_frequency, True, # filtering_checkpoint, lrgraph_reset
            min_num_of_eomi_features, min_eomi_score, min_eomi_frequency) # Eomi extractor

        return adjectives, verbs

    def _word_match_postprocessing(self, nouns, adjectives,
        adjective_stems, verbs, verb_stems, eomis):

        def as_percent(dic):
            return 100 * sum(dic.values()) / total_frequency

        def add_count(dic, word, count):
            dic[word] += count
            return True

        eojeol_counter = self.noun_extractor.lrgraph.to_EojeolCounter(
            reset_lrgraph=True)

        stems = set(adjective_stems)
        stems.update(set(verb_stems))

        total_frequency = 0
        nouns_ = defaultdict(int)
        confused_nouns = defaultdict(int)
        adjectives_ = defaultdict(int)
        verbs_ = defaultdict(int)
        not_covered = {}

        noun_pos = self.noun_extractor._pos_features
        noun_common = self.noun_extractor._common_features
        josas = self.predicator_extractor._josas
        compound_stems = set()

        for i, (eojeol, count) in enumerate(eojeol_counter.items()):

            if self._verbose and i % 1000 == 0:
                print('\r[POS Extractor] postprocessing {:.3f} % ...'.format(
                    100 * i / len(eojeol_counter)), end='')

            # cumulate total frequency
            total_frequency += count
            covered = False

            # check eojeol is noun + predicator compound
            noun = self._separate_predicator_from_noun(eojeol, nouns, adjectives, verbs)
            if noun is not None:
                covered = add_count(nouns_, noun, count)
                if (eojeol in nouns) and (eojeol != noun):
                    add_count(confused_nouns, eojeol, count)
                continue

            # check eojeol is noun + josa (or pos feature of noun)
            noun = self._separate_predicator_from_noun(eojeol, nouns, noun_pos, noun_common)
            if noun is not None:
                covered = add_count(nouns_, noun, count)
                if (eojeol in nouns) and (eojeol != noun):
                    add_count(confused_nouns, eojeol, count)
                continue

            # check eojeol is noun + predicator-suspect compound
            noun = self._separate_predicator_suspect_from_noun(eojeol, nouns, stems, eomis)
            if noun is not None:
                covered = add_count(nouns_, noun, count)
                if (eojeol in nouns) and (eojeol != noun):
                    add_count(confused_nouns, eojeol, count)
                continue

            # check whether eojeol is predicator or noun
            if self._word_match(eojeol, adjectives):
                covered = add_count(adjectives_, eojeol, count)
            if self._word_match(eojeol, verbs):
                covered = add_count(verbs_, eojeol, count)
            if eojeol in nouns:
                covered = add_count(nouns_, eojeol, count)

            # if matched, continue
            if covered:
                continue

            # check eojeol is stem + eomi
            lemmas = self._conjugatable(eojeol, stems, eomis)
            if lemmas is not None:
                covered = True
                stem_adj = {stem for stem, _ in lemmas if stem in adjective_stems}
                stem_v = {stem for stem, _ in lemmas if stem in verb_stems}
                if stem_adj:
                    adjectives_[eojeol] += count
                    adjectives[eojeol] = Predicator(
                        count,
                        {(stem, eomi) for stem, eomi in lemmas if stem in stem_adj})
                if stem_v:
                    verbs_[eojeol] += count
                    verbs[eojeol] = Predicator(
                        count,
                        {(stem, eomi) for stem, eomi in lemmas if stem in stem_v})
                continue

            l, r = eojeol[:1], eojeol[1:]
            if (r in verbs):
                covered = add_count(nouns_, l, count)
                covered = add_count(verbs_, r, count)
            elif (r in adjectives):
                covered = add_count(nouns_, l, count)
                covered = add_count(adjectives_, r, count)
            elif (r in josas):
                covered = add_count(nouns_, l, count)

            two_predicators = self._predicator_compound_to_lr(eojeol, adjectives, verbs)
            if two_predicators is not None:
                covered = True
                l, r, tag = two_predicators
                lemma = (adjectives[r] if tag == 'Adjective' else verbs[r]).lemma
                lemma = {(l+stem, eomi) for stem, eomi in lemma}
                predicator_compound = Predicator(count, lemma)
                if tag == 'Adjective':
                    adjectives[eojeol] = predicator_compound
                    adjectives_[eojeol] += count
                else:
                    verbs[eojeol] = predicator_compound
                    verbs_[eojeol] += count
                compound_stems.update({stem for stem, _ in lemma})

            if (eojeol in josas) or (eojeol in eomis) or (len(eojeol) == 1):
                covered = True

            if not covered:
                not_covered[eojeol] = count

        nouns_, not_covered = self._extract_compound_nouns(
            not_covered, nouns_, josas, adjectives_, verbs_)

        if self._verbose:
            print('\r[POS Extractor] postprocessing was done 100.00 %    ')
            print('[POS Extractor] ## statistics ##')
            templates = '[POS Extractor] {} ({:.3f} %): {}'
            print(templates.format(len(nouns_), as_percent(nouns_), 'Noun + [Josa/Predicator]'))
            print(templates.format(len(confused_nouns), as_percent(confused_nouns), 'Confused nouns'))
            print(templates.format(len(adjectives_), as_percent(adjectives_), '[Noun] + Adjective'))
            print(templates.format(len(verbs_), as_percent(verbs_), '[Noun] + Verb'))
            print(templates.format(len(not_covered), as_percent(not_covered), 'not covered'))

        return nouns_, confused_nouns, adjectives_, verbs_, not_covered

    def _word_match(self, word, references):
        if word in references:
            return word
        return None

    def _separate_predicator_from_noun(self, word, nouns, adjectives, verbs):
        # Ignore 1-syllable noun
        # for i in range(len(word), 1, -1):
        for i in range(2, len(word)):
            l, r = word[:i], word[i:]
            if not (l in nouns):
                continue
            if (r in adjectives) or (r in verbs):
                return l
        return None

    def _separate_predicator_suspect_from_noun(self, word, nouns, stems, eomis):
        for i in range(2, len(word)):
            l, r = word[:i], word[i:]
            if not (l in nouns):
                continue
            if self._conjugatable(r, stems, eomis) is not None:
                return l
        return None

    def _predicator_compound_to_lr(self, word, adjectives, verbs):
        for i in range(2, len(word)):
            l, r = word[:i], word[i:]
            if ((l in adjectives) or (l in verbs)):
                if r in verbs:
                    return (l, r, 'Verb')
                elif r in adjectives:
                    return (l, r, 'Adjective')
        return None

    def _conjugatable(self, word, stems, eomis):
        def only_knowns(lemmas):
            return [lemma for lemma in lemmas if
                    (lemma[0] in stems) and (lemma[1] in eomis)]

        for i in range(len(word) + 1, 0, -1):
            try:
                lemmas = _lemma_candidate(word[:i], word[i:])
            except:
                continue
            lemmas = only_knowns(lemmas)
            if lemmas:
                return lemmas
        return None

    def _extract_compound_nouns(self, words, nouns_, josa, adjectives, verbs):
        def parse_compound(tokens):
            def has_r(word):
                return (word in josa) or (word in adjectives) or (word in verbs)

            # format: (word, begin, end, score, length)
            for token in tokens[:-1]:
                if token[3] <= 0:
                    return None
            # Noun* + Josa
            if len(tokens) >= 3 and has_r(tokens[-1][0]):
                return tuple(t[0] for t in tokens[:-1])
            # all tokens are noun
            if tokens[-1][3] > 0:
                return tuple(t[0] for t in tokens)
            # else, not compound
            return None

        tokenizer = MaxScoreTokenizer(scores = {noun:1 for noun in nouns_ if len(noun) > 1})

        for word, count in words.items():
            tokens = tokenizer.tokenize(word, flatten=False)[0]
            compound_parts = parse_compound(tokens)
            if compound_parts:
                word = ''.join(compound_parts)
                nouns_[word] = nouns_.get(word, 0) + count
                if word in words:
                    words[word] = max(0, words.get(word, 0) - count)

        words = {word:count for word, count in words.items()
                 if (not (word in nouns_)) and (count > 0)}

        return nouns_, words

    def _value_as_Predicator(self, counter, lemma_dict, stem, eomis):
        predicators = {}
        for word, count in counter.items():
            if word in lemma_dict:
                lemma = lemma_dict[word].lemma
            else:
                lemma = self._conjugatable(word, stem, eomis)
            predicators[word] = Predicator(count, lemma)
        return predicators