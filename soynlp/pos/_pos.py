from collections import defaultdict

from soynlp.lemmatizer import _lemma_candidate
from soynlp.lemmatizer import conjugate
from soynlp.noun import LRNounExtractor_v2
from soynlp.predicator import PredicatorExtractor
from soynlp.tokenizer import MaxScoreTokenizer
from soynlp.utils import LRGraph


class POSExtractor:

    def __init__(self, verbose=True, extract_noun_pos_feature=True,
        extract_determiner=True, ensure_normalized=True, extract_eomi=True,
        extract_stem=False):

        self._verbose = verbose
        # noun extraction
        self._extract_noun_pos_feature = extract_noun_pos_feature
        self._extract_determiner = extract_determiner
        self._ensure_normalized = ensure_normalized
        # predicator extraction
        self._extract_eomi = extract_eomi
        self._extract_stem = extract_stem

    def extract(self, sents,
        # noun init
        min_num_of_noun_features=1, max_frequency_when_noun_is_eojeol=30,
        # noun extraction
        min_noun_score=0.3, min_noun_frequency=1, min_eojeol_frequency=1,
        # noun domain pos features
        ignore_features=None, min_noun_frequency_in_pos_extraction=100,
        min_pos_score=0.3, min_pos_feature_frequency=1000,
        min_num_of_unique_lastchar=4, min_entropy_of_lastchar=0.5,
        min_noun_entropy=1.5,
        # predicator train
        min_predicator_frequency=1,
        # Eomi extractor
        min_num_of_eomi_features=5, min_eomi_score=0.3, min_eomi_frequency=1,
        # Stem extractor
        min_num_of_unique_R_char=10, min_entropy_of_R_char=0.5,
        min_entropy_of_R=1.5, min_stem_score=0.7, min_stem_frequency=100, debug=False):

        nouns = self._extract_nouns(sents, min_num_of_noun_features,
            max_frequency_when_noun_is_eojeol, min_noun_score,
            min_noun_frequency, min_eojeol_frequency,
            # noun domain pos features
            ignore_features, min_noun_frequency_in_pos_extraction,
            min_pos_score, min_pos_feature_frequency,
            min_num_of_unique_lastchar, min_entropy_of_lastchar,
            min_noun_entropy)

        adjectives, verbs = self._extract_predicators(
            nouns, sents, min_predicator_frequency,
            min_eojeol_frequency, min_num_of_eomi_features, min_eomi_score,
            min_eomi_frequency, min_num_of_unique_R_char, min_entropy_of_R_char,
            min_entropy_of_R, min_stem_score, min_stem_frequency)

        adjective_stems = self.predicator_extractor._adjective_stems
        verb_stems = self.predicator_extractor._verb_stems
        eomis = self.predicator_extractor._eomis

        nouns_, confused_nouns, adjectives_, verbs_, not_covered = self._word_match_postprocessing(
            nouns, adjectives, adjective_stems, verbs, verb_stems, eomis)

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
        # noun init
        min_num_of_features=1, max_frequency_when_noun_is_eojeol=30,
        # noun extraction
        min_noun_score=0.3, min_noun_frequency=1, min_eojeol_frequency=1,
        # noun domain pos features
        ignore_features=None, min_noun_frequency_in_pos_extraction=100,
        min_pos_score=0.3, min_pos_feature_frequency=1000,
        min_num_of_unique_lastchar=4, min_entropy_of_lastchar=0.5,
        min_noun_entropy=1.5):

        self.noun_extractor = LRNounExtractor_v2(
            extract_pos_feature = False,
            extract_determiner = self._extract_determiner,
            extract_compound = False,
            ensure_normalized = self._ensure_normalized,
            verbose = self._verbose,
            min_num_of_features = min_num_of_features,
            max_frequency_when_noun_is_eojeol = max_frequency_when_noun_is_eojeol
        )

        self.noun_extractor.train(sents, min_eojeol_frequency)

        if self._extract_noun_pos_feature:
            self.noun_extractor.extract_domain_pos_features(None, # noun candidates
                ignore_features, True, # append_extracted_features
                min_noun_score, min_noun_frequency_in_pos_extraction, min_pos_score,
                min_pos_feature_frequency, min_num_of_unique_lastchar,
                min_entropy_of_lastchar, min_noun_entropy)

        nouns = self.noun_extractor.extract(min_noun_score,
            min_noun_frequency, reset_lrgraph=False)

        return nouns

    def _extract_predicators(self, nouns, sents,
        # predicator train
        min_predicator_frequency=1, min_eojeol_frequency=2,
        # Eomi extractor
        min_num_of_eomi_features=5, min_eomi_score=0.3, min_eomi_frequency=1,
        # Stem extractor
        min_num_of_unique_R_char=10, min_entropy_of_R_char=0.5,
        min_entropy_of_R=1.5, min_stem_score=0.7, min_stem_frequency=100):

        # prepare predicator_lrgraph
        predicator_lrgraph = LRGraph(self.noun_extractor.lrgraph._lr)
        noun_pos_features = {r for r in self.noun_extractor._pos_features}
        noun_pos_features.update({r for r in self.noun_extractor._common_features})

        # predicator extraction
        self.predicator_extractor = PredicatorExtractor(
            nouns,
            noun_pos_features,
            extract_eomi = self._extract_eomi,
            extract_stem = self._extract_stem,
            verbose = self._verbose
        )

        adjectives, verbs = self.predicator_extractor.train_extract(
            sents, min_eojeol_frequency, 100000, #filtering_checkpoint
            None, min_predicator_frequency, True, # filtering_checkpoint, lrgraph_reset
            # Eomi extractor
            min_num_of_eomi_features, min_eomi_score, min_eomi_frequency,
            # Stem extractor
            min_num_of_unique_R_char, min_entropy_of_R_char,
            min_entropy_of_R, min_stem_score, min_stem_frequency)

        return adjectives, verbs

    def _word_match_postprocessing(self, nouns, adjectives,
        adjective_stems, verbs, verb_stems, eomis):

        def as_percent(value, total):
            return 100 * value / total

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

        compound_stems = set()

        for i, (eojeol, count) in enumerate(eojeol_counter.items()):

            if self._verbose and i % 1000 == 0:
                print('\r[POS Extractor] postprocessing {:.3f} % ...'.format(
                    as_percent(i, len(eojeol_counter))), end='')

            # cumulate total frequency
            total_frequency += count
            covered = False

            # check eojeol is noun + predicator compound
            noun = self._separate_predicator_from_noun(
                eojeol, nouns, adjectives, verbs)
            if noun is not None:
                covered = True
                nouns_[noun] += count
                # debug noun extraction results
                if eojeol in nouns:
                    confused_nouns[eojeol] += count
                continue

            # check eojeol is noun + josa (or pos feature of noun)
            noun = self._separate_predicator_from_noun(
                eojeol, nouns,
                self.noun_extractor._pos_features,
                self.noun_extractor._common_features)
            if noun is not None:
                covered = True
                nouns_[noun] += count
                continue

            # check eojeol is noun + predicator-suspect compound
            noun = self._separate_predicator_suspect_from_noun(
                eojeol, nouns, stems, eomis)
            if noun is not None:
                covered = True
                nouns_[noun] += count
                # debug noun extraction results
                if eojeol in nouns:
                    confused_nouns[eojeol] += count
                continue

            # check whether eojeol is predicator or noun
            if self._word_match(eojeol, adjectives):
                covered = True
                adjectives_[eojeol] += count
            if self._word_match(eojeol, verbs):
                covered = True
                verbs_[eojeol] += count
            if eojeol in nouns:
                covered = True
                nouns_[eojeol] += count

            # check eojeol is stem + eomi
            lemmas = self._conjugatable(eojeol, stems, eomis)
            if lemmas is not None:
                covered = True
                stem_adjs = {stem for stem, _ in lemmas if stem in adjective_stems}
                stem_v = {stem for stem, _ in lemmas if stem in verb_stems}
                if stem_adjs:
                    adjectives_[eojeol] += count
                if stem_v:
                    verbs_[eojeol] += count
                if eojeol in nouns:
                    confused_nouns[eojeol] = count
                continue

            l, r = eojeol[1:], eojeol[:1]
            if (r in verbs):
                nouns_[l] += count
                verbs_[r] += count
                covered = True
            elif (r in adjectives):
                nouns_[l] += count
                adjectives_[r] += count
                covered = True

            two_predicators = self._separate_two_predicator(eojeol, adjectives, verbs)
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

            if not covered:
                not_covered[eojeol] = count

        josas = self.predicator_extractor._josas
        nouns, compound_nouns, not_covered = self._extract_compound_nouns(
            not_covered, nouns, josas, adjectives_, verbs_)

        if self._verbose:
            print('\r[POS Extractor] postprocessing was done 100.00 %    ')
            print('[POS Extractor] ## statistics ##')
            print('[POS Extractor] {} ({:.3f} %): Noun + [Josa/Predicator]'.format(
                len(nouns_), as_percent(sum(nouns_.values()), total_frequency)))
            print('[POS Extractor] {} ({:.3f} %): Confused nouns'.format(
                len(confused_nouns), as_percent(sum(confused_nouns.values()), total_frequency)))
            print('[POS Extractor] {} ({:.3f} %): Adjective'.format(
                len(adjectives_), as_percent(sum(adjectives_.values()), total_frequency)))
            print('[POS Extractor] {} ({:.3f} %) Verb'.format(
                len(verbs_), as_percent(sum(verbs_.values()), total_frequency)))
            print('[POS Extractor] {} ({:.3f} %) not covered'.format(
                len(not_covered), as_percent(sum(not_covered.values()), total_frequency)))
            print('[POS Extractor] {} ({:.3f} %) compound nouns'.format(
                len(compound_nouns), as_percent(sum(compound_nouns.values()), total_frequency)))

        return nouns_, confused_nouns, adjectives_, verbs_, not_covered

    def _word_match(self, word, references):
        if word in references:
            return word
        return None

    def _separate_predicator_from_noun(self, word, nouns, adjectives, verbs):

        # Ignore 1-syllable noun
#             for i in range(len(word), 1, -1):
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

    def _separate_two_predicator(self, word, adjectives, verbs):
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

    def _extract_compound_nouns(self, words, nouns, josa, adjectives, verbs):
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

        tokenizer = MaxScoreTokenizer(scores = {noun:1 for noun in nouns})

        compounds = {}
        removals = set()
        for word, count in words.items():
            tokens = tokenizer.tokenize(word, flatten=False)[0]
            compound_parts = parse_compound(tokens, )
            if compound_parts:
                removals.add(word)
                word = ''.join(compound_parts)
                if word in nouns:
                    nouns[word] = nouns[word] + count
                else:
                    compounds[word] = compounds.get(word, 0) + count
                if word in words:
                    words[word] = words.get(word, 0) - count

        words = {word:count for word, count in words.items() if not (word in removals)}
        return nouns, compounds, words