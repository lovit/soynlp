from collections import defaultdict

from soynlp.noun import LRNounExtractor_v2
from soynlp.predicator import PredicatorExtractor
from soynlp.lemmatizer import _lemma_candidate
from soynlp.lemmatizer import conjugate
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

        nouns_, confused_nouns, adjectives_, verbs_ = self._word_match_postprocessing(
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
        adjectives_ = {}
        verbs_ = {}

        for i, (eojeol, count) in enumerate(eojeol_counter.items()):

            if self._verbose and i % 1000 == 0:
                print('\r[POS Extractor] postprocessing {:.3f} % ...'.format(
                    as_percent(i, len(eojeol_counter))), end='')

            # cumulate total frequency
            total_frequency += count

            # check whether eojeol is predicator or not
            is_predicator = False
            if self._word_match(eojeol, adjectives):
                adjectives_[eojeol] = count
                is_predicator = True
            if self._word_match(eojeol, verbs):
                verbs_[eojeol] = count
                is_predicator = True
            if is_predicator:
                continue

            # check eojeol is noun + predicator compound
            noun = self._separate_predicator_from_noun(
                eojeol, nouns, adjectives, verbs)
            if noun is not None:
                nouns_[noun] += count
                # debug noun extraction results
                if eojeol in nouns:
                    confused_nouns[eojeol] += count
                continue

            # check eojeol is noun + predicator-suspect compound
            noun = self._separate_predicator_suspect_from_noun(
                eojeol, nouns, stems, eomis)
            if noun is not None:
                nouns_[noun] += count
                # debug noun extraction results
                if eojeol in nouns:
                    confused_nouns[eojeol] += count
                continue

            # check eojeol is stem + eomi
            lemmas = self._conjugatable(eojeol, stems, eomis)
            if lemmas is not None:
                stem_adjs = {stem for stem, _ in lemmas if stem in adjective_stems}
                stem_v = {stem for stem, _ in lemmas if stem in verb_stems}
                if stem_adjs:
                    adjectives_[eojeol] = count
                if stem_v:
                    verbs_[eojeol] = count
                if eojeol in nouns:
                    confused_nouns[eojeol] = count
                continue

            # else if eojeol is known as noun
            if eojeol in nouns:
                nouns_[eojeol] += count

        if self._verbose:
            print('\r[POS Extractor] postprocessing was done 100.00 %    ')
            print('[POS Extractor] Noun + [Josa/Predicator]: ({}, {:.3f} %)'.format(
                len(nouns_), as_percent(sum(nouns_.values()), total_frequency)))
            print('[POS Extractor] Confused nouns          : ({}, {:.3f} %)'.format(
                len(confused_nouns), as_percent(sum(confused_nouns.values()), total_frequency)))
            print('[POS Extractor] Adjective : ({}, {:.3f} %)'.format(
                len(adjectives_), as_percent(sum(adjectives_.values()), total_frequency)))
            print('[POS Extractor] Verb      : ({}, {:.3f} %)'.format(
                len(verbs_), as_percent(sum(verbs_.values()), total_frequency)))

        return nouns_, confused_nouns, adjectives_, verbs_

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
