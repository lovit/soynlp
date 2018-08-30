""" TERM DEFINITION
(l, r) : L and R position subwords
stem : stem of Adjective and Verb
ending : suffix, canonical form of ending

stems : set of stem including Adjectives and Verbs
composable_stems : stems that can be compounded with other prefix
    - [] + 하다 : 덕질+하다, 냐옹+하다, 냐옹+하냥
endings : set of ending
pos_l_features : canonical form set of stems (L subwords)
pos_composable_l_features : canonical form set of composable stems (L subwords)
lrgraph : L-R graph including [stem + Ending], Adverbs, 
          and maybe some Noun + Josa
"""

from soynlp.hangle import character_is_complete_korean
from soynlp.utils import LRGraph
from soynlp.utils import get_process_memory
from soynlp.utils import EojeolCounter
from soynlp.utils.utils import installpath
from soynlp.lemmatizer import _lemma_candidate
from soynlp.lemmatizer import _conjugate_stem
from ._eomi import EomiExtractor
from ._stem import StemExtractor

class PredicatorExtractor:

    def __init__(self, nouns, noun_pos_features=None, stems=None,
        eomis=None, extract_eomi=False, extract_stem=False, verbose=True):

        if not noun_pos_features:
            noun_pos_features = self._load_default_noun_pos_features()

        if not stems:
            stems = self._load_default_stems()

        if not eomis:
            eomis = self._load_default_eomis()

        self._nouns = nouns
        self._noun_pos_features = noun_pos_features
        self._stems = stems
        self._eomis = eomis
        self.verbose = verbose
        self.extract_eomi = extract_eomi
        self.extract_stem = extract_stem

        self._stem_surfaces = {l for stem in stems for l in _conjugate_stem(stem)}
        self.lrgraph = None

    def _load_default_noun_pos_features(self):
        path = '%s/trained_models/noun_predictor_ver2_pos' % installpath
        with open(path, encoding='utf-8') as f:
            pos_features = {word.split()[0] for word in f}
        return pos_features

    def _load_default_stems(self, min_count=100):
        dirs = '%s/lemmatizer/dictionary/default/Stem' % installpath
        paths = ['%s/Adjective.txt', '%s/Verb.txt']
        paths = [p % dirs for p in paths]
        stems = set()
        for path in paths:
            with open(path, encoding='utf-8') as f:
                for line in f:
                    word, count = line.split()
                    if int(count) < min_count:
                        continue
                    stems.add(word)
        return stems

    def _load_default_eomis(self, min_count=20):
        path = '%s/lemmatizer/dictionary/default/Eomi/Eomi.txt' % installpath
        eomis = set()
        with open(path, encoding='utf-8') as f:
            for line in f:
                word, count = line.split()
                if int(count) < min_count:
                    continue
                eomis.add(word)
        return eomis

    def _print(self, message, replace=False, newline=True):
        header = '[Predicator Extractor]'
        if replace:
            print('\r{} {}'.format(header, message),
                  end='\n' if newline else '', flush=True)
        else:
            print('{} {}'.format(header, message),
                  end='\n' if newline else '', flush=True)

    @property
    def is_trained(self):
        return self.lrgraph

    def train(self, sentences_or_lrgraph, min_eojeol_count=2,
        filtering_checkpoint=100000):

        if isinstance(sentences_or_lrgraph, soynlp.utils.LRGraph):
            self._train_with_lrgraph(sentences_or_lrgraph)
        else:
            self._train_with_sentences(sentences_or_lrgraph,
                min_eojeol_count, filtering_checkpoint)

    def _train_with_lrgraph(self, lrgraph):
        counter = {}
        for l, rdict in lrgraph._lr.items():
            for r, count in rdict.items():
                counter[l+r] = count

        self._num_of_eojeols = sum(counter.values())
        self._num_of_covered_eojeols = 0
        self.lrgraph = lrgraph

    def _train_with_sentences(self, sentences, min_eojeol_count=2,
        filtering_checkpoint=100000):

        check = filtering_checkpoint > 0

        if self.verbose:
            message = 'counting eojeols'
            self._print(message, replace=False, newline=False)

        # Eojeol counting
        counter = {}

        def contains_noun(eojeol, n):
            for e in range(2, n + 1):
                if eojeol[:e] in self.nouns:
                    return True
            return False

        for i_sent, sent in enumerate(sentences):

            if check and i_sent > 0 and i_sent % filtering_checkpoint == 0:
                counter = {
                    eojeol:count for eojeol, count in counter.items()
                    if count >= min_eojeol_count
                }

            if self.verbose and i_sent % 100000 == 99999:
                message = 'n eojeol = {} from {} sents. mem={} Gb{}'.format(
                    len(counter), i_sent + 1, '%.3f' % get_process_memory(), ' '*20)
                self._print(message, replace=True, newline=False)

            for eojeol in sent.split():

                n = len(eojeol)

                if n <= 1 or contains_noun(eojeol, n):
                    continue

                counter[eojeol] = counter.get(eojeol, 0) + 1

        if self.verbose:
            message = 'counting eojeols was done. {} eojeols, mem={} Gb{}'.format(
                len(counter), '%.3f' % get_process_memory(), ' '*20)
            self._print(message, replace=True, newline=True)

        counter = {
            eojeol:count for eojeol, count in counter.items()
            if count >= min_eojeol_count
        }

        self._num_of_eojeols = sum(counter.values())
        self._num_of_covered_eojeols = 0

        if self.verbose:
            message = 'complete eojeol counter -> lr graph'
            self._print(message, replace=False, newline=True)

        self.lrgraph = EojeolCounter()._to_lrgraph(
            counter,
            l_max_length=10,
            r_max_length=9
        )

        if self.verbose:
            message = 'has been trained. mem={} Gb'.format(
                '%.3f' % get_process_memory())
            self._print(message, replace=False, newline=True)

    def extract(self, candidates=None, min_count=10, reset_lrgraph=True,
        minimum_eomi_score=0.3, minimum_stem_score=0.3):

        # reset covered eojeol count
        self._num_of_covered_eojeols = 0

        # TODO link parameters
        if self.extract_eomi:
            self._extract_eomi()

        # TODO link parameters
        if self.extract_stem:
            self._extract_stem()

        return self._extract_predicator(candidates, min_count, reset_lrgraph)

    def _extract_eomi(self):
        eomi_extractor = EomiExtractor(
            lrgraph = self.lrgraph,
            stems = self._stems,
            nouns = self._nouns,
            min_num_of_features = 5,
            verbose = self.verbose,
            logpath = None
        )
        extracted_eomis = eomi_extractor.extract(
            condition=None,
            minimum_eomi_score=0.3,
            min_count=1,
            reset_lrgraph=True
        )
        extracted_eomis = {eomi for eomi in extracted_eomis if not (eomi in self._eomis)}
        self._eomis.update(extracted_eomis)

        if self.verbose:
            message = '{} eomis have been extracted'.format(len(extracted_eomis))
            self._print(message, replace=False, newline=True)

    def _extract_stem(self):
        stem_extractor = StemExtractor(
            lrgraph = self.lrgraph,
            stems = self._stems,
            eomis = self._eomis,
            min_num_of_unique_R_char=10,
            min_entropy_of_R_char=0.5,
            min_entropy_of_R=1.5
        )
        extracted_stems = stem_extractor.extract(
            L_ignore=None,
            minimum_stem_score=0.7,
            minimum_frequency=100
        )
        extracted_stems = {stem for stem in extracted_stems if not (stem in self._stems)}
        self._stems.update(extracted_stems)

        if self.verbose:
            message = '{} stems have been extracted'.format(len(extracted_stems))
            self._print(message, replace=False, newline=True)

    def _extract_predicator(self, eojeols=None, min_count=10, reset_lrgraph=True):

        lemmas = self._as_lemma_candidates(eojeols, min_count)
        # TODO
        # evaluation
        return lemmas

    def _as_lemma_candidates(self, eojeols=None,  min_count=10):

        def all_character_are_complete_korean(s):
            for c in s:
                if not character_is_complete_korean(c):
                    return False
            return True

        if not eojeols:
            eojeols = {l:rdict.get('', 0) for l, rdict in self.lrgraph._lr.items()}
            eojeols = {eojeol:count for eojeol, count in eojeols.items()
                       if (count > min_count) and all_character_are_complete_korean(eojeol)}

        n_eojeols = len(eojeols)
        lemmas = {}

        for i_eojeol, eojeol in enumerate(eojeols):

            if self.verbose and i_eojeol % 5000 == 0:
                perc = '%.3f'% (100 * i_eojeol / n_eojeols)
                message = 'lemma candidates ... {} %'.format(perc)
                self._print(message, replace=True, newline=False)

            n = len(eojeol)
            lemma_candidates = set()

            for i in range(1, n+1):
                l, r = eojeol[:i], eojeol[i:]
                for stem, eomi in _lemma_candidate(l, r):
                    if (stem in self._stems) and (eomi in self._eomis):
                        lemma_candidates.add((stem, eomi))

            if lemma_candidates:
                lemmas[eojeol] = lemma_candidates

        if self.verbose:
            message = 'lemma candidating was done     '
            self._print(message, replace=True, newline=True)

        return lemmas