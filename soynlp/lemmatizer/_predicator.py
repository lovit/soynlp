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
from ._stem import extract_domain_stem

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
        self._pos_l = {l for stem in stems for l in _conjugate_stem(stem)}
        self._eomis = eomis
        self.verbose = verbose
        self.extract_eomi = extract_eomi
        self.extract_stem = extract_stem
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

    def train(self, sentences, min_eojeol_count=2,
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

    def extract(self, minimum_eomi_score=0.3, min_count=10, reset_lrgraph=True):

        # reset covered eojeol count
        self._num_of_covered_eojeols = 0

        # base prediction
        eomi_candidates = self._eomi_candidates_from_stems()

    def extract_domain_stem(self, append_extracted_stem=True,
        eomi_candidates=None, L_ignore=None,
        min_eomi_score=0.3, min_eomi_frequency=100,
        min_score_of_L=0.7, min_frequency_of_L=100,
        min_num_of_unique_R_char=10, min_entropy_of_R_char=0.5,
        min_entropy_of_R=1.5):

        if self.verbose:
            message = 'batch prediction for extracting stem'
            self._print(message, replace=False, newline=True)

        if not eomi_candidates:
            eomi_candidates = self._eomi_candidates_from_stems()

        # TODO: min_num_of_features -> init argument
        prediction_scores = self._batch_predicting_eomis(
            eomi_candidates, min_eomi_score, min_num_of_features=5)

        R = {r for r, score in prediction_scores.items()
             if ((score[0] >= min_eomi_score)
                 and (score[1] >= min_eomi_frequency))}

        self.lrgraph.reset_lrgraph()

        self._stems_extracted, self._pos_l_extracted = extract_domain_stem(
            self.lrgraph,
            self._pos_l,
            R,
            L_ignore,
            min_score_of_L,
            min_frequency_of_L,
            min_num_of_unique_R_char,
            min_entropy_of_R_char,
            min_entropy_of_R
        )

        if append_extracted_stem:
            self._append_features('stems', self._stems_extracted)
            self._append_features('pos_l', self._pos_l_extracted)

        if self.verbose:
            message = '{} stems ({} L) were extracted'.format(
                len(self._stems_extracted), len(self._pos_l_extracted))
            self._print(message, replace=False, newline=True)

    def _append_features(self, feature_type, features):

        def check_size():
            return (len(self._stems), len(self._pos_l))

        # size before
        n_stems, n_pos_l = check_size()

        if feature_type == 'stems':
            self._stems.update(features)
        elif feature_type == 'pos_l':
            self._pos_l.update(features)

        # size after
        n_stems_, n_pos_l_ = check_size()

        if self.verbose:
            message = 'stems appended: stems={} -> {}, L={} -> {}'.format(
                n_stems, n_pos_l, n_stems_, n_pos_l_)
            self._print(message, replace=False, newline=True)

    def extract_predicator(self, eojeols=None, minimum_eomi_score=0.3,
        minimum_stem_score=0.3, min_count=10, reset_lrgraph=True):

        #if self.extract_stem:
        # TODO

        # if self.extract_eomi:
        # TODO

        lemmas = self._as_lemma_candidates(eojeols, min_count)
        # TODO
        # evaluation
        return lemmas

    def _as_lemma_candidates(self, eojeols=None,  min_count=10):

        if not eojeols:
            eojeols = {l:rdict.get('', 0) for l, rdict in self.lrgraph._lr.items()}
            eojeols = [eojeol for eojeol, count in eojeols.items()
                       if count > min_count]

        def all_character_are_complete_korean(s):
            for c in s:
                if not character_is_complete_korean(c):
                    return False
            return True

        eojeols = [eojeol for eojeol in eojeols
                   if all_character_are_complete_korean(eojeol)]

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