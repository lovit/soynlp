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

from soynlp.utils import LRGraph
from soynlp.utils import get_process_memory
from soynlp.utils import EojeolCounter
from soynlp.utils.utils import installpath
from soynlp.lemmatizer import _lemma_candidate
from ._stem import extract_domain_stem

class EomiExtractor:

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

    @property
    def is_trained(self):
        return self.lrgraph

    def train(self, sentences, min_eojeol_count=2,
        filtering_checkpoint=100000):

        check = filtering_checkpoint > 0

        if self.verbose:
            print('[Eomi Extractor] counting eojeols', end='')

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
                message = '\r[Eomi Extractor] n eojeol = {} from {} sents. mem={} Gb{}'.format(
                    len(counter), i_sent + 1, '%.3f' % get_process_memory(), ' '*20)
                print(message, flush=True, end='')

            for eojeol in sent.split():

                n = len(eojeol)

                if n <= 1 or contains_noun(eojeol, n):
                    continue

                counter[eojeol] = counter.get(eojeol, 0) + 1

        if self.verbose:
            message = '\r[Eomi Extractor] counting eojeols was done. {} eojeols, mem={} Gb{}'.format(
                len(counter), '%.3f' % get_process_memory(), ' '*20)
            print(message)

        counter = {
            eojeol:count for eojeol, count in counter.items()
            if count >= min_eojeol_count
        }

        self._num_of_eojeols = sum(counter.values())
        self._num_of_covered_eojeols = 0

        if self.verbose:
            print('[Eomi Extractor] complete eojeol counter -> lr graph')

        self.lrgraph = EojeolCounter()._to_lrgraph(
            counter,
            l_max_length=10,
            r_max_length=9
        )

        if self.verbose:
            print('[Eomi Extractor] has been trained. mem={} Gb'.format(
                '%.3f' % get_process_memory()))

    def extract(self, minimum_eomi_score=0.3, min_count=10, reset_lrgraph=True):

        # reset covered eojeol count
        self._num_of_covered_eojeols = 0

        # base prediction
        eomi_candidates = self._eomi_candidates_from_stems()

    def _eomi_candidates_from_stems(self, condition=None):

        def satisfy(word, e):
            return word[-e:] == condition

        # noun candidates from positive featuers such as Josa
        R_from_L = {}

        for l in self._pos_l:
            for r, c in self.lrgraph.get_r(l, -1):

                # candidates filtering for debugging
                # condition is last chars in R
                if not condition:
                    R_from_L[r] = R_from_L.get(r,0) + c
                    continue

                # for debugging
                if satisfy(r, len(condition)):
                    R_from_L[r] = R_from_L.get(r,0) + c

        return R_from_L

    def predict_r(self, r, minimum_r_score=0.3,
        min_num_of_features=5, debug=False):

        features = self.lrgraph.get_l(r, -1)

        pos, neg, unk = self._predict_r(features, r)

        base = pos + neg
        score = 0 if base == 0 else (pos - neg) / base
        support = pos + unk if score >= minimum_r_score else neg + unk

        features_ = self._refine_features(features, r)
        n_features_ = len(features_)

        if debug:
            print('pos={}, neg={}, unk={}, n_features_={}'.format(
                pos, neg, unk, n_features_))

        if n_features_ >= min_num_of_features:
            return score, support
        else:
            # TODO
            return (0, 0)

    def _predict_r(self, features, r):

        pos, neg, unk = 0, 0, 0

        for l, freq in features:
            if self._exist_longer_pos(l, r): # ignore
                continue
            if l in self._pos_l:
                pos += freq
            elif self._is_aNoun_Verb(l):
                pos += freq
            elif self._has_stem_at_last(l):
                unk += freq
            else:
                neg += freq

        return pos, neg, unk

    def _exist_longer_pos(self, l, r):
        for i in range(1, len(r)+1):
            if (l + r[:i]) in self._pos_l:
                return True
        return False

    def _is_aNoun_Verb(self, l):
        return (l[0] in self._nouns) and (l[1:] in self._pos_l)

    def _has_stem_at_last(self, l):
        for i in range(1, len(l)):
            if l[-i:] in self._pos_l:
                return True
        return False

    def _refine_features(self, features, r):
        return [(l, count) for l, count in features if
            ((l in self._pos_l) and (not self._exist_longer_pos(l, r)))]

    def _batch_predicting_eomis(self,
        eomi_candidates, minimum_eomi_score=0.3, min_num_of_features=5):

        prediction_scores = {}

        n = len(eomi_candidates)
        for i, r in enumerate(sorted(eomi_candidates, key=lambda x:-len(x))):

            if self.verbose and i % 1000 == 999:
                percentage = '%.3f' % (100 * (i+1) / n)
                print('\r  -- batch prediction {} % of {} words'.format(
                    percentage, n), flush=True, end='')

            # base prediction
            score, support = self.predict_r(
                r, minimum_eomi_score, min_num_of_features)
            prediction_scores[r] = (score, support)

            # if their score is higher than minimum_eomi_score,
            # remove eojeol pattern from lrgraph
            if score >= minimum_eomi_score:
                for l, count in self.lrgraph.get_l(r, -1):
                    if ((l in self._pos_l) or
                        self._is_aNoun_Verb(l)):
                        self.lrgraph.remove_eojeol(l+r, count)

        if self.verbose:
            print('\r[Eomi Extractor] batch prediction was completed for {} words'.format(
                n), flush=True)

        return prediction_scores

    def extract_domain_stem(self, append_extracted_stem=True,
        eomi_candidates=None, L_ignore=None,
        min_eomi_score=0.3, min_eomi_frequency=100,
        min_score_of_L=0.7, min_frequency_of_L=100,
        min_num_of_unique_R_char=10, min_entropy_of_R_char=0.5,
        min_entropy_of_R=1.5):

        if self.verbose:
            print('[Eomi Extractor] batch prediction for extracting stem')

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
            print('[Eomi Extractor] {} stems ({} L) were extracted'.format(
                len(self._stems_extracted), len(self._pos_l_extracted)))

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
            message = 'stems={} -> {}, L={} -> {}'.format(
                n_stems, n_pos_l, n_stems_, n_pos_l_)
            print('[Eomi Extractor] stems appended. {}'.format(message))

    def extract_predicator(self, minimum_eomi_score=0.3,
        minimum_stem_score=0.3, min_count=10, reset_lrgraph=True):

        # if self.extract_eomi:
            # TODO

        # if self.extract_stem:
            # TODO

        eojeols = {} # Eojeol initialization
        n_eojeols = len(eojeols)

        for i_eojeol, eojeol in enumerate(eojeols):
            n = len(eojeol)
            candidates = set()
            for i in range(1, n+1):
                l, r = eojeol[:i], eojeol[i:]
                candidates.update(_lemma_candidate(l, r))
            candidates_ = []
            for l, r in candidates:
                if (l in self._stems) and (r in self._eomis):
                    candidates_.append((l, r))
            # TODO: evaluation lemma of (stem, eomi)

        raise NotImplemented