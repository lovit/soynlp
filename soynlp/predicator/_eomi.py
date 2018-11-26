from collections import defaultdict
from collections import namedtuple
from soynlp.hangle import decompose
from soynlp.lemmatizer import lemma_candidate
from soynlp.lemmatizer import _conjugate_stem

EomiScore = namedtuple('EomiScore', 'frequency score')

class EomiExtractor:

    def __init__(self, lrgraph, stems, nouns,
        min_num_of_features=5, verbose=True, logpath=None):

        self.lrgraph = lrgraph
        self._stems = stems
        self._nouns = nouns
        self.min_num_of_features = min_num_of_features
        self.verbose = verbose
        self.logpath = logpath
        self._eomis = None

    @property
    def is_trained(self):
        return self._eomis is not None

    def _print(self, message, replace=False, newline=True):
        header = '[Eomi Extractor]'
        if replace:
            print('\r{} {}'.format(header, message),
                  end='\n' if newline else '', flush=True)
        else:
            print('{} {}'.format(header, message),
                  end='\n' if newline else '', flush=True)

    def extract(self, condition=None, min_eomi_score=0.3,
        min_eomi_frequency=1, reset_lrgraph=True):

        # reset covered eojeol count and extracted eomis
        self._num_of_covered_eojeols = 0
        self._eomis = {}

        self._stem_surfaces = {l for stem in self._stems for l in _conjugate_stem(stem)}

        # base prediction
        candidates = self._candidates_from_stem_surfaces(condition)

        prediction_scores = self._batch_prediction(
            candidates, min_eomi_score, self.min_num_of_features)

        eomi_surfaces = {eomi:score for eomi, score in prediction_scores.items()
            if (score[1] >= min_eomi_score)}

        if self.verbose:
            message = 'eomi lemmatization with {} candidates'.format(len(eomi_surfaces))
            self._print(message, replace=False, newline=True)

        self.lrgraph.reset_lrgraph()
        lemmas = self._eomi_lemmatize(eomi_surfaces)

        lemmas = {eomi:score for eomi, score in lemmas.items()
            if (score[0] >= min_eomi_frequency) and (score[1] >= min_eomi_score)}

        if self.logpath:
            with open(self.logpath+'_eomi_prediction_score.log', 'w', encoding='utf-8') as f:
                f.write('eomi frequency score\n')

                for word, score in sorted(prediction_scores.items(), key=lambda x:-x[1][1]):
                    f.write('{} {} {}\n'.format(word, score[0], score[1]))

        if self.verbose:
            message = '{} eomis extracted with min frequency = {}, min score = {}'.format(
                len(lemmas), min_eomi_frequency, min_eomi_score)
            self._print(message, replace=False, newline=True)

        self._check_covered_eojeols(lemmas) # TODO with lemma

        self._eomis = lemmas

        if reset_lrgraph:
            self.lrgraph.reset_lrgraph()

        del self._stem_surfaces

        lemmas_ = {eomi:EomiScore(score[0], score[1]) for eomi, score in lemmas.items()}
        return lemmas_

    def predict(self, r, min_eomi_score=0.3,
        min_num_of_features=5, debug=False):

        features = self.lrgraph.get_l(r, -1)

        pos, neg, unk = self._predict(features, r)

        base = pos + neg
        score = 0 if base == 0 else (pos - neg) / base
        support = pos + unk if score >= min_eomi_score else neg + unk

        features_ = self._refine_features(features, r)
        n_features_ = len(features_)

        if debug:
            print('pos={}, neg={}, unk={}, n_features_={}'.format(
                pos, neg, unk, n_features_))

        if n_features_ >= min_num_of_features:
            return support, score
        else:
            # TODO
            return (0, 0)

    def _predict(self, features, r):

        pos, neg, unk = 0, 0, 0

        for l, freq in features:
            if (l + r) in self._nouns:
                continue
            if self._exist_longer_pos(l, r): # ignore
                continue
            if l in self._stem_surfaces:
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
            if (l + r[:i]) in self._stem_surfaces:
                return True
        return False

    def _is_aNoun_Verb(self, l):
        return (l[0] in self._nouns) and (l[1:] in self._stem_surfaces)

    def _has_stem_at_last(self, l):
        for i in range(1, len(l)):
            if l[-i:] in self._stem_surfaces:
                return True
        return False

    def _refine_features(self, features, r):
        return [(l, count) for l, count in features if
            ((l in self._stem_surfaces) and (not self._exist_longer_pos(l, r)))]

    def _candidates_from_stem_surfaces(self, condition=None):

        def satisfy(word, e):
            return word[-e:] == condition

        # noun candidates from positive featuers such as Josa
        R_from_L = {}

        for l in self._stem_surfaces:
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

    def _batch_prediction(self, eomi_candidates,
        min_eomi_score=0.3, min_num_of_features=5):

        prediction_scores = {}

        n = len(eomi_candidates)
        for i, r in enumerate(sorted(eomi_candidates, key=lambda x:-len(x))):

            if self.verbose and i % 10000 == 9999:
                percentage = '%.2f' % (100 * (i+1) / n)
                message = '  -- batch prediction {} % of {} words'.format(percentage, n)
                self._print(message, replace=True, newline=False)

            # base prediction
            support, score = self.predict(
                r, min_eomi_score, min_num_of_features)
            prediction_scores[r] = (support, score)

            # if their score is higher than min_eomi_score,
            # remove eojeol pattern from lrgraph
            if score >= min_eomi_score:
                for l, count in self.lrgraph.get_l(r, -1):
                    if ((l in self._stem_surfaces) or
                        self._is_aNoun_Verb(l)):
                        self.lrgraph.remove_eojeol(l+r, count)

        self.lrgraph.reset_lrgraph()

        if self.verbose:
            message = 'batch prediction was completed for {} words'.format(n)
            self._print(message, replace=True, newline=True)

        return prediction_scores

    def _eomi_lemmatize(self, eomis):

        def merge_score(freq0, score0, freq1, score1):
            return (freq0 + freq1, (score0 * freq0 + score1 * freq1) / (freq0 + freq1))

        eomis_ = {}
        #lrgraph = defaultdict(lambda: defaultdict(int))
        #lemma_to_word = defaultdict(lambda: [])
        for eomi, (_, score0) in eomis.items():
            for stem_surface, count in self.lrgraph.get_l(eomi, -1):
                try:
                    for stem_, eomi_ in lemma_candidate(stem_surface, eomi):
                        if not (stem_ in self._stems):
                            continue
                        eomis_[eomi_] = merge_score(count, score0, *eomis_.get(eomi_, (0, 0)))
                        #lrgraph[stem_][eomi_] += count
                        #lemma_to_word[(stem_, eomi_)].append(stem_surface + eomi)
                # stem 이 한글이 아닌 경우 불가
                except Exception as e:
                    continue

        return eomis_

    def _postprocess(self, eomis):
        eomis_ = {}
        for eomi, score in eomis.items():
            # TODO
            # Remove E + V + E : -서가지고
            # Remove V + E : -싶구나
            eomis_[eomi] = score

        return eomis_

    def _check_covered_eojeols(self, eomis):
        # TODO
        return None