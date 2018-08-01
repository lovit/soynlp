class EomiExtractor:

    def __init__(self, lrgraph, surfacial_stem, nouns, verbose=True):
        self.lrgraph = lrgraph
        self._pos_l = surfacial_stem
        self._nouns = nouns
        self.verbose = verbose

    def _print(self, message, replace=False, newline=True):
        header = '[Predicator Extractor]'
        if replace:
            print('\r{} {}'.format(header, message),
                  end='\n' if newline else '', flush=True)
        else:
            print('{} {}'.format(header, message),
                  end='\n' if newline else '', flush=True)

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

    def _batch_predicting_eomis(self,
        eomi_candidates, minimum_eomi_score=0.3, min_num_of_features=5):

        prediction_scores = {}

        n = len(eomi_candidates)
        for i, r in enumerate(sorted(eomi_candidates, key=lambda x:-len(x))):

            if self.verbose and i % 1000 == 999:
                percentage = '%.3f' % (100 * (i+1) / n)
                message = '  -- batch prediction {} % of {} words'.format(percentage, n)
                self._print(message, replace=True, newline=False)

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
            message = 'batch prediction was completed for {} words'.format(n)
            self._print(message, replace=True, newline=True)

        return prediction_scores