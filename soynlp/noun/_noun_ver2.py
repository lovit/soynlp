from collections import defaultdict
import os

from soynlp.utils import EojeolCounter
from soynlp.utils import LRGraph


class LRNounExtractor_v2:
    def __init__(self, max_l_len=10, max_r_len=9,
        predictor_headers=None, verbose=True, min_num_of_features=1):

        self.max_l_len = max_l_len
        self.max_r_len = max_r_len
        self.lrgraph = None
        self.verbose = verbose
        self.min_num_of_features = min_num_of_features

        if not predictor_headers:
            predictor_headers = self._set_default_predictor_header()
        self._load_predictor(predictor_headers)

    @property
    def is_trained(self):
        return self.lrgraph

    def _set_default_predictor_header(self):

        if self.verbose:
            print('[Noun Extractor] use default predictors')

        dirname = '/'.join(os.path.abspath(__file__).replace('\\', '/').split('/')[:-2])
        predictor_header = ['{}/trained_models/noun_predictor_ver2'.format(dirname)]

        return predictor_header

    def _load_predictor(self, headers):

        if type(headers) == str:
            headers = [headers]
        
        pos, neg = set(), set()
        for header in headers:

            # load positive features such as Josa
            pos_path = '{}_pos'.format(header)
            with open(pos_path, encoding='utf-8') as f:
                pos.update({feature.strip() for feature in f})

            # load negative features such as ending (Eomi)
            neg_path = '{}_neg'.format(header)
            with open(neg_path, encoding='utf-8') as f:
                neg.update({feature.strip() for feature in f})

        # common features such as -은 (조사/어미), -라고(조사/어미) 
        common = pos.intersection(neg)

        # remove common features from pos and neg
        pos = {feature for feature in pos if not (feature in common)}
        neg = {feature for feature in neg if not (feature in common)}

        if self.verbose:
            print('[Noun Extractor] num features: pos={}, neg={}, common={}'.format(
                len(pos), len(neg), len(common)))

        self._pos_features = pos
        self._neg_features = neg
        self._common_features = common

    def train_extract(self, sentences, minimum_noun_score=0.3,
        min_count=1, min_eojeol_count=1):

        self.train(sentences, min_eojeol_count)

        return self.extract(minimum_noun_score, min_count)

    def train(self, sentences, min_eojeol_count=1):

        if self.verbose:
            print('[Noun Extractor] counting eojeols')
        eojeol_counter = EojeolCounter(sentences, min_eojeol_count,
            max_length=self.max_l_len + self.max_r_len)

        if self.verbose:
            print('[Noun Extractor] complete eojeol counter -> lr graph')
        self.lrgraph = eojeol_counter.to_lrgraph(
            self.max_l_len, self.max_r_len)

        if self.verbose:
            print('[Noun Extractor] has been trained.')
    
    def extract(self, minimum_noun_score=0.3, min_count=1):
        return {}

    def _get_nonempty_features(self, word, features):
        return [r for r, _ in features if ((not self._exist_longer_pos(word, r)) and 
            (r in self._pos_features) or (r in self._neg_features))]

    def _exist_longer_pos(self, word, r):
        for e in range(len(word)-1, -1, -1):
            if (word[e:]+r) in self._pos_features:
                return True
        return False

    def predict(self, word, minimum_noun_score=0.3, debug=False):

        # scoring
        features = self.lrgraph.get_r(word, -1)
        pos, common, neg, unk, end = self._predict(word, features)

        base = pos + neg
        score = 0 if base == 0 else (pos - neg) / base
        support = pos + end + common if score >= minimum_noun_score else neg + end + common

        # debug code
        if debug:
            print(pos, common, neg, unk, end)

        features_ = self._get_nonempty_features(word, features)
        if len(features_) > self.min_num_of_features:        
            return score, support
        else:
            # exception case
            sum_ = pos + common + neg + unk + end
            if sum_ == 0:
                return 0, support

            if (common > 0 or pos > 0) and (end / sum_ >= 0.3) and (common >= neg):
                # 아이웨딩 + [('', 90), ('은', 3), ('측은', 1)] # 은 common / 대부분 단일어절 / 측은 unknown. 
                # 아이엠텍 + [('은', 2), ('', 2)]
                support = pos + common + end
                return (support / sum_, support)

            # 경찰국 + [(은, 1), (에, 1), (에서, 1)] -> {은, 에}
            first_chars = set()
            for r, _ in features:
                if not r:
                    continue
                if r in self._pos_features or r in self._common_features:
                    if not self._exist_longer_pos(word, r):
                        first_chars.add(r[0])
                if not (r in self._pos_features or r in self._common_features):
                    first_chars.add(r[0])

            if len(first_chars) >= 2:
                support = pos + common + end
                return (support / sum_, support)

            # Handling for post-processing in NounExtractor
            # Case 1.
            # 아이러브영주사과 -> 아이러브영주사 + [(과,1)] (minimum r feature 적용해야 하는 케이스) : 복합명사
            # 아이러브영주사과 + [('', 1)] 이므로, 후처리 이후 '아이러브영주사' 후보에서 제외됨
            # Case 2.
            # 아이였으므로 -> 아이였으므 + [(로, 2)] (minimum r feature 적용)
            # "명사 + Unknown R" 로 후처리
            return (0, support)

    def _predict(self, word, features):

        pos, common, neg, unk, end = 0, 0, 0, 0, 0

        for r, freq in features:
            if r == '':
                end += freq
                continue
            if self._exist_longer_pos(word, r): # ignore
                continue
            if r in self._common_features:
                common += freq
            elif r in self._pos_features:            
                pos += freq
            elif r in self._neg_features:
                neg += freq
            else:
                unk += freq

        return pos, common, neg, unk, end

    def _noun_candidates_from_positive_features(self, condition=None):

        def satisfy(word, e):
            return word[:e] == condition

        # noun candidates from positive featuers such as Josa
        N_from_J = {}
        for r in self._pos_features:
            for l, c in self.lrgraph.get_l(r, -1):
                # candidates filtering for debugging
                # condition is first chars in L
                if not condition:
                    N_from_J[l] = N_from_J.get(l,0) + c
                    continue
                # for debugging
                if not satisfy(l, len(condition)):
                    continue
                N_from_J[l] = N_from_J.get(l,0) + c

        # sort by length of word
        N_from_J = sorted(N_from_J.items(), key=lambda x:-len(x[0]))

        return N_from_J

    def _batch_prediction_order_by_word_length(self,
        noun_candidates, minimum_noun_score=0.3):

        prediction_scores = {}

        n = len(noun_candidates)
        for i, (word, _) in enumerate(noun_candidates):

            if self.verbose and i % 1000 == 999:
                percentage = '%.3f' % (100 * (i+1) / n)
                print('\r  -- batch prediction {} % of {} words'.format(
                    percentage, n), flush=True, end='')

            # base prediction
            score, support = self.predict(word, minimum_noun_score)
            prediction_scores[word] = (score, support)

            # if their score is higher than minimum_noun_score,
            # remove eojeol pattern from lrgraph
            if score >= minimum_noun_score:
                for r, count in self.lrgraph.get_r(word, -1):
                    if r == '' or (r in self._pos_features):
                        self.lrgraph.remove_eojeol(word+r, count)
        if self.verbose:
            print('\r[Noun Extractor] batch prediction was completed.', flush=True)

        return prediction_scores