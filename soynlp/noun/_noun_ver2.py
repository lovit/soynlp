from collections import defaultdict
from collections import namedtuple
import os

from soynlp.utils import check_dirs
from soynlp.utils import EojeolCounter
from soynlp.utils import LRGraph
from soynlp.utils import get_process_memory
from soynlp.tokenizer import MaxScoreTokenizer
from ._josa import extract_domain_pos_features
from ._noun_postprocessing import detaching_features
from ._noun_postprocessing import ignore_features

NounScore = namedtuple('NounScore', 'frequency score')

class LRNounExtractor_v2:
    def __init__(self, l_max_length=10, r_max_length=9, predictor_headers=None,
        verbose=True, min_num_of_features=1, max_count_when_noun_is_eojeol=30,
        eojeol_counter_filtering_checkpoint=0, extract_compound=True,
        extract_pos_feature=False, extract_determiner=False, postprocessing=None,
        logpath=None):

        self.l_max_length = l_max_length
        self.r_max_length = r_max_length
        self.lrgraph = None
        self.verbose = verbose
        self.min_num_of_features = min_num_of_features
        self.max_count_when_noun_is_eojeol = max_count_when_noun_is_eojeol
        self.eojeol_counter_filtering_checkpoint = eojeol_counter_filtering_checkpoint
        self.extract_compound = extract_compound
        self.extract_pos_feature = extract_pos_feature
        self.extract_determiner = extract_determiner
        self.logpath = logpath
        if logpath:
            check_dirs(logpath)

        if not postprocessing:
            postprocessing = ['detaching_features', 'ignore_features']
        elif isinstance(postprocessing) == str:
            postprocessing = [postprocessing]
        self.postprocessing = postprocessing

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

    def _append_features(self, feature_type, features):

        def check_feature_size():
            return (len(self._pos_features),
                    len(self._neg_features),
                    len(self._common_features))

        # size before
        n_pos, n_neg, n_common = check_feature_size()

        if feature_type == 'pos':
            commons = {f for f in features if (f in self._neg_features)}
            self._pos_features.update({f for f in features if not (f in commons)})
        elif feature_type == 'neg':
            commons = {f for f in features if (f in self._pos_features)}
            self._neg_features.update({f for f in features if not (f in commons)})
        elif feature_type == 'common':
            commons = features
        else:
            raise ValueError('Feature type was wrong. Choice = [pos, neg, common]')
        self._common_features.update(commons)

        # size after
        n_pos_, n_neg_, n_common_ = check_feature_size()

        if self.verbose:
            message = 'pos={} -> {}, neg={} -> {}, common={} -> {}'.format(
                n_pos, n_pos_, n_neg, n_neg_, n_common, n_common_)
            print('[Noun Extractor] features appended. {}'.format(message))

    def train_extract(self, sentences, minimum_noun_score=0.3,
        min_count=1, min_eojeol_count=1, reset_lrgraph=True):

        self.train(sentences, min_eojeol_count)

        return self.extract(minimum_noun_score, min_count, reset_lrgraph)

    def train(self, sentences, min_eojeol_count=1):

        if self.verbose:
            print('[Noun Extractor] counting eojeols')

        eojeol_counter = EojeolCounter(sentences, min_eojeol_count,
            max_length=self.l_max_length + self.r_max_length,
            filtering_checkpoint=self.eojeol_counter_filtering_checkpoint,
            verbose=self.verbose)
        self._num_of_eojeols = eojeol_counter._count_sum
        self._num_of_covered_eojeols = 0

        if self.verbose:
            print('[Noun Extractor] complete eojeol counter -> lr graph')
        self.lrgraph = eojeol_counter.to_lrgraph(
            self.l_max_length, self.r_max_length)

        if self.verbose:
            print('[Noun Extractor] has been trained. mem={} Gb'.format(
                '%.3f'%get_process_memory()))

    def _extract_determiner(self):
        raise NotImplemented

    def extract_domain_pos_features(self, append_extracted_features=True,
        noun_candidates=None, ignore_features=None,
        min_noun_score=0.3, min_noun_frequency=100,
        min_pos_score=0.3, min_pos_feature_frequency=1000,
        min_num_of_unique_lastchar=4, min_entropy_of_lastchar=0.5,
        min_noun_entropy=1.5):

        if self.verbose:
            print('[Noun Extractor] batch prediction for extracting pos feature')

        if not noun_candidates:
            noun_candidates = self._noun_candidates_from_positive_features()

        prediction_scores = self._batch_prediction_order_by_word_length(
            noun_candidates, min_noun_score)
        self.lrgraph.reset_lrgraph()

        self._pos_features_extracted = extract_domain_pos_features(
            prediction_scores, self.lrgraph,
            self._pos_features, ignore_features,
            min_noun_score, min_noun_frequency,
            min_pos_score, min_pos_feature_frequency,
            min_num_of_unique_lastchar, min_entropy_of_lastchar,
            min_noun_entropy
        )

        self._append_features('pos', self._pos_features_extracted)
        if self.verbose:
            print('[Noun Extractor] {} pos features were extracted'.format(
                len(self._pos_features_extracted)))

    def extract(self, minimum_noun_score=0.3, min_count=1, reset_lrgraph=True):

        # reset covered eojeol count
        self._num_of_covered_eojeols = 0

        # base prediction
        noun_candidates = self._noun_candidates_from_positive_features()

        if self.extract_pos_feature:
            self.extract_domain_pos_features(noun_candidates)

        prediction_scores = self._batch_prediction_order_by_word_length(
            noun_candidates, minimum_noun_score)

        if self.logpath:
            with open(self.logpath+'_prediction_score.log', 'w', encoding='utf-8') as f:
                f.write('noun score frequency\n')
                for word, score in sorted(prediction_scores.items(), key=lambda x:-x[1][1]):
                    f.write('{} {} {}\n'.format(word, score[0], score[1]))

        # E = N*J+ or N*Posi+
        if self.extract_compound:
            candidates = {l:sum(rdict.values()) for l,rdict in
                self.lrgraph._lr.items() if len(l) >= 4}
            compounds = self.extract_compounds(
                candidates, prediction_scores, minimum_noun_score)
        else:
            compounds = {}

        # combine single nouns and compounds
        nouns = {noun:score for noun, score in prediction_scores.items()
            if score[0] >= minimum_noun_score}
        nouns.update(compounds)

        # frequency filtering
        nouns = {noun:score for noun, score in nouns.items()
            if score[1] >= min_count}

        nouns = self._post_processing(nouns, prediction_scores, compounds)

        if self.verbose:
            print('[Noun Extractor] {} nouns ({} compounds) with min count={}'.format(
                len(nouns), len(compounds), min_count), flush=True)
            print('[Noun Extractor] flushing ... ', flush=True, end='')

        self._check_covered_eojeols(nouns)

        self._nouns = nouns
        if reset_lrgraph:
            # when extracting predicates, do not reset lrgraph.
            # the remained lrgraph is predicate (root - ending) graph
            self.lrgraph.reset_lrgraph()
        if self.verbose:
            print('done. mem={} Gb'.format('%.3f'%get_process_memory()))
            coverage = '%.2f' % (100 * self._num_of_covered_eojeols
                / self._num_of_eojeols)
            print('[Noun Extractor] {} % eojeols are covered'.format(coverage), flush=True)

        nouns_ = {noun:NounScore(score[1], score[0]) for noun, score in nouns.items()}
        return nouns_

    def _get_nonempty_features(self, word, features):
        return [r for r, _ in features if (
            ( (r in self._pos_features) and (not self._exist_longer_pos(word, r)) ) or
            ( (r in self._neg_features) and (not self._exist_longer_neg(word, r)) ) )]

    def _exist_longer_pos(self, word, r):
        for e in range(len(word)-1, -1, -1):
            if (word[e:]+r) in self._pos_features:
                return True
        return False

    def _exist_longer_neg(self, word, r):
        for e in range(len(word)-1, -1, -1):
            if (word[e:]+r) in self._neg_features:
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
            print('pos={}, common={}, neg={}, unk={}, end={}'.format(
                pos, common, neg, unk, end))

        features_ = self._get_nonempty_features(word, features)
        if len(features_) > self.min_num_of_features:        
            return score, support
        else:
            # exception case
            sum_ = pos + common + neg + unk + end
            if sum_ == 0:
                return 0, support

            # exception. frequent nouns may have various positive R such as Josa
            if ((end > self.max_count_when_noun_is_eojeol) and (neg >= pos) ):
                return score, support

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
            if self._exist_longer_neg(word, r): # negative -다고
                neg += freq
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
                    if r == '' or (r in self._pos_features) or (r in self._common_features):
                        self.lrgraph.remove_eojeol(word+r, count)
                        # Do 'eojeol counting covered' in flushing step
                        # self._num_of_covered_eojeols += count
        if self.verbose:
            print('\r[Noun Extractor] batch prediction was completed for {} words'.format(
                n), flush=True)

        return prediction_scores

    def extract_compounds(self, candidates, prediction_scores, minimum_noun_score=0.3):

        noun_scores = {noun:len(noun) for noun, score in prediction_scores.items()
                       if score[0] > minimum_noun_score and len(noun) > 1}
        self._compound_decomposer = MaxScoreTokenizer(scores=noun_scores)

        candidates = {l:sum(rdict.values()) for l,rdict in self.lrgraph._lr.items()
            if (len(l) >= 4) and not (l in noun_scores)}

        n = len(candidates)
        compounds_scores = {}
        compounds_counts = {}
        compounds_components = {}

        for i, (word, count) in enumerate(sorted(candidates.items(), key=lambda x:-len(x[0]))):

            if self.verbose and i % 1000 == 999:
                percentage = '%.2f' % (100 * i / n)
                print('\r  -- check compound {} %'.format(percentage), flush=True, end='')

            # skip if candidate is substring of longer compound
            if candidates.get(word, 0) <= 0:
                continue

            tokens = self._compound_decomposer.tokenize(word, flatten=False)[0]
            compound_parts = self._parse_compound(tokens)

            if compound_parts:
                # store compound components
                noun = ''.join(compound_parts)
                compounds_components[noun] = compound_parts
                # cumulate count and store compound score
                compound_score = max((prediction_scores.get(t, (0,0))[0] for t in compound_parts))
                compounds_scores[noun] = max(compounds_scores.get(noun,0), compound_score)
                compounds_counts[noun] = compounds_counts.get(noun,0) + count
                # reduce frequency of substrings
                for e in range(2, len(word)):
                    subword = word[:e]
                    if not subword in candidates:
                        continue
                    candidates[subword] = candidates.get(subword, 0) - count
                # eojeol coverage
                self.lrgraph.remove_eojeol(word)
                self._num_of_covered_eojeols += count

        if self.verbose:
            print('\r[Noun Extractor] checked compounds. discovered {} compounds'.format(
                len(compounds_scores)))

        compounds = {noun:(score, compounds_counts.get(noun,0))
             for noun, score in compounds_scores.items()}

        self._compounds_components = compounds_components

        return compounds

    def decompose_compound(self, word):

        tokens = self._compound_decomposer.tokenize(word, flatten=False)[0]
        compound_parts = self._parse_compound(tokens)

        return (word, ) if not compound_parts else compound_parts

    def _parse_compound(self, tokens):
        """Check Noun* or Noun*Josa"""

        # format: (word, begin, end, score, length)
        for token in tokens[:-1]:
            if token[3] <= 0:
                return None
        # Noun* + Josa
        if len(tokens) >= 3 and tokens[-1][0] in self._pos_features:
            return tuple(t[0] for t in tokens[:-1])
        # all tokens are noun
        if tokens[-1][3] > 0:
            return tuple(t[0] for t in tokens)
        # else, not compound
        return None

    def _post_processing(self, nouns, prediction_scores, compounds):

        n_before = len(nouns)

        for method in self.postprocessing:

            if method == 'detaching_features':
                logheader = '## Ignore noun candidates from detaching pos features\n'
                nouns, removals = detaching_features(nouns,
                    self._pos_features, self.logpath+'_postprocessing.log', logheader)

            elif method == 'ignore_features':
                features = {f for f in self._pos_features}
                # features.update(self._neg_features)
                features.update(self._common_features)
                nouns, removals = ignore_features(nouns,
                    features, self.logpath+'_postprocessing.log')

        n_after = len(nouns)
        if self.verbose:
            print('[Noun Extractor] postprocessing {} : {} -> {}'.format(
                method, n_before, n_after))

        return nouns

    def _check_covered_eojeols(self, nouns):

        def get_r_from_origin(l):
            return self.lrgraph._lr_origin.get(l, {}).items()

        for noun in nouns:
            for r, count in get_r_from_origin(noun):
                if ((r == '') or
                    (r in self._pos_features) or
                    (r in self._common_features)):
                    self._num_of_covered_eojeols += count