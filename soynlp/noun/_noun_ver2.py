from collections import defaultdict
from collections import namedtuple
import os

from soynlp.utils import EojeolCounter
from soynlp.utils import LRGraph
from soynlp.utils import get_process_memory
from soynlp.tokenizer import MaxScoreTokenizer

NounScore = namedtuple('NounScore', 'frequency score')

class LRNounExtractor_v2:
    def __init__(self, l_max_length=10, r_max_length=9, predictor_headers=None,
        verbose=True, min_num_of_features=1, max_count_when_noun_is_eojeol=30,
        eojeol_counter_filtering_checkpoint=0, extract_compound=True,
        extract_determiner=False, extract_josa=False, postprocessing=None):

        self.l_max_length = l_max_length
        self.r_max_length = r_max_length
        self.lrgraph = None
        self.verbose = verbose
        self.min_num_of_features = min_num_of_features
        self.max_count_when_noun_is_eojeol = max_count_when_noun_is_eojeol
        self.eojeol_counter_filtering_checkpoint = eojeol_counter_filtering_checkpoint
        self.extract_compound = extract_compound

        if not postprocessing:
            postprocessing = ['detaching_features']
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

    def _extract_josa(self):
        raise NotImplemented

    def extract(self, minimum_noun_score=0.3, min_count=1, reset_lrgraph=True):

        # reset covered eojeol count
        self._num_of_covered_eojeols = 0

        # base prediction
        noun_candidates = self._noun_candidates_from_positive_features()
        prediction_scores = self._batch_prediction_order_by_word_length(
            noun_candidates, minimum_noun_score)

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

            coverage = '%.2f' % (100 * self._num_of_covered_eojeols
                / self._num_of_eojeols)
            print('[Noun Extractor] {} % eojeols are covered'.format(coverage), flush=True)

        if self.verbose:
            print('[Noun Extractor] flushing ... ', flush=True, end='')

        self._nouns = nouns
        if reset_lrgraph:
            # when extracting predicates, do not reset lrgraph.
            # the remained lrgraph is predicate (root - ending) graph
            self.lrgraph.reset_lrgraph()
        if self.verbose:
            print('done. mem={} Gb'.format('%.3f'%get_process_memory()))

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
            print(pos, common, neg, unk, end)

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
        for method in self.postprocessing:
            if method == 'detaching_features':
                n_before = len(nouns)
                nouns = _postprocess_detaching_features(nouns, self._pos_features)
                n_after = len(nouns)
                if self.verbose:
                    print('[NounExtractor] postprocessing {} : {} -> {}'.format(
                        method, n_before, n_after))
        return nouns

def _postprocess_detaching_features(nouns, features):
    removals = set()
    for word in nouns:
        if len(word) <= 2:
            continue
        for e in range(2, len(word)):
            if (word[:e] in nouns) and (word[e:] in features):
                removals.add(word)
                break
    ## debug code ##
    # print(sorted(removals, key=lambda x:-nouns[x][1])[:50])
    nouns_ = {word:score for word, score in nouns.items() if (word in removals) == False}
    return nouns_