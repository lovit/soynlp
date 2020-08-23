import os
import re
from collections import namedtuple, OrderedDict
from datetime import datetime
from pprint import pprint
from tqdm import tqdm

from soynlp.tokenizer import MaxScoreTokenizer
from soynlp.utils import DoublespaceLineCorpus, EojeolCounter, LRGraph, get_process_memory
from .postprocessing import detaching_features, ignore_features, check_N_is_NJ


installpath = os.path.abspath(os.path.dirname(__file__))
NounScore = namedtuple('NounScore', 'frequency score')


class LRNounExtractor():
    def __init__(
        self,
        max_l_length=10,
        max_r_length=9,
        pos_features=None,
        neg_features=None,
        verbose=True,
        debug_dir=None,
    ):

        self.max_l_length = max_l_length
        self.max_r_length = max_r_length
        self.verbose = verbose
        self.debug_dir = debug_dir
        self.pos, self.neg, self.common = prepare_r_features(pos_features, neg_features)

        self.lrgraph = None
        self.compounds_components = None

    @property
    def is_trained(self):
        return self.lrgraph is not None

    def extract(
        self,
        train_data=None,
        min_noun_score=0.3,
        min_noun_frequency=1,
        min_num_of_features=1,
        min_eojeol_frequency=1,
        min_eojeol_is_noun_frequency=30,
        extract_compounds=True,
        exclude_syllables=False,
        exclude_numbers=True,
        custom_exclude_function=None
    ):
        if (not self.is_trained) and (train_data is None):
            raise ValueError('`train_data` must not be `None` if noun extractor has no LRGraph')

        if train_data is not None:
            self.lrgraph = train_lrgraph(
                train_data, min_eojeol_frequency,
                self.max_l_length, self.max_r_length, self.verbose)
        else:
            self.lrgraph.reset_lrgraph()

        candidates = prepare_noun_candidates(
            self.lrgraph, self.pos, min_noun_frequency,
            exclude_syllables, exclude_numbers, custom_exclude_function)
        nouns = longer_first_prediction(
            candidates, self.lrgraph, self.pos, self.neg,
            self.common, min_noun_score, min_num_of_features,
            min_eojeol_is_noun_frequency, self.verbose)
        nouns = {noun: score for noun, score in nouns.items() if score[1] >= min_noun_score}

        if extract_compounds:
            returns = extract_compounds_func(
                self.lrgraph, nouns, min_noun_frequency,
                min_noun_score, self.pos, self.verbose)
            compounds, self.compounds_components, self.compound_decomposer = returns
            nouns.update(compounds)

        features_to_be_detached = {r for r in self.pos}
        features_to_be_detached.update(self.common)
        nouns = postprocessing(nouns, self.lrgraph, features_to_be_detached, min_noun_score, self.verbose)

        self.lrgraph.reset_lrgraph()
        self.nouns = {noun: NounScore(frequency, score) for noun, (frequency, score) in nouns.items()}
        return self.nouns

    def decompose_compound(self, compound):
        """Decompose input `compound` into nouns if `compound` is true compound

        Args:
            compound (str) : input words

        Returns:
            tokens (list of str or None) : noun list if input is true compound

        Examples::
            >>> noun_extractor.decompose_compound('아이폰아이스크림아이비리그')
            $ ['아이폰', '아이스크림', '아이비리그']

            >>> noun_extractor.decompose_compound('아이폰아이스크림아이비리그봤다')
            $ None
        """
        if self.compound_decomposer is None:
            raise ValueError('[LRNounExtractor] retrain using `extract(extract_compounds=True)` first')
        tokens = self.compound_decomposer.tokenize(compound)
        for token in tokens:
            if token not in self.nouns:
                return None
        return tokens

    def predict(
        self,
        word,
        word_features=None,
        min_noun_score=0.3,
        min_num_of_features=1,
        min_eojeol_is_noun_frequency=30,
        debug=False
    ):
        """Predict noun scores

        Args:
            word (str) : input word; L-part
            word_features (list of str or None) : R parts
                When the value is `None`, it uses trained L-R graph.
            min_noun_score (float) :
                If the predicted score is less than `min_noun_score`,
                LRNounExtractor consider `word` is not Noun.
            min_num_of_features (int) :
                The number of active features used in prediction.
                When the number of features is too small, LRNounExtractor
                consider `word` is not Noun.
            min_eojeol_is_noun_frequency (int):
                Sometimes, especially in news domain, proper nouns appear alone in eojeol.
                ... 설명 더 적어야 함
            debug (Boolean) : If True, it shows classification details

        Returns:
            nonu_score (collections.namedtuple) : NounScore(frequency, score)

        Examples::
            >>> noun_extractor.predict('아이오아이')
            $ NounScore(frequency=127, score=1.0)

            >>> noun_extractor.predict('아이오아이', debug=True)
            $ OrderedDict([('word', '아이오아이'),
               ('pos', 87),
               ('common', 40),
               ('neg', 0),
               ('unk', 0),
               ('end', 0),
               ('num_features', 12),
               ('score', 1.0),
               ('support', 127)])
              NounScore(frequency=127, score=1.0)

            >>> word_features = [('의', 100), ('는', 50), ('니까', 15), ('가', 10), ('끼리', 5)]
            >>> noun_extractor.predict('아이오아이', word_features, debug=True)
            $ OrderedDict([('word', '아이오아이'),
               ('pos', 100),
               ('common', 50),
               ('neg', 0),
               ('unk', 5),
               ('end', 0),
               ('num_features', 1),
               ('score', 1.0),
               ('support', 150)])
              NounScore(frequency=150, score=0.967741935483871)
        """
        if word_features is None:
            word_features = self.lrgraph.get_r(word, -1)

        support, score = base_predict(
            word, word_features, self.pos, self.neg, self.common,
            min_noun_score, min_num_of_features, min_eojeol_is_noun_frequency, debug)
        return NounScore(support, score)


def prepare_r_features(pos_features=None, neg_features=None):
    """
    Check `pos_features` and `neg_features`
    If the argument is not defined, soynlp uses default R features

    Args:
        pos_features (collection of str)
        neg_features (collection of str)

    Returns:
        pos_features (set of str) : positive feature set excluding common features
        neg_features (set of str) : negative feature set excluding common features
        common_features (set of str) : feature appeared in both `pos_features` and `neg_features`
    """
    def load_features(path):
        with open(path, encoding='utf-8') as f:
            features = [line.strip() for line in f]
        features = {feature for feature in features if feature}
        return features

    default_feature_dir = f'{installpath}/pretrained_models/'

    if pos_features is None:
        pos_features = load_features(f'{default_feature_dir}/lrnounextractor.features.pos.v2')
    elif isinstance(pos_features, str) and (os.path.exists(pos_features)):
        pos_features = load_features(pos_features)

    if neg_features is None:
        neg_features = load_features(f'{default_feature_dir}/lrnounextractor.features.neg.v2')
    elif isinstance(neg_features, str) and (os.path.exists(neg_features)):
        neg_features = load_features(neg_features)

    if not isinstance(pos_features, set):
        pos_features = set(pos_features)
    if not isinstance(neg_features, set):
        neg_features = set(neg_features)

    common_features = pos_features.intersection(neg_features)
    pos_features = {feature for feature in pos_features if feature not in common_features}
    neg_features = {feature for feature in neg_features if feature not in common_features}
    return pos_features, neg_features, common_features


def print_message(message):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f'[LRNounExtractor] {now}, mem={get_process_memory():.4} GB : {message}')


def train_lrgraph(train_data, min_eojeol_frequency, max_l_length, max_r_length, verbose):
    if isinstance(train_data, LRGraph):
        if verbose:
            print_message('input is LRGraph')
        return train_data

    if isinstance(train_data, EojeolCounter):
        lrgraph = train_data.to_lrgraph(max_l_length, max_r_length)
        if verbose:
            print_message('transformed EojeolCounter to LRGraph')
        return lrgraph

    if isinstance(train_data, str) and os.path.exists(train_data):
        train_data = DoublespaceLineCorpus(train_data, iter_sent=True)

    eojeol_counter = EojeolCounter(
        sents=train_data,
        min_count=min_eojeol_frequency,
        max_length=(max_l_length + max_r_length),
        verbose=verbose
    )
    lrgraph = eojeol_counter.to_lrgraph(max_l_length, max_r_length)
    if verbose:
        print_message(f'finished building LRGraph from {len(eojeol_counter)} eojeols')
    return lrgraph


number_pattern = re.compile('[0-9]+')

def prepare_noun_candidates(lrgraph, pos_features, min_noun_frequency,
    exclude_syllables=False, exclude_numbers=True, custom_exclude_function=None):

    def include(word, e):
        return word[:e] == l_prefix

    def is_number(word):
        """
        Examples::
            >>> print(is_number('1234'))  # True
            >>> print(is_number('abc'))   # False
            >>> print(is_number('1234a')) # False
        """
        return number_pattern.sub('', word) == ''

    if custom_exclude_function is None:
        def func(x):
            return False
        custom_exclude_function = func  # all pass

    # noun candidates from positive featuers such as Josa
    N_from_J = {}
    for r in pos_features:
        for l, c in lrgraph.get_l(r, -1):
            if exclude_syllables and len(l) == 1:
                continue
            if exclude_numbers and is_number(l):
                continue
            if custom_exclude_function(l):
                continue
            N_from_J[l] = N_from_J.get(l, 0) + c
    N_from_J = {candidate for candidate, count in N_from_J.items() if count >= min_noun_frequency}
    return N_from_J


def longer_first_prediction(candidates, lrgraph, pos_features, neg_features,
    common_features, min_noun_score, min_num_of_features,
    min_eojeol_is_noun_frequency, verbose):

    sorted_candidates = sorted(candidates, key=lambda x: -len(x))
    if verbose:
        iterator = tqdm(sorted_candidates, desc='[LRNounExtractor] base prediction', total=len(candidates))
    else:
        iterator = sorted_candidates

    prediction_scores = {}
    for word in iterator:
        # base prediction
        word_features = lrgraph.get_r(word, -1)
        support, score = base_predict(
            word,
            word_features,
            pos_features,
            neg_features,
            common_features,
            min_noun_score,
            min_num_of_features,
            min_eojeol_is_noun_frequency
        )
        prediction_scores[word] = (support, score)

        # if predicted score is higher than `min_noun_score`, remove `L + R` from lrgraph
        if score >= min_noun_score:
            for r, count in word_features:
                # remove all eojeols that including word at left-side.
                lrgraph.remove_eojeol(word + r, count)
    return prediction_scores


def base_predict(word, word_features, pos_features, neg_features, common_features,
    min_noun_score=0.3, min_num_of_features=1, min_eojeol_is_noun_frequency=30, debug=False):  # noqa E125,E128

    refined_features, ambiguous_set = remove_ambiguous_features(
        word, word_features, pos_features, neg_features, common_features)

    pos, common, neg, unk, end = _base_predict(
        word, refined_features, pos_features, neg_features, common_features)

    denominator = pos + neg
    score = 0 if denominator == 0 else (pos - neg) / denominator
    support = (pos + end + common) if score >= min_noun_score else (neg + end + common)

    active_features = [r for r, _ in refined_features if (r in pos_features) or (r in neg_features)]
    num_features = len(active_features)

    if debug:
        pprint(OrderedDict(
            {'word': word, 'pos': pos, 'common': common, 'neg': neg, 'unk': unk,
             'end': end, 'num_features': num_features, 'score': score, 'support': support}
        ))

    if num_features > min_num_of_features:
        return support, score

    # when the number of features used in prediction is less than `min_num_of_features`
    sum_ = pos + common + neg + unk + end
    if sum_ == 0:
        return 0, 0

    if (end > min_eojeol_is_noun_frequency) and (pos >= neg):
        support = pos + common + end
        return support, support / sum_

    # 아이웨딩 + [('', 90), ('은', 3), ('측은', 1)] # `은`: common, `측은`: unknown, 대부분 단일어절
    if ((end > min_eojeol_is_noun_frequency) and (pos > neg)):
        support = pos + common + end
        return support, score

    # TODO: fix hard-coding `end / sum_ >= 0.3`
    # 아이엠텍 + [('은', 2), ('', 2)]
    if ((common > 0 or pos > 0) and  # noqa W504
        (end / sum_ >= 0.3) and      # noqa W504
        (common >= neg) and          # noqa W504
        (pos >= neg)):               # noqa W504
        support = pos + common + end
        return support, support / sum_

    # 경찰국 + [(은, 1), (에, 1), (에서, 1)] -> {은, 에}
    first_chars = {r[0] for r, _ in refined_features if (r and (r not in ambiguous_set))}
    # TODO: fix hard-coding `len(first_chars) >= 2`
    if len(first_chars) >= 2:
        support = pos + common + end
        return support, support / sum_

    # TODO: Handling for post-processing in NounExtractor
    # Case 1.
    #   아이러브영주사과 -> 아이러브영주사 + [(과,1)] (minimum r feature 적용해야 하는 케이스) : 복합명사
    #   아이러브영주사과 + [('', 1)] 이므로, 후처리 이후 '아이러브영주사' 후보에서 제외됨
    # Case 2.
    #   아이였으므로 -> 아이였으므 + [(로, 2)] (minimum r feature 적용)
    #   "명사 + Unknown R" 로 후처리
    return support, 0


def remove_ambiguous_features(word, word_features, pos_features, neg_features, common_features):
    def exist_longer_feature(word, r):
        for e in range(len(word) - 1, -1, -1):
            longer = word[e:] + r
            if (longer in pos_features) or (longer in neg_features) or (longer in common_features):
                return True
        return False

    def satisfy(word, r):
        if exist_longer_feature(word, r):
            # negative -다고, -자는
            # ('관계자' 의 경우 '관계 + 자는'으로 고려될 수 있음)
            return False
        return True

    refined = [r_freq for r_freq in word_features if satisfy(word, r_freq[0])]
    ambiguous = {r_freq[0] for r_freq in word_features if not satisfy(word, r_freq[0])}
    return refined, ambiguous


def _base_predict(word, word_features, pos_features, neg_features, common_features):
    pos, common, neg, unk, end = 0, 0, 0, 0, 0
    for r, freq in word_features:
        if not r:
            end += freq
            continue
        if r in common_features:
            common += freq
        elif r in pos_features:
            pos += freq
        elif r in neg_features:
            neg += freq
        else:
            unk += freq
    return pos, common, neg, unk, end


def extract_compounds_func(lrgraph, noun_scores,
    min_noun_frequency, min_noun_score, pos_features, verbose):

    candidates = {
        l: rdict.get('', 0) for l, rdict in lrgraph._lr_origin.items()
        if (len(l) >= 4) and (l not in noun_scores)}
    candidates = {l: count for l, count in candidates.items() if count >= min_noun_frequency}
    n = len(candidates)

    word_scores = {
        noun: len(noun) for noun, score in noun_scores.items()
        if score[1] > min_noun_score and len(noun) > 1}
    compound_decomposer = MaxScoreTokenizer(scores=word_scores)

    compounds_scores = {}
    compounds_counts = {}
    compounds_components = {}

    iterator = sorted(candidates.items(), key=lambda x: -len(x[0]))
    if verbose:
        iterator = tqdm(iterator, desc='[LRNounExtractor] extract compounds', total=n)

    for word, count in iterator:
        tokens = compound_decomposer.tokenize(word, flatten=False)[0]
        compound_parts = parse_compound(tokens, pos_features)
        if not compound_parts:
            continue

        # store compound components
        noun = ''.join(compound_parts)
        compounds_components[noun] = compound_parts

        # cumulate count and store compound score
        compound_score = max((noun_scores.get(t, (0, 0))[1] for t in compound_parts))
        compounds_scores[noun] = max(compounds_scores.get(noun, 0), compound_score)
        compounds_counts[noun] = compounds_counts.get(noun, 0) + count

        # discount frequency of substrings
        for e in range(2, len(word)):
            subword = word[:e]
            if subword not in candidates:
                continue
            candidates[subword] = candidates.get(subword, 0) - count

        # eojeol coverage
        lrgraph.remove_eojeol(word)

    compounds = {
        noun: (score, compounds_counts.get(noun, 0))
        for noun, score in compounds_scores.items()}

    if verbose:
        print_message(f'found {len(compounds)} compounds (min frequency={min_noun_frequency})')
    return compounds, compounds_components, compound_decomposer


def parse_compound(tokens, pos_features):
    """Check Noun* or Noun*Josa"""
    # format: (word, begin, end, score, length)
    # 점수는 단어의 길이이며, 0 점인 경우는 단어가 명사로 등록되지 않은 경우
    # 마지막 단어가 명사가 아니면 None
    for token in tokens[:-1]:
        if token[3] <= 0:
            return None

    # Noun* + Josa
    # 마지막 단어가 positive features 이고, 그 앞의 단어가 명사이면
    if (len(tokens) >= 3) and (tokens[-1][0] in pos_features) and (tokens[-2][3] > 0):
        return tuple(t[0] for t in tokens[:-1])

    # all tokens are noun
    # 앞의 단어는 미등록단어여도 마지막 단어가 명사이면 compounds
    # TODO: fix conditioin: 모든 단어의 점수가 0 이상인지 확인?
    if tokens[-1][3] > 0:
        return tuple(t[0] for t in tokens)

    # else, not compound
    return None


def postprocessing(nouns, lrgraph, features_to_be_detached, min_noun_score, verbose):
    num_before = len(nouns)
    nouns, removals = detaching_features(nouns, features_to_be_detached)
    if verbose:
        print_message(f'postprocessing: detaching_features: {num_before} -> {len(nouns)}')

    num_before = len(nouns)
    nouns, removals = ignore_features(nouns, features_to_be_detached)
    if verbose:
        print_message(f'postprocessing: ignore_features: {num_before} -> {len(nouns)}')

    num_before = len(nouns)
    nouns, removals = check_N_is_NJ(nouns, lrgraph)
    if verbose:
        print_message(f'postprocessing: check_N_is_NJ: {num_before} -> {len(nouns)}')

    return nouns
