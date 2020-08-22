import os
from collections import namedtuple, OrderedDict
from datetime import datetime
from pprint import pprint
from tqdm import tqdm

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
        postprocessing=None,
        verbose=True,
        debug_dir=None,
    ):

        self.max_l_length = max_l_length
        self.max_r_length = max_r_length
        self.verbose = verbose
        self.debug_dir = debug_dir
        self.pos, self.neg, self.common = prepare_r_features(pos_features, neg_features)
        self.postprocessing = prepare_postprocessing(postprocessing)

        self.lrgraph = None

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
        l_prefix=None
    ):
        if (not self.is_trained) and (train_data is None):
            raise ValueError('`train_data` must not be `None` if noun extractor has no LRGraph')

        if self.lrgraph is None:
            self.lrgraph = train_lrgraph(
                train_data, min_eojeol_frequency,
                self.max_l_length, self.max_r_length, self.verbose)

        candidates = prepare_noun_candidates(
            self.lrgraph, self.pos, min_noun_frequency, l_prefix)
        nouns = longer_first_prediction(
            candidates, self.lrgraph, self.pos, self.neg,
            self.common, min_noun_score, min_num_of_features,
            min_eojeol_is_noun_frequency, self.verbose)

        # TODO
        if extract_compounds:
            nouns = extract_compounds_func(candidates, nouns, self.verbose)

        # TODO: check
        features_to_be_detached = {r for r in self.pos}
        features_to_be_detached.update(self.common)
        nouns = postprocessing(nouns, self.lrgraph, features_to_be_detached, self.verbose)
        nouns = {noun: NounScore(frequency, score) for noun, (frequency, score) in nouns.items()}
        return nouns


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


def prepare_postprocessing(postprocessing):
    # NotImplemented
    return postprocessing


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


def prepare_noun_candidates(lrgraph, pos_features, min_noun_frequency, l_prefix=None):
    def include(word, e):
        return word[:e] == l_prefix

    # noun candidates from positive featuers such as Josa
    N_from_J = {}
    for r in pos_features:
        for l, c in lrgraph.get_l(r, -1):
            if not l_prefix:
                N_from_J[l] = N_from_J.get(l, 0) + c
                continue
            # for debugging
            if include(l, len(l_prefix)):
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


def extract_compounds_func(candidates, nouns, verbose):
    raise NotImplementedError


def postprocessing(nouns, lrgraph, features_to_be_detached, verbose):
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