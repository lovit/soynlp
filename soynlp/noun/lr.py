import os
import re
from collections import namedtuple, OrderedDict
from datetime import datetime
from pprint import pprint
from tqdm import tqdm

from soynlp.tokenizer import MaxScoreTokenizer, NounMatchTokenizer
from soynlp.utils import DoublespaceLineCorpus, EojeolCounter, LRGraph, get_process_memory
from .postprocessing import detaching_features, ignore_features, check_N_is_NJ


installpath = os.path.abspath(os.path.dirname(__file__))
NounScore = namedtuple('NounScore', 'frequency score')


class LRNounExtractor():
    """L-R graph based noun extractor

    Args:
        max_l_length (int) : maximum length of L in L-R graph
        max_r_length (int) : maximum length of R in L-R graph
        pos_features (set of str or None) :
            If None, it uses default positive features such as Josa in Korean
            Or it provides customizing features set
        neg_features (set of str or None) :
            If None, it uses default negative features such as Eomi in Korean(ending)
            Or it provides customizing features set
        verbose (Boolean) :
            If True, it shows progress

    Examples::
        Train noun extractor model

            >>> from soynlp.noun import LRNounExtractor

            >>> # train_data = '../data/2016-10-20.txt'
            >>> train_data = 'path/to/train_text'
            >>> noun_extractor = LRNounExtractor()
            >>> nouns = noun_extractor.extract(train_data)

        Check extracted nouns

            >>> for noun in ['아이디', '아이디어', '아이오아이', '트와이스', '연합뉴스', '비선실세']:
            >>>    print(f'{noun} : {nouns.get(noun, None)}')
            $ 아이디 : NounScore(frequency=59, score=1.0)
              아이디어 : NounScore(frequency=142, score=1.0)
              아이오아이 : NounScore(frequency=127, score=1.0)
              트와이스 : NounScore(frequency=654, score=0.992831541218638)
              연합뉴스 : NounScore(frequency=4628, score=1.0)
              비선실세 : NounScore(frequency=66, score=1.0)

            >>> print(nouns['아이오아이'].frequency)
            $ 127

        Get noun tokenizer and use it

            >>> noun_tokenizer = noun_extractor.get_noun_tokenizer()
            >>> sentence = '네이버의 뉴스기사를 이용하여 학습한 모델예시입니다'
            >>> noun_tokenizer.tokenize(sentence)
            $ ['네이버', '뉴스기사', '이용', '학습', '모델예시']

            >>> noun_tokenizer.tokenize(sentence, concat_compound=False)
            $ ['네이버', '뉴스', '기사', '이용', '학습', '모델', '예시']
    """
    def __init__(
        self,
        max_l_length=10,
        max_r_length=9,
        pos_features=None,
        neg_features=None,
        verbose=True
    ):

        self.max_l_length = max_l_length
        self.max_r_length = max_r_length
        self.verbose = verbose
        self.pos, self.neg, self.common = prepare_r_features(pos_features, neg_features)
        if verbose:
            print_message(f'#pos={len(self.pos)}, #neg={len(self.neg)}, #common={len(self.common)}')

        self.lrgraph = None
        self.compounds_components = None
        self.nouns = None

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
        """Extract nouns from `train_data` or trained L-R graph

        Args:
            train_data (str,
                        list of str like,
                        soynlp.utils.DoublespaceLineCorpus,
                        soynlp.utils.EojeolCounter,
                        soynlp.utils.LRGraph) :
                Training input data.

                    >>> nouns = LRNounExtractor().extract('path/to/corpus')
                    >>> nouns = LRNounExtractor().extract(
                    >>>    soynlp.utils.DoublespaceLineCorpus('path/to/corpus'))

            min_noun_score (float) :
                If the predicted score is less than `min_noun_score`,
                LRNounExtractor consider `word` is not Noun.
            min_noun_frequency (int) :
                Required minimum frequency of noun candidates.
                It is used in finding noun candidates
            min_num_of_features (int) :
                The number of active features used in prediction.
                When the number of features is too small, LRNounExtractor
                consider `word` is not Noun.
            min_eojeol_frequency (int) :
                Required minimum frequency of eojeol.
                It is used in constructing L-R graph.
            min_eojeol_is_noun_frequency (int):
                Sometimes, especially in news domain, proper nouns appear alone in eojeol.
                TODO ... 설명 더 적어야 함
            extract_compounds (Boolean) :
                If True, it extracts compound nouns and train `self.compound_decomposer`.
            exclude_syllables (Boolean) :
                If True, it excludes syllables from noun candidates.
            exclude_numbers (Boolean) :
                If True, it excludes numbers such as '2016', '10' from noun candidates.
            custom_exclude_function (callable or None) :
                Custom exclude function. If you want to extract nouns of which suffix is '아이' then

                    >>> def custom_exclude_function(l):
                    >>>     return l[:2] != '아이'
                    >>>
                    >>> noun_extractor.extract(custom_exclude_function=custom_exclude_function)
                    $ {'아이폰7플러스': NounScore(frequency=8, score=1.0),
                       '아이카이스트랩': NounScore(frequency=18, score=1.0),
                       '아이콘트롤스': NounScore(frequency=8, score=1.0),
                       '아이엠벤쳐스': NounScore(frequency=10, score=1.0),
                       '아이피노믹스': NounScore(frequency=3, score=1.0),
                       '아이돌그룹': NounScore(frequency=16, score=1.0),
                       '아이덴티티': NounScore(frequency=25, score=1.0),
                         ... }

            debug (Boolean) :
                If True, it shows classification details.

        Returns:
            nouns ({str: namedtuple}) : {word: NonuScore}

        Note:
            LRNounExtractor 의 명사 추출 원리는 크게 두 가지 입니다.

            첫째, 명사의 오른쪽에는 조사의 등장 비율이 높고, 어미의 등장 비율이 낮습니다.
            `아이디어`는 명사이기 때문에 R parts 에 조사인 `-는`, `-의`. `-를` 와 함께 어절에 등장하여
            `아이디어 + 는`, `아이디어 + 의`, `아이디어 + 를` 을 이룹니다.
            `아이디어` 의 오른쪽에 명사의 오른쪽에 등장한 R parts 의 substring 의 비율이 높을수록 `아이디어`가
            명사일 점수가 높아집니다. 이 원리로 주어진 L=`아이디어`가 명사인지 판단하는 함수가
            `LRNounExtractor.predict()` 입니다. L 의 오른쪽에 등장하는 R 의 distribution 을 BOW 형태로
            입력하면 이를 바탕으로 L 의 명사 점수를 계산합니다.

                >>> noun_extractor = LRNounExtractor()
                >>> l = '아이오아이'
                >>> word_features = [('의', 100), ('는', 50), ('니까', 15), ('가', 10), ('끼리', 5)]
                >>> noun_extractor.predict(l, word_features)

            하지만 명사는 그 자체로 한 어절, `아이디어`를 이루기도 합니다. 그리고 `아이디` 역시 명사이기 때문에
            `-는`, `-의`. `-를` 과 함께 어절 `아이디 + 는`, `아이디 + 의`, `아이디 + 를`을 이룹니다.

            그러나 `-어` 는 `먹 + 어`, `넣 + 어`, `썰 + 어` 등에 등장하는 대표적인 어미입니다.
            `아이디어` 가 명사 자체로 어절을 이루는 경우가 많을 때는 `아이디` 에 `-어`가 결합된 것처럼 보일 수 있습니다.
            R parts 에서 조사들의 비율이 `-어` 보다 압도적으로 많다면 `아이디`도 명사로 추출될 수 있지만,
            이를 보장할 수는 없습니다.

            이러한 문제를 해결하기 위하여 `LRNounExtractor` 에서는 L-R graph 에서 길이가 긴 L 부터 명사유무를 판단한 다음,
            L 이 명사이면 `L + ?` 형태인 모든 어절을 L-R graph 에서 제거합니다.
            `아이디어` 가 명사로 판단되면 `아이디어`를 포함한 `아이디어 + ?`가 모두 지워지기 때문에 `아이디`의 R parts 에는
            `-어`를 제외한  `-는`, `-의`. `-를` 만 남아있어 `아이디` 도 명사로 추출됩니다.

            위의 과정은 `soynlp.noun.lr.longer_first_prediction()` 에 구현되어 있습니다.
        """
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
                TODO ... 설명 더 적어야 함
            debug (Boolean) :
                If True, it shows classification details

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

        support, score = predict_single_noun(
            word, word_features, self.pos, self.neg, self.common,
            min_noun_score, min_num_of_features, min_eojeol_is_noun_frequency, debug)
        return NounScore(support, score)

    def get_noun_tokenizer(self):
        """Get soynlp.tokenizer.NounMatchTokenizer using extracted nouns

        Examples::
            Train noun extractor model

                >>> from soynlp.noun import LRNounExtractor
                >>> train_data = '../data/2016-10-20.txt'
                >>> noun_extractor = LRNounExtractor()
                >>> _ = noun_extractor.extract(train_data)

            Get noun tokenizer and use it

                >>> noun_tokenizer = noun_extractor.get_noun_tokenizer()
                >>> sentence = '네이버의 뉴스기사를 이용하여 학습한 모델예시입니다'
                >>> noun_tokenizer.tokenize(sentence)
                $ ['네이버', '뉴스기사', '이용', '학습', '모델예시']

                >>> noun_tokenizer.tokenize(sentence, concat_compound=False)
                $ ['네이버', '뉴스', '기사', '이용', '학습', '모델', '예시']

                >>> noun_tokenizer.tokenize(sentence, return_words=False)
                $ [Token(네이버, score=1.0, position=(0, 3), eojeol_id=0),
                   Token(뉴스기사, score=0.972972972972973, position=(5, 9), eojeol_id=1),
                   Token(이용, score=0.9323344610923151, position=(11, 13), eojeol_id=2),
                   Token(학습, score=0.9253731343283582, position=(16, 18), eojeol_id=3),
                   Token(모델예시, score=1.0, position=(20, 24), eojeol_id=4)]
        """
        if not self.is_trained:
            raise RuntimeError('Train LRNounExtractor firts. LRNonuExtractor().extract(train-data)')
        noun_scores = {noun: score.score for noun, score in self.nouns.items()}
        return NounMatchTokenizer(noun_scores)


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
        support, score = predict_single_noun(
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
        # `아이디어` 와 `아이디` 를 예시로 든 작동 원리를 LRNounExtractor.extract() 의 docstring 에 적어뒀습니다.
        if score >= min_noun_score:
            for r, count in word_features:
                # remove all eojeols that including word at left-side.
                lrgraph.remove_eojeol(word + r, count)
    return prediction_scores


def predict_single_noun(word, word_features, pos_features, neg_features, common_features,
    min_noun_score=0.3, min_num_of_features=1, min_eojeol_is_noun_frequency=30, debug=False):  # noqa E125,E128

    refined_features, ambiguous_set = remove_ambiguous_features(
        word, word_features, pos_features, neg_features, common_features)

    pos, common, neg, unk, end = check_r_features(
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


def check_r_features(word, word_features, pos_features, neg_features, common_features):
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
        tokens = compound_decomposer.tokenize(word, return_words=False)
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
