import logging
import os
import re
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pprint import pformat

from tqdm import tqdm

from soynlp.tokenizer import MaxScoreTokenizer, NounMatchTokenizer, Token
from soynlp.utils import CorpusLoader, EojeolCounter, LRGraph

from .postprocessing import check_N_is_NJ, detaching_features, ignore_features

logger = logging.getLogger(__name__)

installpath = os.path.abspath(os.path.dirname(__file__))


@dataclass(slots=True)
class NounScore:
    frequency: int
    score: float


class LRNounExtractor:
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
        max_l_length: int = 10,
        max_r_length: int = 9,
        pos_features: set[str] | Iterable[str] | str | None = None,
        neg_features: set[str] | Iterable[str] | str | None = None,
        verbose: bool = True,
    ) -> None:
        self.max_l_length = max_l_length
        self.max_r_length = max_r_length
        self.verbose = verbose
        self.pos, self.neg, self.common = prepare_r_features(pos_features, neg_features)
        logger.info(f"#pos={len(self.pos)}, #neg={len(self.neg)}, #common={len(self.common)}")

        self.lrgraph: LRGraph | None = None
        self.compounds_components: dict | None = None
        self.compound_decomposer: MaxScoreTokenizer | None = None
        self.nouns: dict[str, NounScore] | None = None

    @property
    def is_trained(self) -> bool:
        return self.lrgraph is not None

    def extract(
        self,
        train_data: str | list[str] | CorpusLoader | EojeolCounter | LRGraph | None = None,
        min_noun_score: float = 0.3,
        min_noun_frequency: int = 1,
        min_num_of_features: int = 1,
        min_eojeol_frequency: int = 1,
        min_eojeol_is_noun_frequency: int = 30,
        extract_compounds: bool = True,
        exclude_syllables: bool = False,
        exclude_numbers: bool = True,
        custom_exclude_function: Callable[[str], bool] | None = None,
    ) -> dict[str, NounScore]:
        """Extract nouns from `train_data` or trained L-R graph

        Args:
            train_data (str,
                        list of str like,
                        soynlp.utils.CorpusLoader,
                        soynlp.utils.EojeolCounter,
                        soynlp.utils.LRGraph) :
                Training input data.

                    >>> nouns = LRNounExtractor().extract('path/to/corpus.jsonl')
                    >>> nouns = LRNounExtractor().extract(
                    >>>    soynlp.utils.CorpusLoader('path/to/corpus.jsonl', format='jsonl'))

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
            min_eojeol_is_noun_frequency (int) :
                Sometimes, especially in news domain, proper nouns appear alone in eojeol.
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
                       '아이돌그룹': NounScore(frequency=16, score=1.0),
                       '아이덴티티': NounScore(frequency=25, score=1.0),
                         ... }

        Returns:
            nouns ({str: NounScore}) : {word: NounScore}

        Note:
            LRNounExtractor 의 명사 추출 원리는 크게 두 가지 입니다.

            첫째, 명사의 오른쪽에는 조사의 등장 비율이 높고, 어미의 등장 비율이 낮습니다.
            `아이디어`는 명사이기 때문에 R parts 에 조사인 `-는`, `-의`. `-를` 와 함께 어절에 등장하여
            `아이디어 + 는`, `아이디어 + 의`, `아이디어 + 를` 을 이룹니다.

            이 원리로 주어진 L=`아이디어`가 명사인지 판단하는 함수가
            `LRNounExtractor.predict()` 입니다. L 의 오른쪽에 등장하는 R 의 distribution 을 BOW 형태로
            입력하면 이를 바탕으로 L 의 명사 점수를 계산합니다.

                >>> noun_extractor = LRNounExtractor()
                >>> l = '아이오아이'
                >>> word_features = [('의', 100), ('는', 50), ('니까', 15), ('가', 10), ('끼리', 5)]
                >>> noun_extractor.predict(l, word_features)

            둘째, L-R graph 에서 길이가 긴 L 부터 명사유무를 판단한 다음,
            L 이 명사이면 `L + ?` 형태인 모든 어절을 L-R graph 에서 제거합니다.
            `아이디어` 가 명사로 판단되면 `아이디어 + ?`가 모두 지워지기 때문에 `아이디`의 R parts 에는
            `-어`를 제외한 `-는`, `-의`. `-를` 만 남아있어 `아이디` 도 명사로 추출됩니다.

            위의 과정은 `soynlp.noun.lr.longer_first_prediction()` 에 구현되어 있습니다.
        """
        if (not self.is_trained) and (train_data is None):
            raise ValueError("`train_data` must not be `None` if noun extractor has no LRGraph")

        if train_data is not None:
            self.lrgraph = train_lrgraph(train_data, min_eojeol_frequency, self.max_l_length, self.max_r_length, self.verbose)
        else:
            if self.lrgraph is None:
                raise ValueError("`train_data` must not be `None` if noun extractor has no LRGraph")
            self.lrgraph.reset_lrgraph()

        lrgraph = self.lrgraph  # guaranteed non-None after above block
        candidates = prepare_noun_candidates(
            lrgraph, self.pos, min_noun_frequency, exclude_syllables, exclude_numbers, custom_exclude_function
        )
        nouns = longer_first_prediction(
            candidates,
            lrgraph,
            self.pos,
            self.neg,
            self.common,
            min_noun_score,
            min_num_of_features,
            min_eojeol_is_noun_frequency,
            self.verbose,
        )
        nouns = {noun: score for noun, score in nouns.items() if score[1] >= min_noun_score}

        if extract_compounds:
            returns = extract_compounds_func(lrgraph, nouns, min_noun_frequency, min_noun_score, self.pos, self.verbose)
            compounds, self.compounds_components, self.compound_decomposer = returns
            nouns.update(compounds)

        features_to_be_detached = {r for r in self.pos}
        features_to_be_detached.update(self.common)
        nouns = postprocessing(nouns, lrgraph, features_to_be_detached, min_noun_score, self.verbose)

        lrgraph.reset_lrgraph()
        self.nouns = {noun: NounScore(frequency, score) for noun, (frequency, score) in nouns.items()}
        return self.nouns

    def decompose_compound(self, compound: str) -> list[str] | None:
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
            raise ValueError("[LRNounExtractor] retrain using `extract(extract_compounds=True)` first")
        tokens = self.compound_decomposer.tokenize(compound)
        for token in tokens:
            if token not in self.nouns:
                return None
        return tokens

    def predict(
        self,
        word: str,
        word_features: list[tuple[str, int]] | None = None,
        min_noun_score: float = 0.3,
        min_num_of_features: int = 1,
        min_eojeol_is_noun_frequency: int = 30,
        debug: bool = False,
    ) -> NounScore:
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
            min_eojeol_is_noun_frequency (int) :
                Sometimes, especially in news domain, proper nouns appear alone in eojeol.
            debug (Boolean) :
                If True, it shows classification details

        Returns:
            noun_score (NounScore) : NounScore(frequency, score)

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
            if self.lrgraph is None:
                raise ValueError("Train LRNounExtractor first")
            word_features = self.lrgraph.get_r(word, -1)

        support, score = predict_single_noun(
            word,
            word_features,
            self.pos,
            self.neg,
            self.common,
            min_noun_score,
            min_num_of_features,
            min_eojeol_is_noun_frequency,
            debug,
        )
        return NounScore(support, score)

    def get_noun_tokenizer(self) -> NounMatchTokenizer:
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
        """
        if not self.is_trained or self.nouns is None:
            raise RuntimeError("Train LRNounExtractor first. LRNounExtractor().extract(train-data)")
        noun_scores = {noun: score.score for noun, score in self.nouns.items()}
        return NounMatchTokenizer(noun_scores)


def _load_features(path: str) -> set[str]:
    """파일에서 특징(feature) 목록을 읽어 집합으로 반환하는 내부 함수"""
    with open(path, encoding="utf-8") as f:
        features = [line.strip() for line in f]
    features = {feature for feature in features if feature}
    return features


def prepare_r_features(
    pos_features: set[str] | Iterable[str] | str | None = None,
    neg_features: set[str] | Iterable[str] | str | None = None,
) -> tuple[set[str], set[str], set[str]]:
    """Check `pos_features` and `neg_features`
    If the argument is not defined, soynlp uses default R features

    Args:
        pos_features (collection of str)
        neg_features (collection of str)

    Returns:
        pos_features (set of str) : positive feature set excluding common features
        neg_features (set of str) : negative feature set excluding common features
        common_features (set of str) : feature appeared in both `pos_features` and `neg_features`
    """
    default_feature_dir = f"{installpath}/pretrained_models/"

    if pos_features is None:
        pos_features = _load_features(f"{default_feature_dir}/lrnounextractor.features.pos.v2")
    elif isinstance(pos_features, str) and (os.path.exists(pos_features)):
        pos_features = _load_features(pos_features)

    if neg_features is None:
        neg_features = _load_features(f"{default_feature_dir}/lrnounextractor.features.neg.v2")
    elif isinstance(neg_features, str) and (os.path.exists(neg_features)):
        neg_features = _load_features(neg_features)

    if not isinstance(pos_features, set):
        pos_features = set(pos_features)
    if not isinstance(neg_features, set):
        neg_features = set(neg_features)

    common_features = pos_features.intersection(neg_features)
    pos_features = {feature for feature in pos_features if feature not in common_features}
    neg_features = {feature for feature in neg_features if feature not in common_features}
    return pos_features, neg_features, common_features


def train_lrgraph(
    train_data: str | list[str] | CorpusLoader | EojeolCounter | LRGraph,
    min_eojeol_frequency: int,
    max_l_length: int,
    max_r_length: int,
    verbose: bool,
) -> LRGraph:
    if isinstance(train_data, LRGraph):
        logger.info("input is LRGraph")
        return train_data

    if isinstance(train_data, EojeolCounter):
        lrgraph = train_data.to_lrgraph(max_l_length, max_r_length)
        logger.info("transformed EojeolCounter to LRGraph")
        return lrgraph

    if isinstance(train_data, str) and os.path.exists(train_data):
        fmt = "jsonl" if train_data.endswith(".jsonl") else "text"
        train_data = CorpusLoader(train_data, format=fmt)

    eojeol_counter = EojeolCounter(
        sents=train_data,
        min_count=min_eojeol_frequency,
        max_length=(max_l_length + max_r_length),
        verbose=verbose,
    )
    lrgraph = eojeol_counter.to_lrgraph(max_l_length, max_r_length)
    logger.info(f"finished building LRGraph from {len(eojeol_counter)} eojeols")
    return lrgraph


number_pattern = re.compile(r"[0-9]+")


def prepare_noun_candidates(
    lrgraph: LRGraph,
    pos_features: set[str],
    min_noun_frequency: int,
    exclude_syllables: bool = False,
    exclude_numbers: bool = True,
    custom_exclude_function: Callable[[str], bool] | None = None,
) -> set[str]:
    def is_number(word: str) -> bool:
        return number_pattern.sub("", word) == ""

    if custom_exclude_function is None:

        def func(x: str) -> bool:
            return False

        custom_exclude_function = func

    N_from_J = {}
    for r in pos_features:
        for l, c in lrgraph.get_l(r, -1):  # noqa: E741
            if exclude_syllables and len(l) == 1:
                continue
            if exclude_numbers and is_number(l):
                continue
            if custom_exclude_function(l):
                continue
            N_from_J[l] = N_from_J.get(l, 0) + c
    N_from_J = {candidate for candidate, count in N_from_J.items() if count >= min_noun_frequency}
    return N_from_J


def longer_first_prediction(
    candidates: set[str],
    lrgraph: LRGraph,
    pos_features: set[str],
    neg_features: set[str],
    common_features: set[str],
    min_noun_score: float,
    min_num_of_features: int,
    min_eojeol_is_noun_frequency: int,
    verbose: bool,
) -> dict[str, tuple[int, float]]:
    sorted_candidates = sorted(candidates, key=lambda x: -len(x))
    if verbose:
        iterator = tqdm(sorted_candidates, desc="[LRNounExtractor] base prediction", total=len(candidates))
    else:
        iterator = sorted_candidates

    prediction_scores = {}
    for word in iterator:
        word_features = lrgraph.get_r(word, -1)
        support, score = predict_single_noun(
            word,
            word_features,
            pos_features,
            neg_features,
            common_features,
            min_noun_score,
            min_num_of_features,
            min_eojeol_is_noun_frequency,
        )
        prediction_scores[word] = (support, score)

        if score >= min_noun_score:
            for r, count in word_features:
                lrgraph.remove_eojeol(word + r, count)
    return prediction_scores


def predict_single_noun(
    word: str,
    word_features: list[tuple[str, int]],
    pos_features: set[str],
    neg_features: set[str],
    common_features: set[str],
    min_noun_score: float = 0.3,
    min_num_of_features: int = 1,
    min_eojeol_is_noun_frequency: int = 30,
    debug: bool = False,
) -> tuple[int, float]:
    refined_features, ambiguous_set = remove_ambiguous_features(
        word, word_features, pos_features, neg_features, common_features
    )

    pos, common, neg, unk, end = check_r_features(word, refined_features, pos_features, neg_features, common_features)

    denominator = pos + neg
    score = 0 if denominator == 0 else (pos - neg) / denominator
    support = (pos + end + common) if score >= min_noun_score else (neg + end + common)

    active_features = [r for r, _ in refined_features if (r in pos_features) or (r in neg_features)]
    num_features = len(active_features)

    if debug:
        logger.debug(
            pformat(
                OrderedDict(
                    {
                        "word": word,
                        "pos": pos,
                        "common": common,
                        "neg": neg,
                        "unk": unk,
                        "end": end,
                        "num_features": num_features,
                        "score": score,
                        "support": support,
                    }
                )
            )
        )

    if num_features > min_num_of_features:
        return support, score

    sum_ = pos + common + neg + unk + end
    if sum_ == 0:
        return 0, 0

    if (end > min_eojeol_is_noun_frequency) and (pos >= neg):
        support = pos + common + end
        return support, support / sum_

    if (end > min_eojeol_is_noun_frequency) and (pos > neg):
        support = pos + common + end
        return support, score

    if (common > 0 or pos > 0) and (end / sum_ >= 0.3) and (common >= neg) and (pos >= neg):
        support = pos + common + end
        return support, support / sum_

    first_chars = {r[0] for r, _ in refined_features if (r and (r not in ambiguous_set))}
    if len(first_chars) >= 2:
        support = pos + common + end
        return support, support / sum_

    return support, 0


def remove_ambiguous_features(
    word: str,
    word_features: list[tuple[str, int]],
    pos_features: set[str],
    neg_features: set[str],
    common_features: set[str],
) -> tuple[list[tuple[str, int]], set[str]]:
    def exist_longer_feature(word: str, r: str) -> bool:
        for e in range(len(word) - 1, -1, -1):
            longer = word[e:] + r
            if (longer in pos_features) or (longer in neg_features) or (longer in common_features):
                return True
        return False

    def satisfy(word: str, r: str) -> bool:
        if exist_longer_feature(word, r):
            return False
        return True

    refined = [r_freq for r_freq in word_features if satisfy(word, r_freq[0])]
    ambiguous = {r_freq[0] for r_freq in word_features if not satisfy(word, r_freq[0])}
    return refined, ambiguous


def check_r_features(
    word: str,
    word_features: list[tuple[str, int]],
    pos_features: set[str],
    neg_features: set[str],
    common_features: set[str],
) -> tuple[int, int, int, int, int]:
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


def extract_compounds_func(
    lrgraph: LRGraph,
    noun_scores: dict[str, tuple[int, float]],
    min_noun_frequency: int,
    min_noun_score: float,
    pos_features: set[str],
    verbose: bool,
) -> tuple[dict[str, tuple[int, float]], dict[str, tuple[str, ...]], MaxScoreTokenizer]:
    candidates = {
        l: rdict.get("", 0)
        for l, rdict in lrgraph._lr_origin.items()  # noqa: E741
        if (len(l) >= 4) and (l not in noun_scores)
    }
    candidates = {l: count for l, count in candidates.items() if count >= min_noun_frequency}  # noqa: E741
    n = len(candidates)

    word_scores = {noun: len(noun) for noun, score in noun_scores.items() if score[1] > min_noun_score and len(noun) > 1}
    compound_decomposer = MaxScoreTokenizer(scores=word_scores)

    compounds_scores = {}
    compounds_counts = {}
    compounds_components = {}

    iterator = sorted(candidates.items(), key=lambda x: -len(x[0]))
    if verbose:
        iterator = tqdm(iterator, desc="[LRNounExtractor] extract compounds", total=n)

    for word, count in iterator:
        tokens = compound_decomposer.tokenize(word, return_words=False)
        compound_parts = parse_compound(tokens, pos_features)
        if not compound_parts:
            continue

        noun = "".join(compound_parts)
        compounds_components[noun] = compound_parts

        compound_score = max((noun_scores.get(t, (0, 0))[1] for t in compound_parts))
        compounds_scores[noun] = max(compounds_scores.get(noun, 0), compound_score)
        compounds_counts[noun] = compounds_counts.get(noun, 0) + count

        for e in range(2, len(word)):
            subword = word[:e]
            if subword not in candidates:
                continue
            candidates[subword] = candidates.get(subword, 0) - count

        lrgraph.remove_eojeol(word)

    compounds = {noun: (compounds_counts.get(noun, 0), score) for noun, score in compounds_scores.items()}

    logger.info(f"found {len(compounds)} compounds (min frequency={min_noun_frequency})")
    return compounds, compounds_components, compound_decomposer


def parse_compound(tokens: list[Token], pos_features: set[str]) -> tuple[str, ...] | None:
    """Check Noun* or Noun*Josa"""
    for token in tokens[:-1]:
        if token.score <= 0:
            return None

    if (len(tokens) >= 3) and (tokens[-1].word in pos_features) and (tokens[-2].score > 0):
        return tuple(t.word for t in tokens[:-1])

    if tokens[-1].score > 0:
        return tuple(t.word for t in tokens)

    return None


def postprocessing(
    nouns: dict[str, tuple[int, float]],
    lrgraph: LRGraph,
    features_to_be_detached: set[str],
    min_noun_score: float,
    verbose: bool,
) -> dict[str, tuple[int, float]]:
    num_before = len(nouns)
    nouns, removals = detaching_features(nouns, features_to_be_detached)
    logger.info(f"postprocessing: detaching_features: {num_before} -> {len(nouns)}")

    num_before = len(nouns)
    nouns, removals = ignore_features(nouns, features_to_be_detached)
    logger.info(f"postprocessing: ignore_features: {num_before} -> {len(nouns)}")

    num_before = len(nouns)
    nouns, removals = check_N_is_NJ(nouns, lrgraph)
    logger.info(f"postprocessing: check_N_is_NJ: {num_before} -> {len(nouns)}")

    return nouns
