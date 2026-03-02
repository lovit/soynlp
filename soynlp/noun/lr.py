import os
import re
from collections import OrderedDict, namedtuple
from datetime import datetime
from pprint import pprint

from tqdm import tqdm

from soynlp.tokenizer import MaxScoreTokenizer, NounMatchTokenizer
from soynlp.utils import DoublespaceLineCorpus, EojeolCounter, LRGraph, get_process_memory

from .postprocessing import check_N_is_NJ, detaching_features, ignore_features

installpath = os.path.abspath(os.path.dirname(__file__))
NounScore = namedtuple("NounScore", "frequency score")


class LRNounExtractor:
    """L-R graph based noun extractor

    Args:
        max_l_length (int) : maximum length of L in L-R graph
        max_r_length (int) : maximum length of R in L-R graph
        pos_features (set of str or None) :
            If None, it uses default positive features such as Josa in Korean
        neg_features (set of str or None) :
            If None, it uses default negative features such as Eomi in Korean(ending)
        verbose (Boolean) :
            If True, it shows progress
    """

    def __init__(
        self,
        max_l_length=10,
        max_r_length=9,
        pos_features=None,
        neg_features=None,
        verbose=True,
    ):
        self.max_l_length = max_l_length
        self.max_r_length = max_r_length
        self.verbose = verbose
        self.pos, self.neg, self.common = prepare_r_features(pos_features, neg_features)
        if verbose:
            print_message(f"#pos={len(self.pos)}, #neg={len(self.neg)}, #common={len(self.common)}")

        self.lrgraph: LRGraph | None = None
        self.compounds_components: dict | None = None
        self.compound_decomposer: MaxScoreTokenizer | None = None
        self.nouns: dict[str, NounScore] | None = None

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
        custom_exclude_function=None,
    ):
        """Extract nouns from `train_data` or trained L-R graph

        Args:
            train_data : Training input data.
            min_noun_score (float) : minimum noun score threshold
            min_noun_frequency (int) : minimum noun frequency
            min_num_of_features (int) : minimum number of active features
            min_eojeol_frequency (int) : minimum eojeol frequency
            min_eojeol_is_noun_frequency (int) : minimum frequency for eojeol-is-noun
            extract_compounds (Boolean) : If True, extracts compound nouns
            exclude_syllables (Boolean) : If True, excludes syllables
            exclude_numbers (Boolean) : If True, excludes numbers
            custom_exclude_function (callable or None) : custom exclude function

        Returns:
            nouns ({str: NounScore})
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

    def decompose_compound(self, compound):
        """Decompose input `compound` into nouns if `compound` is true compound"""
        if self.compound_decomposer is None:
            raise ValueError("[LRNounExtractor] retrain using `extract(extract_compounds=True)` first")
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
        debug=False,
    ):
        """Predict noun scores

        Args:
            word (str) : input word; L-part
            word_features (list of str or None) : R parts
            min_noun_score (float) : minimum noun score
            min_num_of_features (int) : minimum number of features
            min_eojeol_is_noun_frequency (int) : minimum eojeol frequency
            debug (Boolean) : If True, shows classification details

        Returns:
            noun_score (NounScore)
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

    def get_noun_tokenizer(self):
        """Get soynlp.tokenizer.NounMatchTokenizer using extracted nouns"""
        if not self.is_trained or self.nouns is None:
            raise RuntimeError("Train LRNounExtractor first. LRNounExtractor().extract(train-data)")
        noun_scores = {noun: score.score for noun, score in self.nouns.items()}
        return NounMatchTokenizer(noun_scores)


def prepare_r_features(pos_features=None, neg_features=None):
    """Check `pos_features` and `neg_features`

    Returns:
        pos_features (set of str)
        neg_features (set of str)
        common_features (set of str)
    """

    def load_features(path):
        with open(path, encoding="utf-8") as f:
            features = [line.strip() for line in f]
        features = {feature for feature in features if feature}
        return features

    default_feature_dir = f"{installpath}/pretrained_models/"

    if pos_features is None:
        pos_features = load_features(f"{default_feature_dir}/lrnounextractor.features.pos.v2")
    elif isinstance(pos_features, str) and (os.path.exists(pos_features)):
        pos_features = load_features(pos_features)

    if neg_features is None:
        neg_features = load_features(f"{default_feature_dir}/lrnounextractor.features.neg.v2")
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
    print(f"[LRNounExtractor] {now}, mem={get_process_memory():.4} GB : {message}")


def train_lrgraph(train_data, min_eojeol_frequency, max_l_length, max_r_length, verbose):
    if isinstance(train_data, LRGraph):
        if verbose:
            print_message("input is LRGraph")
        return train_data

    if isinstance(train_data, EojeolCounter):
        lrgraph = train_data.to_lrgraph(max_l_length, max_r_length)
        if verbose:
            print_message("transformed EojeolCounter to LRGraph")
        return lrgraph

    if isinstance(train_data, str) and os.path.exists(train_data):
        train_data = DoublespaceLineCorpus(train_data, iter_sent=True)

    eojeol_counter = EojeolCounter(
        sents=train_data,
        min_count=min_eojeol_frequency,
        max_length=(max_l_length + max_r_length),
        verbose=verbose,
    )
    lrgraph = eojeol_counter.to_lrgraph(max_l_length, max_r_length)
    if verbose:
        print_message(f"finished building LRGraph from {len(eojeol_counter)} eojeols")
    return lrgraph


number_pattern = re.compile("[0-9]+")


def prepare_noun_candidates(
    lrgraph, pos_features, min_noun_frequency, exclude_syllables=False, exclude_numbers=True, custom_exclude_function=None
):
    def is_number(word):
        return number_pattern.sub("", word) == ""

    if custom_exclude_function is None:

        def func(x):
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
    candidates,
    lrgraph,
    pos_features,
    neg_features,
    common_features,
    min_noun_score,
    min_num_of_features,
    min_eojeol_is_noun_frequency,
    verbose,
):
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
    word,
    word_features,
    pos_features,
    neg_features,
    common_features,
    min_noun_score=0.3,
    min_num_of_features=1,
    min_eojeol_is_noun_frequency=30,
    debug=False,
):
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
        pprint(
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


def remove_ambiguous_features(word, word_features, pos_features, neg_features, common_features):
    def exist_longer_feature(word, r):
        for e in range(len(word) - 1, -1, -1):
            longer = word[e:] + r
            if (longer in pos_features) or (longer in neg_features) or (longer in common_features):
                return True
        return False

    def satisfy(word, r):
        if exist_longer_feature(word, r):
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


def extract_compounds_func(lrgraph, noun_scores, min_noun_frequency, min_noun_score, pos_features, verbose):
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

    if verbose:
        print_message(f"found {len(compounds)} compounds (min frequency={min_noun_frequency})")
    return compounds, compounds_components, compound_decomposer


def parse_compound(tokens, pos_features):
    """Check Noun* or Noun*Josa"""
    for token in tokens[:-1]:
        if token[3] <= 0:
            return None

    if (len(tokens) >= 3) and (tokens[-1][0] in pos_features) and (tokens[-2][3] > 0):
        return tuple(t[0] for t in tokens[:-1])

    if tokens[-1][3] > 0:
        return tuple(t[0] for t in tokens)

    return None


def postprocessing(nouns, lrgraph, features_to_be_detached, min_noun_score, verbose):
    num_before = len(nouns)
    nouns, removals = detaching_features(nouns, features_to_be_detached)
    if verbose:
        print_message(f"postprocessing: detaching_features: {num_before} -> {len(nouns)}")

    num_before = len(nouns)
    nouns, removals = ignore_features(nouns, features_to_be_detached)
    if verbose:
        print_message(f"postprocessing: ignore_features: {num_before} -> {len(nouns)}")

    num_before = len(nouns)
    nouns, removals = check_N_is_NJ(nouns, lrgraph)
    if verbose:
        print_message(f"postprocessing: check_N_is_NJ: {num_before} -> {len(nouns)}")

    return nouns
