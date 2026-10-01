import logging
from collections import namedtuple
from pathlib import Path

from soynlp._legacy.tokenizer import MaxScoreTokenizer
from soynlp._legacy.utils import EojeolCounter, LRGraph, check_corpus, check_dirs, get_process_memory
from soynlp.core import LRGraph as CorpusLRGraph
from soynlp.normalizer import normalize_sent_for_lrgraph
from soynlp.utils.utils import EojeolCounter as CorpusEojeolCounter

from ._josa import extract_domain_pos_features
from ._noun_postprocessing import check_N_is_NJ, detaching_features, ignore_features


def _log(*args, **kwargs):
    logging.getLogger(__name__).info(" ".join(map(str, args)))


NounScore = namedtuple("NounScore", "frequency score")


class LRNounExtractor_v2:
    def __init__(
        self,
        max_left_length=10,
        max_right_length=9,
        predictor_headers=None,
        verbose=True,
        min_num_of_features=1,
        max_frequency_when_noun_is_eojeol=30,
        eojeol_counter_filtering_checkpoint=500000,
        extract_compound=True,
        extract_pos_feature=False,
        extract_determiner=False,
        ensure_normalized=False,
        postprocessing=None,
        logpath=None,
    ):
        self.max_left_length = max_left_length
        self.max_right_length = max_right_length
        self.lrgraph = None
        self.verbose = verbose
        self.min_num_of_features = min_num_of_features
        self.max_frequency_when_noun_is_eojeol = max_frequency_when_noun_is_eojeol
        self.eojeol_counter_filtering_checkpoint = eojeol_counter_filtering_checkpoint
        self.extract_compound = extract_compound
        self.extract_pos_feature = extract_pos_feature
        self.extract_determiner = extract_determiner
        self.ensure_normalized = ensure_normalized
        self.logpath = logpath
        if logpath:
            check_dirs(logpath)
        if not postprocessing:
            postprocessing = ["detaching_features", "ignore_features", "ignore_NJ"]
        elif isinstance(postprocessing, str):
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
            _log("[Noun Extractor] use default predictors")
        dirname = str(Path(__file__).resolve().parents[2])
        predictor_header = [f"{dirname}/trained_models/noun_predictor_ver2"]
        return predictor_header

    def _load_predictor(self, headers):
        if isinstance(headers, str):
            headers = [headers]
        pos, neg = (set(), set())
        for header in headers:
            pos_path = f"{header}_pos"
            with open(pos_path, encoding="utf-8") as f:
                pos.update({feature.strip() for feature in f})
            neg_path = f"{header}_neg"
            with open(neg_path, encoding="utf-8") as f:
                neg.update({feature.strip() for feature in f})
        common = pos.intersection(neg)
        pos = {feature for feature in pos if feature not in common}
        neg = {feature for feature in neg if feature not in common}
        if self.verbose:
            _log(f"[Noun Extractor] num features: pos={len(pos)}, neg={len(neg)}, common={len(common)}")
        self._pos_features = pos
        self._neg_features = neg
        self._common_features = common

    def _append_features(self, feature_type, features):

        def check_feature_size():
            return (len(self._pos_features), len(self._neg_features), len(self._common_features))

        n_pos, n_neg, n_common = check_feature_size()
        if feature_type == "pos":
            commons = {f for f in features if f in self._neg_features}
            self._pos_features.update({f for f in features if f not in commons})
        elif feature_type == "neg":
            commons = {f for f in features if f in self._pos_features}
            self._neg_features.update({f for f in features if f not in commons})
        elif feature_type == "common":
            commons = features
        else:
            raise ValueError("Feature type was wrong. Choice = [pos, neg, common]")
        self._common_features.update(commons)
        n_pos_, n_neg_, n_common_ = check_feature_size()
        if self.verbose:
            message = f"pos={n_pos} -> {n_pos_}, neg={n_neg} -> {n_neg_}, common={n_common} -> {n_common_}"
            _log(f"[Noun Extractor] features appended. {message}")

    def train_extract(self, inputs, min_noun_score=0.3, min_noun_frequency=1, min_eojeol_frequency=1, reset_lrgraph=True):
        self.train(inputs, min_eojeol_frequency)
        return self.extract(min_noun_score, min_noun_frequency, reset_lrgraph)

    def train(self, inputs, min_eojeol_frequency=1):
        if isinstance(inputs, CorpusLRGraph):
            inputs = LRGraph(dict(inputs.iter_original_lr()), l_max_length=inputs.max_l_length, r_max_length=inputs.max_r_length)
        if isinstance(inputs, LRGraph):
            self._train_with_lrgraph(inputs)
        elif isinstance(inputs, (EojeolCounter, CorpusEojeolCounter)):
            self._train_with_eojeol_counter(inputs)
        else:
            self._train_with_sentences(inputs, min_eojeol_frequency)

    def _train_with_sentences(self, sentences, min_eojeol_frequency=1):
        if self.verbose:
            _log("[Noun Extractor] counting eojeols")
        if not self.ensure_normalized:

            def preprocess(sent: str) -> str:
                return sent
        else:
            preprocess = normalize_sent_for_lrgraph
        check_corpus(sentences)
        eojeol_counter = EojeolCounter(
            sentences,
            min_count=min_eojeol_frequency,
            max_length=self.max_left_length + self.max_right_length,
            filtering_checkpoint=self.eojeol_counter_filtering_checkpoint,
            verbose=self.verbose,
            preprocess=preprocess,
        )
        self._train_with_eojeol_counter(eojeol_counter)

    def _train_with_eojeol_counter(self, eojeol_counter):
        lrgraph = eojeol_counter.to_lrgraph(self.max_left_length, self.max_right_length)
        num_of_eojeols = eojeol_counter._count_sum
        if self.verbose:
            _log("[Noun Extractor] complete eojeol counter -> lr graph")
        self._train_with_lrgraph(lrgraph, num_of_eojeols)

    def _train_with_lrgraph(self, lrgraph, num_of_eojeols=-1):
        self.lrgraph = lrgraph
        self._num_of_covered_eojeols = 0
        if num_of_eojeols == -1:
            num_of_eojeols = lrgraph.to_EojeolCounter()._count_sum
        self._num_of_eojeols = num_of_eojeols
        if self.verbose:
            _log(
                "[Noun Extractor] has been trained. #eojeols={}, mem={} Gb".format(
                    num_of_eojeols, "%.3f" % get_process_memory()
                )
            )

    def _extract_determiner(self):
        raise NotImplementedError

    def extract_domain_pos_features(
        self,
        noun_candidates=None,
        ignore_features=None,
        append_extracted_features=True,
        min_noun_score=0.3,
        min_noun_frequency=100,
        min_pos_score=0.3,
        min_pos_feature_frequency=1000,
        min_num_of_unique_lastchar=4,
        min_entropy_of_lastchar=0.5,
        min_noun_entropy=1.5,
    ):
        if self.lrgraph is None:
            raise ValueError("train() must be called before extraction")
        if self.verbose:
            _log("[Noun Extractor] batch prediction for extracting pos feature")
        if not noun_candidates:
            noun_candidates = self._noun_candidates_from_positive_features()
        prediction_scores = self._batch_predicting_nouns(noun_candidates, min_noun_score)
        self.lrgraph.reset_lrgraph()
        self._pos_features_extracted = extract_domain_pos_features(
            prediction_scores,
            self.lrgraph,
            self._pos_features,
            ignore_features,
            min_noun_score,
            min_noun_frequency,
            min_pos_score,
            min_pos_feature_frequency,
            min_num_of_unique_lastchar,
            min_entropy_of_lastchar,
            min_noun_entropy,
        )
        if append_extracted_features:
            self._append_features("pos", self._pos_features_extracted)
        if self.verbose:
            _log(f"[Noun Extractor] {len(self._pos_features_extracted)} pos features were extracted")

    def extract(self, min_noun_score=0.3, min_noun_frequency=1, reset_lrgraph=True):
        if self.lrgraph is None:
            raise ValueError("train() must be called before extraction")
        self._num_of_covered_eojeols = 0
        noun_candidates = self._noun_candidates_from_positive_features()
        if self.extract_pos_feature:
            if self.verbose:
                _log("[Noun Extractor] extract and append pos features")
            self.extract_domain_pos_features(noun_candidates)
        prediction_scores = self._batch_predicting_nouns(noun_candidates, min_noun_score)
        if self.logpath:
            with open(self.logpath + "_prediction_score.log", "w", encoding="utf-8") as f:
                f.write("noun score frequency\n")
                f.writelines(
                    (
                        f"{word} {score[0]} {score[1]}\n"
                        for word, score in sorted(prediction_scores.items(), key=lambda x: -x[1][1])
                    )
                )
        if self.extract_compound:
            candidates = {l: sum(rdict.values()) for l, rdict in self.lrgraph._lr.items() if len(l) >= 4}
            compounds = self.extract_compounds(candidates, prediction_scores, min_noun_score)
        else:
            compounds = {}
        nouns = {noun: score for noun, score in prediction_scores.items() if score[1] >= min_noun_score}
        nouns.update(compounds)
        nouns = {noun: score for noun, score in nouns.items() if score[0] >= min_noun_frequency}
        nouns = self._post_processing(nouns, prediction_scores, compounds)
        if self.verbose:
            _log(
                f"[Noun Extractor] {len(nouns)} nouns ({len(compounds)} compounds) with min frequency={min_noun_frequency}",
                flush=True,
            )
            _log("[Noun Extractor] flushing ... ", flush=True, end="")
        self._check_covered_eojeols(nouns)
        self._nouns = nouns
        if reset_lrgraph:
            self.lrgraph.reset_lrgraph()
        nouns_ = {noun: NounScore(score[0], score[1]) for noun, score in nouns.items()}
        return nouns_

    def _get_nonempty_features(self, word, features):
        return [
            r
            for r, _ in features
            if r in self._pos_features
            and (not self._exist_longer_pos(word, r))
            or (r in self._neg_features and (not self._exist_longer_neg(word, r)))
        ]

    def _exist_longer_pos(self, word, r):
        for e in range(len(word) - 1, -1, -1):
            if word[e:] + r in self._pos_features:
                return True
        return False

    def _exist_longer_neg(self, word, r):
        for e in range(len(word) - 1, -1, -1):
            if word[e:] + r in self._neg_features:
                return True
        return False

    def predict(self, word, min_noun_score=0.3, debug=False):
        if self.lrgraph is None:
            raise ValueError("train() must be called before extraction")
        features = self.lrgraph.get_r(word, -1)
        pos, common, neg, unk, end = self._predict(word, features)
        base = pos + neg
        score = 0 if base == 0 else (pos - neg) / base
        support = pos + end + common if score >= min_noun_score else neg + end + common
        features_ = self._get_nonempty_features(word, features)
        n_features_ = len(features_)
        if debug:
            _log(f"pos={pos}, common={common}, neg={neg}, unk={unk}, end={end}, n_features_={n_features_}")
        if n_features_ > self.min_num_of_features:
            return (support, score)
        else:
            sum_ = pos + common + neg + unk + end
            if sum_ == 0:
                return (support, 0)
            if end > self.max_frequency_when_noun_is_eojeol and pos >= neg:
                return (support, score)
            if (common > 0 or pos > 0) and end / sum_ >= 0.3 and (common >= neg):
                support = pos + common + end
                return (support, support / sum_)
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
                return (support, support / sum_)
            return (support, 0)

    def _predict(self, word, features):
        pos, common, neg, unk, end = (0, 0, 0, 0, 0)
        for r, freq in features:
            if r == "":
                end += freq
                continue
            if self._exist_longer_pos(word, r):
                continue
            if self._exist_longer_neg(word, r):
                continue
            if r in self._common_features:
                common += freq
            elif r in self._pos_features:
                pos += freq
            elif r in self._neg_features:
                neg += freq
            else:
                unk += freq
        return (pos, common, neg, unk, end)

    def _noun_candidates_from_positive_features(self, condition=None):

        if self.lrgraph is None:
            raise ValueError("train() must be called before extraction")

        def satisfy(word, e):
            return word[:e] == condition

        N_from_J = {}
        for r in self._pos_features:
            for l, c in self.lrgraph.get_l(r, -1):
                if not condition:
                    N_from_J[l] = N_from_J.get(l, 0) + c
                    continue
                if not satisfy(l, len(condition)):
                    continue
                N_from_J[l] = N_from_J.get(l, 0) + c
        return N_from_J

    def _batch_predicting_nouns(self, noun_candidates, min_noun_score=0.3):
        if self.lrgraph is None:
            raise ValueError("train() must be called before extraction")
        prediction_scores = {}
        n = len(noun_candidates)
        for i, word in enumerate(sorted(noun_candidates, key=lambda x: -len(x))):
            if self.verbose and i % 1000 == 999:
                percentage = "%.3f" % (100 * (i + 1) / n)
                _log(f"\r  -- batch prediction {percentage} % of {n} words", flush=True, end="")
            support, score = self.predict(word, min_noun_score)
            prediction_scores[word] = (support, score)
            if score >= min_noun_score:
                for r, count in self.lrgraph.get_r(word, -1):
                    self.lrgraph.remove_eojeol(word + r, count)
        if self.verbose:
            _log(f"\r[Noun Extractor] batch prediction was completed for {n} words", flush=True)
        return prediction_scores

    def extract_compounds(self, candidates, prediction_scores, min_noun_score=0.3):
        if self.lrgraph is None:
            raise ValueError("train() must be called before extraction")
        noun_scores = {
            noun: len(noun) for noun, score in prediction_scores.items() if score[1] > min_noun_score and len(noun) > 1
        }
        self._compound_decomposer = MaxScoreTokenizer(scores=noun_scores)
        candidates = {
            l: rdict.get("", 0) for l, rdict in self.lrgraph._lr_origin.items() if len(l) >= 4 and l not in noun_scores
        }
        n = len(candidates)
        compounds_scores = {}
        compounds_counts = {}
        compounds_components = {}
        for i, (word, count) in enumerate(sorted(candidates.items(), key=lambda x: -len(x[0]))):
            if self.verbose and i % 1000 == 999:
                percentage = "%.2f" % (100 * i / n)
                _log(f"\r  -- check compound {percentage} %", flush=True, end="")
            tokens = self._compound_decomposer.tokenize(word, flatten=False)[0]
            compound_parts = self._parse_compound(tokens)
            if compound_parts:
                noun = "".join(compound_parts)
                compounds_components[noun] = compound_parts
                compound_score = max((prediction_scores.get(t, (0, 0))[1] for t in compound_parts))
                compounds_scores[noun] = max(compounds_scores.get(noun, 0), compound_score)
                compounds_counts[noun] = compounds_counts.get(noun, 0) + count
                for e in range(2, len(word)):
                    subword = word[:e]
                    if subword not in candidates:
                        continue
                    candidates[subword] = candidates.get(subword, 0) - count
                self.lrgraph.remove_eojeol(word)
        if self.verbose:
            _log(f"\r[Noun Extractor] checked compounds. discovered {len(compounds_scores)} compounds")
        compounds = {noun: (score, compounds_counts.get(noun, 0)) for noun, score in compounds_scores.items()}
        self._compounds_components = compounds_components
        return compounds

    def decompose_compound(self, word):
        tokens = self._compound_decomposer.tokenize(word, flatten=False)[0]
        compound_parts = self._parse_compound(tokens)
        return (word,) if not compound_parts else compound_parts

    def _parse_compound(self, tokens):
        """Check Noun* or Noun*Josa"""
        for token in tokens[:-1]:
            if token[3] <= 0:
                return None
        if len(tokens) >= 3 and tokens[-1][0] in self._pos_features:
            return tuple((t[0] for t in tokens[:-1]))
        if tokens[-1][3] > 0:
            return tuple((t[0] for t in tokens))
        return None

    def _post_processing(self, nouns, prediction_scores, compounds):

        if self.lrgraph is None:
            raise ValueError("train() must be called before extraction")

        def print_status(method, nouns, removals):
            n_after = len(nouns)
            n_before = n_after + len(removals)
            _log(f"[Noun Extractor] postprocessing {method} : {n_before} -> {n_after}")

        logpath = self.logpath + "_postprocessing.log" if self.logpath else None
        if logpath:
            with open(logpath, "w", encoding="utf-8") as f:
                f.write("")
        for method in self.postprocessing:
            if method == "detaching_features":
                logheader = "## Ignore noun candidates from detaching pos features\n"
                nouns, removals = detaching_features(nouns, self._pos_features, logpath, logheader)
                if self.verbose:
                    print_status(method, nouns, removals)
            elif method == "ignore_features":
                features = {f for f in self._pos_features}
                features.update(self._common_features)
                nouns, removals = ignore_features(nouns, features, logpath)
                if self.verbose:
                    print_status(method, nouns, removals)
            elif method == "ignore_NJ":
                nouns, removals = check_N_is_NJ(nouns, self.lrgraph, logpath=logpath)
                if self.verbose:
                    print_status(method, nouns, removals)
        return nouns

    def _check_covered_eojeols(self, nouns):
        if self.lrgraph is None:
            raise ValueError("train() must be called before extraction")
        self.lrgraph.reset_lrgraph()
        noun_candidates = self._noun_candidates_from_positive_features()
        n = len(noun_candidates)
        for i, word in enumerate(sorted(noun_candidates, key=lambda x: -len(x))):
            if self.verbose and i % 1000 == 999:
                percentage = "%.3f" % (100 * (i + 1) / n)
                _log(f"\r[Noun Extractor] flushing ...  {percentage} %", flush=True, end="")
            if word not in nouns:
                continue
            for r, count in self.lrgraph.get_r(word, -1):
                if len(word) > 1 or r == "" or r in self._pos_features or (r in self._common_features):
                    self.lrgraph.remove_eojeol(word + r, count)
                    self._num_of_covered_eojeols += count
        if self.verbose:
            _log(
                "\r[Noun Extractor] flushing was done. mem={} Gb{}".format("%.3f" % get_process_memory(), " " * 20), flush=True
            )
            coverage = "%.2f" % (100 * self._num_of_covered_eojeols / self._num_of_eojeols)
            _log(f"[Noun Extractor] {coverage} % eojeols are covered", flush=True)
