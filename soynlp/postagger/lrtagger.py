import dataclasses
import logging
from collections import OrderedDict
from dataclasses import dataclass
from math import log

from soynlp.tokenizer import MaxScoreTokenizer

from .dictionary import Dictionary

logger = logging.getLogger(__name__)

default_profile = OrderedDict(
    [
        ("cohesion_l", 0.5),
        ("droprate_l", 0.5),
        ("log_count_l", 0.1),
        ("prob_l2r", 0.1),
        ("log_count_l2r", 0.1),
        ("known_LR", 1.0),
        ("R_is_syllable", -0.1),
        ("log_length", 0.5),
    ]
)


@dataclass(slots=True)
class ScoreTable:
    cohesion_l: float
    droprate_l: float
    log_count_l: float
    prob_l2r: float
    log_count_l2r: float
    known_LR: float
    R_is_syllable: float
    log_length: float


@dataclass(slots=True)
class Table:
    L: tuple
    R: tuple
    begin: int
    end: int
    length: int
    lr_prop: float
    lr_count: int
    cohesion_l: float
    droprate_l: float
    lcount: int


class LREvaluator:
    def __init__(self, profile: OrderedDict | None = None):
        self.profile = profile if profile else default_profile

    def evaluate(self, candidates: list, preference: dict | None = None) -> list[tuple]:
        scores = []
        for c in candidates:
            score = self._evaluate(
                self.make_scoretable(
                    c.L[0], c.L[1], c.R[0], c.R[1], c.cohesion_l, c.droprate_l, c.lcount, c.lr_prop, c.lr_count, c.length
                )
            )
            if preference:
                if c.L[1] and c.L[1] in preference:
                    score += preference.get(c.L[1], {}).get(c.L[0], 0)
                if c.R[1] and c.R[1] in preference:
                    score += preference.get(c.R[1], {}).get(c.R[0], 0)
            scores.append((c, score))
        return sorted(scores, key=lambda x: -x[-1])

    def make_scoretable(
        self,
        l: str,
        pos_l,
        r: str,
        pos_r,
        cohesion: float,
        droprate: float,
        lcount: int,
        lr_prop: float,
        lr_count: int,
        len_LR: int,
    ) -> ScoreTable:
        return ScoreTable(
            cohesion,
            droprate,
            log(lcount + 1),
            lr_prop,
            log(lr_count + 1),
            1 if (pos_l and pos_r) else 0,
            1 if len(r) == 1 else 0,
            log(len_LR),
        )

    def _evaluate(self, scoretable: ScoreTable) -> float:
        return sum(score * self.profile.get(field, 0) for field, score in dataclasses.asdict(scoretable).items())  # type: ignore[arg-type]


class LRMaxScoreTagger:
    def __init__(
        self,
        domain_dictionary_folders: list[str] | str | None = None,
        use_base_dictionary: bool = True,
        dictionary_word_mincount: int = 3,
        evaluator: LREvaluator | None = None,
        sents: list[str] | None = None,
        lrgraph: dict | None = None,
        lrgraph_lmax: int = 12,
        lrgraph_rmax: int = 8,
        base_tokenizer: object = None,
        preference: dict | None = None,
        verbose: bool = False,
    ):
        self.dictionary = Dictionary(domain_dictionary_folders, use_base_dictionary, dictionary_word_mincount, verbose=verbose)  # type: ignore[arg-type]
        self.evaluator = evaluator if evaluator else LREvaluator()
        self.preference = preference if preference else {}
        self.lrgraph = lrgraph if lrgraph else {}

        if (not self.lrgraph) and sents:
            self.lrgraph = self._build_lrgraph(sents, lrgraph_lmax, lrgraph_rmax)

        self.lrgraph_norm, self.lcount, self.cohesion_l, self.droprate_l = self._initialize_scores(self.lrgraph)

        self.base_tokenizer = base_tokenizer if base_tokenizer else (lambda x: x.split())
        if not base_tokenizer:
            try:
                self.base_tokenizer = MaxScoreTokenizer(scores=self.cohesion_l)
            except Exception as e:
                logger.warning("MaxScoreTokenizer(cohesion) exception: %s", e)

    def _build_lrgraph(self, sents, lmax: int = 12, rmax: int = 8) -> dict:
        from collections import Counter, defaultdict

        eojeols = Counter(eojeol for sent in sents for eojeol in sent.split() if eojeol)
        lrgraph: dict = defaultdict(lambda: defaultdict(int))
        for eojeol, count in eojeols.items():
            n = len(eojeol)
            for i in range(1, min(n, lmax) + 1):
                l, r = eojeol[:i], eojeol[i:]
                if len(r) > rmax:
                    continue
                lrgraph[l][r] += count
        return lrgraph

    def _initialize_scores(self, lrgraph: dict) -> tuple[dict, dict, dict, dict]:
        def to_counter(dd: dict) -> dict:
            return {k: sum(d.values()) for k, d in dd.items()}

        def to_normalized_graph(dd: dict) -> dict:
            normed = {}
            for k, d in dd.items():
                sum_ = sum(d.values())
                normed[k] = {k1: c / sum_ for k1, c in d.items()}
            return normed

        lrgraph_norm = to_normalized_graph(lrgraph)
        lcount = to_counter(lrgraph)
        cohesion_l = {w: pow(c / lcount[w[0]], 1 / (len(w) - 1)) for w, c in lcount.items() if len(w) > 1}
        droprate_l = {w: c / lcount[w[:-1]] for w, c in lcount.items() if len(w) > 1 and w[:-1] in lcount}

        return lrgraph_norm, lcount, cohesion_l, droprate_l

    def pos(self, sent: str, flatten: bool = True, debug: bool = False) -> list:
        sent_ = [self._pos(eojeol, debug) for eojeol in sent.split() if eojeol]
        if flatten:
            sent_ = [word for words in sent_ for word in words]
        return sent_

    def _pos(self, eojeol: str, debug: bool = False) -> list:
        candidates = self._initialize(eojeol)
        scores = self._scoring(candidates)
        best = self._find_best(scores)
        if best:
            post = self._postprocessing(eojeol, best)
        else:
            post = self._base_tokenizing_subword(eojeol, 0)

        if not debug:
            post = [w for lr in post for w in ([lr.L, lr.R] if isinstance(lr, Table) else lr[:2]) if w[0]]
        return post

    def _initialize(self, t: str) -> list:
        candidates = self._initialize_L(t)
        return self._initialize_LR(t, candidates)

    def _initialize_L(self, t: str) -> list:
        n = len(t)
        candidates = []
        for b in range(n):
            for e in range(b + 2, min(n, b + self.dictionary._lmax) + 1):  # type: ignore[operator]
                l = t[b:e]
                l_pos = self.dictionary.pos_L(l)  # type: ignore[attr-defined]
                if not l_pos:
                    continue
                candidates.append([l, l_pos, b, e, e - b])

        candidates = self._remove_l_subsets(candidates)
        return sorted(candidates, key=lambda x: x[2])

    def _remove_l_subsets(self, candidates: list) -> list:
        candidates_ = []
        for pos in ["Noun", "Verb", "Adjective", "Adverb", "Exclamation"]:
            sorted_ = sorted(filter(lambda x: x[1] == pos, candidates), key=lambda x: -x[4])
            while sorted_:
                candidates_.append(sorted_.pop(0))
                b, e = candidates_[-1][2], candidates_[-1][3]
                removals = [i for i, c in enumerate(sorted_) if b <= c[2] and e >= c[3]]
                for idx in reversed(removals):
                    del sorted_[idx]
        return candidates_

    def _initialize_LR(self, t: str, candidates: list, threshold_prop: float = 0.001, threshold_count: int = 2) -> list:
        n = len(t)
        expanded = []

        for l, pos, b, e, len_l in candidates:
            for len_r in range(min(self.dictionary._rmax, n - e) + 1):  # type: ignore[operator]
                r = t[e : e + len_r]
                lr_prop = self.lrgraph_norm.get(l, {}).get(r, 0)
                lr_count = self.lrgraph.get(l, {}).get(r, 0)

                if r and ((lr_prop <= threshold_prop) or (lr_count <= threshold_count)):
                    continue

                expanded.append(
                    [
                        (l, pos),
                        (r, None if not r else self.dictionary.pos_R(r)),  # type: ignore[attr-defined]
                        b,
                        e,
                        e + len_r,
                        len_r,
                        len_l + len_r,
                        lr_prop,
                        lr_count,
                    ]
                )

        expanded = self._remove_r_subsets(expanded)
        return sorted(expanded, key=lambda x: x[2])

    def _remove_r_subsets(self, expanded: list) -> list:
        expanded_ = []
        for pos in ["Josa", "Verb", "Adjective", None]:
            sorted_ = sorted(filter(lambda x: x[1][1] == pos, expanded), key=lambda x: -x[5])
            while sorted_:
                expanded_.append(sorted_.pop(0))
                b, e = expanded_[-1][3], expanded_[-1][4]
                removals = [i for i, c in enumerate(sorted_) if b <= c[3] and e >= c[4]]
                for idx in reversed(removals):
                    del sorted_[idx]
        return [[L, R, p0, p2, len_LR, prop, count] for L, R, p0, p1, p2, len_R, len_LR, prop, count in expanded_]

    def _scoring(self, candidates: list) -> list:
        candidates = [self._to_table(c) for c in candidates]
        return self.evaluator.evaluate(candidates, self.preference if self.preference else None)

    def _to_table(self, c: list) -> Table:
        return Table(
            c[0],
            c[1],
            c[2],
            c[3],
            c[4],
            c[5],
            c[6],
            self.cohesion_l.get(c[0][0], 0),
            self.droprate_l.get(c[0][0], 0),
            self.lcount.get(c[0][0], 0),
        )

    def _find_best(self, scores: list) -> list:
        best = []
        sorted_ = sorted(scores, key=lambda x: -x[-1])
        while sorted_:
            best.append(sorted_.pop(0)[0])
            b, e = best[-1].begin, best[-1].end
            removals = [i for i, (c, _) in enumerate(sorted_) if b < c.end and e > c.begin]
            for idx in reversed(removals):
                del sorted_[idx]
        return sorted(best, key=lambda x: x.begin)

    def _postprocessing(self, t: str, words: list) -> list:
        n = len(t)
        adds = []
        if words and words[0].begin > 0:
            adds += self._add_first_subword(t, words)
        if words and words[-1].end < n:
            adds += self._add_last_subword(t, words, n)
        adds += self._add_inter_subwords(t, words)
        post = list(words) + [self._to_table(a) for a in adds]
        return sorted(post, key=lambda x: x.begin)

    def _infer_subword_information(self, subword: str) -> tuple:
        pos = self.dictionary.pos_L(subword)  # type: ignore[attr-defined]
        prop = self.lrgraph_norm.get(subword, {}).get("", 0.0)
        count = self.lrgraph.get(subword, {}).get("", 0)
        if not pos:
            pos = self.dictionary.pos_R(subword)  # type: ignore[attr-defined]
        return (pos, prop, count)

    def _add_inter_subwords(self, t: str, words: list) -> list:
        adds = []
        for i, base in enumerate(words[:-1]):
            if base.end == words[i + 1].begin:
                continue
            b = base.end
            e = words[i + 1].begin
            subword = t[b:e]
            adds += self._base_tokenizing_subword(subword, b)
        return adds

    def _add_last_subword(self, t: str, words: list, n: int) -> list:
        b = words[-1].end
        subword = t[b:]
        return self._base_tokenizing_subword(subword, b)

    def _add_first_subword(self, t: str, words: list) -> list:
        e = words[0].begin
        subword = t[0:e]
        return self._base_tokenizing_subword(subword, 0)

    def _base_tokenizing_subword(self, t: str, b: int) -> list:
        subwords = []
        _subwords = self.base_tokenizer.tokenize(t, flatten=False)  # type: ignore[call-arg]
        if not _subwords:
            return []
        for w in _subwords[0]:
            pos, prop, count = self._infer_subword_information(w[0])
            subwords.append([(w[0], pos), ("", None), b + w[1], b + w[2], w[2] - w[1], prop, count, 0.0])
        return subwords

    def add_words_into_dictionary(self, words: set[str] | list[str] | str, tag: str) -> None:
        if tag not in self.dictionary._pos:  # type: ignore[attr-defined]
            raise ValueError(f"{tag} does not exist in base dictionary")
        self.dictionary.add_words(words, tag)  # type: ignore[arg-type]

    def remove_words_from_dictionary(self, words: set[str] | list[str] | str, tag: str) -> None:
        if tag not in self.dictionary._pos:  # type: ignore[attr-defined]
            raise ValueError(f"{tag} does not exist in base dictionary")
        self.dictionary.remove_words(words, tag)  # type: ignore[arg-type]

    def save_domain_dictionary(self, folder: str, head: str | None = None) -> None:
        self.dictionary.save_domain_dictionary(folder, head)  # type: ignore[attr-defined]

    def set_word_preference(self, words: set[str] | list[str] | str, tag: str, preference: int = 10) -> None:
        if isinstance(words, str):
            words = {words}
        preference_table = self.preference.get(tag, {})
        preference_table.update({word: preference for word in words})
        self.preference[tag] = preference_table
