import math
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from tqdm import tqdm

from soynlp.utils import DoublespaceLineCorpus, get_process_memory


class WordExtractor:
    def __init__(self, max_l_length=10, max_r_length=6, verbose=True, R_suffix="▁"):
        self.max_l_length = max_l_length
        self.max_r_length = max_r_length
        self.verbose = verbose
        self.R_suffix = R_suffix

        self.L = {}
        self.R = {}
        self.prev_sub = {}
        self.sub_next = {}

    @property
    def is_trained(self):
        return self.L and self.R

    def extract(
        self,
        train_data=None,
        cumulate=True,
        extract_cohesion_only=False,
        min_frequency=5,
        min_cohesion_leftside=0.05,
        min_cohesion_rightside=0.0,
        min_brancingentropy_leftside=0.1,
        min_brancingentropy_rightside=0.1,
        min_accessorvariety_leftside=2,
        min_accessorvariety_rightside=2,
        prune_per_lines=-1,
        remove_subwords=False,
    ):
        if isinstance(train_data, str) and os.path.exists(train_data):
            train_data = DoublespaceLineCorpus(train_data)
        L, R, prev_sub, sub_next = initialize_counters(self.L, self.R, self.prev_sub, self.sub_next, cumulate)
        self.L, self.R, self.prev_sub, self.sub_next = count_substrings(
            train_data=train_data,
            L=L,
            R=R,
            prev_sub=prev_sub,
            sub_next=sub_next,
            max_left_length=self.max_l_length,
            max_right_length=self.max_r_length,
            min_frequency=min_frequency,
            prune_per_lines=prune_per_lines,
            cohesion_only=extract_cohesion_only,
            verbose=self.verbose,
        )
        self.L, self.R, self.prev_sub, self.sub_next = L, R, prev_sub, sub_next
        cohesions = calculate_cohesion_batch(
            L=self.L,
            R=self.R,
            min_cohesion_leftside=min_cohesion_leftside,
            min_cohesion_rightside=min_cohesion_rightside,
            verbose=self.verbose,
        )
        if extract_cohesion_only:
            return {"cohesion": cohesions}
        av, be = calculate_branching_entropy_accessor_variety_batch(
            L=self.L,
            R=self.R,
            prev_sub=self.prev_sub,
            sub_next=self.sub_next,
            min_brancingentropy_leftside=min_brancingentropy_leftside,
            min_brancingentropy_rightside=min_brancingentropy_rightside,
            min_accessorvariety_leftside=min_accessorvariety_leftside,
            min_accessorvariety_rightside=min_accessorvariety_rightside,
            verbose=self.verbose,
            R_suffix=self.R_suffix,
        )
        return {"cohesion": cohesions, "accessor_variety": av, "branching_entropy": be}


def print_message(message):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[WordExtractor] {now}, mem={get_process_memory():.4} GB : {message}")


def initialize_counters(L, R, prev_sub, sub_next, cumulate: bool):
    if cumulate:
        return tuple(defaultdict(int, d) for d in [L, R, prev_sub, sub_next])
    return tuple(defaultdict(int) for _ in range(4))


def prune_counter(counter, min_count):
    return defaultdict(int, {key: count for key, count in counter.items() if count >= min_count})


def count_substrings(
    train_data,
    L,
    R,
    prev_sub,
    sub_next,
    max_left_length,
    max_right_length,
    min_frequency,
    prune_per_lines,
    cohesion_only,
    verbose,
):
    if not verbose:
        train_iterator = train_data
    else:
        if hasattr(train_data, "__len__"):
            total = len(train_data)
        else:
            total = None
        desc = "[WordExtractor] counting subwords"
        train_iterator = tqdm(train_data, desc=desc, total=total)

    for i_line, line in enumerate(train_iterator):
        # prune
        if (prune_per_lines > 0) and (i_line % prune_per_lines == 0):
            L, R, prev_sub, sub_next = [prune_counter(d, 2) for d in [L, R, prev_sub, sub_next]]

        words = line.split()

        # cohesion only
        for word in words:
            if (not word) or (len(word) <= 1):
                continue
            n = len(word)
            for i in range(1, min(max_left_length, n) + 1):
                L[word[:i]] += 1
            for i in range(1, min(max_right_length + 1, n)):
                R[word[-i:]] += 1

        # branching entropy & accessor variety
        if (cohesion_only) or (len(words) <= 1):
            continue

        prev_words = [words[-1]] + words[:-1]
        next_words = words[1:] + [words[0]]
        for prev_word, word, next_word in zip(prev_words, words, next_words):
            prev_char = prev_word[-1]
            next_char = next_word[0]
            n = len(word)
            if n <= max_left_length:
                sub_next[(word, next_char)] += 1
            for i in range(1, min(max_left_length, n) + 1):
                prev_sub[(prev_char, word[:i])] += 1
            for i in range(1, min(max_right_length + 1, n)):
                sub_next[(word[-i:], next_char)] += 1

    L = dict(prune_counter(L, min_frequency))
    R = dict(prune_counter(R, min_frequency))
    prev_sub = dict(prune_counter(prev_sub, min_frequency))
    sub_next = dict(prune_counter(sub_next, min_frequency))
    return L, R, prev_sub, sub_next


def calculate_cohesion(word: str, L: dict, R: dict):
    n = len(word)
    if n <= 1:
        return (0, 0)
    inv_p = 1 / (n - 1)
    l_nominator, r_nominator = L.get(word, 0), R.get(word, 0)
    l_denominator, r_denominator = L.get(word[0], 0), R.get(word[-1], 0)
    l_score, r_score = 0, 0
    if l_denominator > 0:
        l_score = np.power((l_nominator / l_denominator), inv_p)
    if r_denominator > 0:
        r_score = np.power((r_nominator / r_denominator), inv_p)
    return (l_score, r_score)


def calculate_cohesion_batch(
    L: dict,
    R: dict,
    min_cohesion_leftside: float,
    min_cohesion_rightside: float,
    verbose: bool = True,
):
    words = set(L).union(set(R))
    if verbose:
        desc = "[WordExtractor] calculating cohesions"
        word_iterator = tqdm(words, desc=desc, total=len(words))
    else:
        word_iterator = words
    extracteds = {}
    for word in word_iterator:
        l_score, r_score = calculate_cohesion(word, L, R)
        if (l_score < min_cohesion_leftside) or (r_score < min_cohesion_rightside):
            continue
        extracteds[word] = CohesionScore(word, l_score, r_score)
    return extracteds


def get_entropy(collection_of_numbers):
    if not collection_of_numbers:
        return 0.0
    total = sum(collection_of_numbers)
    entropy = 0
    for number in collection_of_numbers:
        prob = float(number) / total
        entropy += prob * math.log(prob)
    return -1 * entropy


def calculate_branching_entropy_accessor_variety_batch(
    L: dict,
    R: dict,
    prev_sub: dict,
    sub_next: dict,
    min_brancingentropy_leftside: float,
    min_brancingentropy_rightside: float,
    min_accessorvariety_leftside: int,
    min_accessorvariety_rightside: int,
    verbose: bool = True,
    R_suffix="▁",
):
    l_groupby_len, r_groupby_len = defaultdict(lambda: {}), defaultdict(lambda: {})
    for l, count in L.items():  # noqa: E741
        l_groupby_len[len(l)][l] = count
    for r, count in R.items():
        r_groupby_len[len(r)][r] = count

    total_l, total_r = len(L), len(R)
    be_l, be_r, av_l, av_r = {}, {}, {}, {}

    offset = 0
    max_l_length = max(l_groupby_len)
    for l_len, l_count in sorted(l_groupby_len.items()):
        if l_len == 1:
            continue
        prev_dict = defaultdict(lambda: {})
        for (prev, sub), count in prev_sub.items():
            if len(sub) == l_len:
                prev_dict[sub][prev] = count
        extensions_left, extensions_right = defaultdict(lambda: []), defaultdict(lambda: [])
        if verbose:
            l_count_iterator = tqdm(
                l_count.items(),
                desc="[WordExtractor] calculating AV/BE L",
                initial=offset,
                total=total_l,
                leave=(l_len == max_l_length),
            )
        else:
            l_count_iterator = l_count.items()
        for l, count in l_count_iterator:  # noqa: E741
            for sub, count in prev_dict.get(l, {}).items():
                extensions_left[l].append(count)
            extensions_right[l[:-1]].append(count)
        for l, counts in extensions_left.items():  # noqa: E741
            be_l[l] = get_entropy(counts)
            av_l[l] = len(counts)
        for l, counts in extensions_right.items():  # noqa: E741
            be_r[l] = get_entropy(counts)
            av_r[l] = len(counts)
        offset += len(l_count)

    offset = 0
    max_r_length = max(r_groupby_len)
    for r_len, r_count in sorted(r_groupby_len.items()):
        if r_len == 1:
            continue
        prev_dict = defaultdict(lambda: {})
        for (sub, next_char), count in sub_next.items():
            if len(sub) == r_len:
                prev_dict[sub][next_char] = count
        extensions_left, extensions_right = defaultdict(lambda: []), defaultdict(lambda: [])
        if verbose:
            r_count_iterator = tqdm(
                r_count.items(),
                desc="[WordExtractor] calculating AV/BE R",
                initial=offset,
                total=total_r,
                leave=(r_len == max_r_length),
            )
        else:
            r_count_iterator = r_count.items()
        for r, count in r_count_iterator:
            for sub, count in prev_dict.get(r, {}).items():
                extensions_right[r].append(count)
            extensions_left[r[1:]].append(count)
        for r, counts in extensions_left.items():
            be_l[f"{r}{R_suffix}"] = get_entropy(counts)
            av_l[f"{r}{R_suffix}"] = len(counts)
        for r, counts in extensions_right.items():
            be_r[f"{r}{R_suffix}"] = get_entropy(counts)
            av_r[f"{r}{R_suffix}"] = len(counts)
        offset += len(r_count)

    av, be = {}, {}
    for term in be_l:
        if (av_l.get(term, 0) >= min_accessorvariety_leftside) and (av_r.get(term, 0) >= min_accessorvariety_rightside):
            av[term] = AccessorVariety(term, av_l.get(term, 0), av_r.get(term, 0))
        if (be_l.get(term, 0) >= min_brancingentropy_leftside) and (be_r.get(term, 0) >= min_brancingentropy_rightside):
            be[term] = BranchingEntropy(term, be_l.get(term, 0.0), be_r.get(term, 0.0))
    return av, be


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class CohesionScore:
    subword: str
    leftside: float
    rightside: float


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class BranchingEntropy:
    subword: str
    leftside: float
    rightside: float


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class AccessorVariety:
    subword: str
    leftside: float
    rightside: float
