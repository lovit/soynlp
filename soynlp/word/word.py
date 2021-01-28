import math
from collections import defaultdict
from dataclasses import dataclass
from tqdm import tqdm

from soynlp.utils import get_process_memory


class WordExtractor:
    def __init__(self, max_l_length=10, max_r_length=6, verbose=True):
        self.max_l_length = max_l_length
        self.max_r_length = max_r_length

        self.L = {}
        self.R = {}
        self.prev_L = {}
        self.R_next = {}

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
        min_brancingentropy_leftside=0.0,
        min_brancingentropy_rightside=0.0,
        min_accessorvariety_leftside=1,
        min_accessorvariety_rightside=1,
        prune_per_lines=-1,
        remove_subwords=False,
    ):
        L, R, prev_L, R_next = initialize_counters(
            self.L, self.R, self.prev_L, self.R_next, cumulate
        )
        L, R, prev_L, R_next = count_substrings(
            train_data=train_data,
            L=L,
            R=R,
            prev_L=prev_L,
            R_next=R_next,
            max_left_length=self.max_left_length,
            max_right_length=self.max_right_length,
            min_frequency=min_frequency,
            prune_per_lines=prune_per_lines,
            cohesion_only=cohesion_only,
            verbose=self.verbose,
        )
        self.L, self.R, self.prev_L, self.R_next = L, R, prev_L, R_next


def print_message(message):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[WordExtractor] {now}, mem={get_process_memory():.4} GB : {message}")


def initializer_counters(L, R, prev_L, R_next, cumulate: bool):
    if cumulate:
        return tuple(defaultdict(int, d) for d in [L, R, prev_L, R_next])
    return tuple(defaultdict(int) for _ in range(4))


def prune_counter(counter, min_count):
    return defaultdict(
        int, {key: count for key, count in counter.items() if count >= min_count}
    )


def count_substrings(
    train_data,
    L,
    R,
    prev_L,
    R_next,
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
            L, R, prev_L, R_next = [prune_counter(d, 2) for d in [L, R, prev_L, R_next]]

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
            for i in range(1, min(max_left_length, n) + 1):
                prev_L[f"{prev_char} {word[:i]}"] += 1
            for i in range(1, min(max_right_length + 1, n)):
                R_next[f"{word[-i:]} {next_char}"] += 1

    L = dict(prune_counter(L, min_frequency))
    R = dict(prune_counter(R, min_frequency))
    prev_L = dict(prune_counter(prev_L, min_frequency))
    R_next = dict(prune_counter(R_next, min_frequency))
    return L, R, prev_L, R_next


def get_entropy(collection_of_numbers):
    """
    Args:
        collection_of_numbers (collection of number)

    Returns:
        entropy (float)

    Examples::
        >>> get_entropy([3, 4, 3])
        $ 1.0888999753452238

        >>> get_entropy([100, 1, 1])
        $ 0.11010008192339721
    """
    if not collection_of_numbers:
        return 0.0
    total = sum(collection_of_numbers)
    entropy = 0
    for number in collection_of_numbers:
        prob = float(number) / total
        entropy += prob * math.log(prob)
    return -1 * entropy


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class CohesionScore:
    """
    Args:
        subword (str) : Substring of word
        leftside (float) :
            Forward-direction cohesion score
                cohesion_l('abc') = (#(abc-) / #(a-))^(1/2)
                #(a-) is frequency of subsrting `a` which positions on left of words in a corpus
                #(abc-) is frequency of substring `abc` which positions on left of words in a corpus
        rightside (float) :
            Backward-direction cohesion score
                cohesion_r('cba') = (#(-cba) / #(-a))^(1/2)
                #(-a) is frequency of subsrting `a` which positions on right of words in a corpus
                #(-cba) is frequency of substring `cba` which positions on right of words in a corpus

    Examples:
        # 토크나이저 + 는/의/를 등 다양한 조사가 등장하여 `토크나이저` 보다 점수가 낮음
        >>> cs = CohesionScore("토크나이저는", 0.35, 0.15)

        # 어절의 왼쪽에 `크나이저는` 이란 subword 가 등장한 적이 없음
        >>> cs = CohesionScore("크나이저는", 0.02, 0.13)

        # `토크나이저` 단독으로 어절로 이용되는 경우가 있으면 `leftside`, `rightside`  모두 점수가 높음
        >>> cs = CohesionScore("토크나이저", 0.8, 0.5)

        >>> cs
        $ CohesionScore(subword='토크나이저', leftside=0.8, rightside=0.5)

        >>> cs.leftside
        $ 0.8
    """

    subword: str
    leftside: float
    rightside: float


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class BranchingEntropy:
    """
    Args:
        subword (str) : Substring of word
        leftside (float) :
            Backward-direction, left-side Branching Entropy
                be_l(abc) = entropy of (? + abc)
        rightside (float) :
            Forward-direction, right-side Branching Entropy
                be_r(abc) = entropy of (abc + ?)

    Examples:
        # 토크나이 + 저 로 오른쪽에 한 종류의 글자가 등장. Entropy = 0
        >>> be = BranchingEntropy("토크나이", 1.53, 0)

        # 토 + 크나이저 로 왼쪽에 한 종류의 글자가 등장. Entropy = 0
        >>> be = BranchingEntropy("크나이저", 0, 2.25)

        # 토크나이저 + 는/의/를 등 다양한 조사가 다양한 확률로 등장
        >>> be = BranchingEntropy("토크나이저", 1.53, 2.25)

        >>> be
        $ BranchingEntropy(subword='토크나이저', leftside=1.53, rightside=2.25)

        >>> be.leftside
        $ 1.53
    """

    subword: str
    leftside: float
    rightside: float


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class AccessorVariety:
    """
    Args:
        subword (str) : Substring of word
        leftside (float) :
            Backward-direction, left-side Accessor Variety
                av_l(abc) = num of unique characters in (? + abc)
        rightside (float) :
            Forward-direction, right-side Accessor Variety
                av_r(abc) = num of unique characters in (abc + ?)

    Examples:
        # 토크나이 + 저 로 오른쪽에 한 종류의 글자가 등장
        >>> av = AccessorVariety("토크나이", 15, 1)

        # 토 + 크나이저 로 왼쪽에 한 종류의 글자가 등장
        >>> av = AccessorVariety("크나이저", 1, 9)

        # 토크나이저 + 는/의/를 등 다양한 조사가 출현
        >>> av = AccessorVariety("토크나이저", 15, 9)

        >>> av
        $ AccessorVariety(subword='토크나이저', leftside=0.8, rightside=0.5)

        >>> av.leftside
        $ 0.8
    """

    subword: str
    leftside: float
    rightside: float
