import math
from dataclasses import dataclass


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
